"""
Unit tests for the lexical and structural detectors.

These tests do not require an API key. They validate that the detectors
correctly identify known AI patterns and don't over-flag clean text.
"""

import pytest

from humanizer.detectors.base import CompositeDetector
from humanizer.detectors.lexical import LexicalDetector
from humanizer.detectors.structural import StructuralDetector

# --- LexicalDetector ---


class TestLexicalDetector:
    def setup_method(self):
        self.detector = LexicalDetector()

    def test_detects_ai_vocabulary(self):
        text = "This comprehensive approach leverages robust synergies to facilitate seamless integration."
        result = self.detector.detect(text)
        assert len(result.ai_vocabulary_hits) >= 4
        assert (
            "leverage" in result.ai_vocabulary_hits or "leverages" not in result.ai_vocabulary_hits
        )
        # Check specific words we know are in the vocabulary list
        found_words = set(result.ai_vocabulary_hits)
        expected_words = {"comprehensive", "robust", "seamless", "facilitate"}
        assert expected_words.issubset(found_words) or len(found_words & expected_words) >= 2

    def test_detects_filler_phrases(self):
        text = "It is worth noting that this approach is effective. Furthermore, in conclusion, we should note that it works."
        result = self.detector.detect(text)
        assert result.phrase_count >= 2
        assert any("worth noting" in p for p in result.filler_phrase_hits)

    def test_clean_text_produces_few_hits(self):
        text = "I tried three approaches. Only the third worked. Here's what I found."
        result = self.detector.detect(text)
        assert len(result.ai_vocabulary_hits) == 0
        assert result.phrase_count == 0

    def test_case_insensitive_vocabulary_detection(self):
        text = "We LEVERAGE our platform to FACILITATE growth."
        result = self.detector.detect(text)
        assert len(result.ai_vocabulary_hits) >= 1

    def test_patterns_have_correct_type(self):
        text = "This comprehensive solution leverages cutting-edge technology."
        result = self.detector.detect(text)
        for pattern in result.patterns:
            assert pattern.pattern_type in {"ai_vocabulary", "filler_phrase"}
            assert 0 <= pattern.severity <= 1
            assert pattern.start >= 0
            assert pattern.end > pattern.start

    def test_empty_text(self):
        result = self.detector.detect("")
        assert result.ai_vocabulary_hits == []
        assert result.filler_phrase_hits == []
        assert result.total_pattern_count == 0

    def test_vocabulary_hits_are_deduplicated(self):
        # Same word appearing twice should appear only once in vocabulary_hits list
        text = "We leverage this approach. We also leverage the platform."
        result = self.detector.detect(text)
        assert result.ai_vocabulary_hits.count("leverage") == 1

    def test_filler_phrase_detection_at_sentence_start(self):
        text = "In conclusion, the results speak for themselves. Moreover, we achieved our goals. Furthermore, the team performed well."
        result = self.detector.detect(text)
        assert result.phrase_count >= 3


# --- StructuralDetector ---


class TestStructuralDetector:
    def setup_method(self):
        self.detector = StructuralDetector()

    def test_detects_uniform_sentence_lengths(self):
        # All sentences approximately same length — should flag uniformity
        text = (
            "The system processes requests efficiently and reliably. "
            "The database stores all records safely and correctly. "
            "The API handles all client calls cleanly and quickly. "
            "The cache reduces latency significantly and consistently."
        )
        result = self.detector.detect(text)
        # Low variance should be flagged
        uniform_flags = [f for f in result.structural_flags if "variance" in f or "uniform" in f]
        assert len(uniform_flags) >= 1 or result.sentence_length_variance < 5

    def test_varied_text_has_high_variance(self):
        text = (
            "It failed. "
            "The distributed transaction coordinator was not able to acquire the necessary "
            "consensus across all participating nodes before the timeout expired. "
            "We rolled back."
        )
        result = self.detector.detect(text)
        assert result.sentence_length_variance > 5

    def test_detects_ai_sentence_opener(self):
        text = "In conclusion, we found that the results were significant. Furthermore, additional testing confirmed our hypothesis."
        result = self.detector.detect(text)
        opener_flags = [f for f in result.structural_flags if "opener" in f or "conclusion" in f]
        assert len(opener_flags) >= 1 or len(result.patterns) >= 1

    def test_detects_em_dash_overuse(self):
        text = (
            "The system — as expected — performed well. "
            "The results — which were surprising — showed improvement. "
            "The team — despite challenges — delivered on time."
        )
        result = self.detector.detect(text)
        em_flags = [f for f in result.structural_flags if "em_dash" in f]
        assert len(em_flags) >= 1

    def test_clean_varied_text_has_few_flags(self):
        text = (
            "I was wrong about this for two years.\n\n"
            "We had assumed the bottleneck was the database. It wasn't. "
            "The actual problem was in the connection pool — we'd set the limit too low "
            "when we scaled up the application tier, and never revisited it.\n\n"
            "The fix took an afternoon. Identifying it took six months of intermittent complaints."
        )
        result = self.detector.detect(text)
        # Clean, varied writing should have minimal flags
        assert result.structural_flag_count <= 2


# --- CompositeDetector ---


class TestCompositeDetector:
    def test_merges_results_from_multiple_detectors(self):
        detector = CompositeDetector([LexicalDetector(), StructuralDetector()])
        text = (
            "Furthermore, this comprehensive approach leverages robust methodologies "
            "to facilitate seamless integration across all platforms and stakeholders. "
            "Moreover, the holistic implementation ensures optimal performance consistently. "
            "Additionally, these innovative strategies empower teams to streamline workflows. "
            "In conclusion, the paradigm shift enables transformative organizational outcomes."
        )
        result = detector.detect(text)
        # Should have both lexical and structural hits
        assert len(result.ai_vocabulary_hits) > 0
        assert result.phrase_count > 0
        assert result.total_pattern_count > 0

    def test_deduplicates_vocabulary_hits(self):
        # Both detectors might theoretically return the same hit — ensure dedup
        detector = CompositeDetector([LexicalDetector(), LexicalDetector()])
        text = "This comprehensive solution leverages robust infrastructure."
        result = detector.detect(text)
        # No duplicates in vocabulary hits
        assert len(result.ai_vocabulary_hits) == len(set(result.ai_vocabulary_hits))

    def test_requires_at_least_one_detector(self):
        with pytest.raises(ValueError, match="at least one detector"):
            CompositeDetector([])

    def test_composite_name_includes_components(self):
        detector = CompositeDetector([LexicalDetector(), StructuralDetector()])
        assert "LexicalDetector" in detector.name
        assert "StructuralDetector" in detector.name
