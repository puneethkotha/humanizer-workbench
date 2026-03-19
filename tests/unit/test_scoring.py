"""
Unit tests for the heuristic AI-likeness scorer.

Tests verify that:
  1. Heavily AI-patterned text scores high (>= 50).
  2. Clean, human-like text scores low (< 30).
  3. Score is always in [0, 100].
  4. Individual components are computed correctly.
  5. Grade labels match score ranges.
"""

import pytest

from humanizer.detectors.base import CompositeDetector
from humanizer.detectors.lexical import LexicalDetector
from humanizer.detectors.structural import StructuralDetector
from humanizer.scoring.scorer import AIScorer


def detect(text: str):
    detector = CompositeDetector([LexicalDetector(), StructuralDetector()])
    return detector.detect(text)


HEAVY_AI_TEXT = """
In today's fast-paced world, it is worth noting that comprehensive communication
strategies play a crucial role in facilitating organizational success. Furthermore,
leveraging robust and holistic methodologies empowers teams to seamlessly navigate
complex challenges. Moreover, these transformative and innovative approaches enable
stakeholders to optimize performance and streamline workflows across all dimensions.
In conclusion, it is important to note that this paradigm shift represents a paramount
step toward achieving unprecedented synergistic outcomes that will undoubtedly yield
substantial and impactful results across the entire enterprise ecosystem.
""".strip()

CLEAN_HUMAN_TEXT = """
I've been working on this problem for about three months.

Last week I finally figured out what was going wrong. The batch job was silently
dropping records when the input exceeded 10MB. No error, no warning — just fewer
rows in the output than went in.

The fix was a one-line change. Getting there required reading a lot of code I didn't
write and hadn't touched in two years.
""".strip()

MODERATE_AI_TEXT = """
The system uses a microservices architecture to handle different components independently.
Each service communicates through REST APIs and message queues. This approach helps with
scalability — teams can deploy services separately without affecting others.

However, this introduces complexity around data consistency and distributed transactions.
We addressed this using the Saga pattern, which coordinates multi-step transactions
through a series of local transactions and compensating actions.
""".strip()


class TestAIScorer:
    def setup_method(self):
        self.scorer = AIScorer()

    def test_high_ai_text_scores_high(self):
        detection = detect(HEAVY_AI_TEXT)
        score = self.scorer.score(HEAVY_AI_TEXT, detection)
        assert score >= 50, f"Expected score >= 50 for AI text, got {score}"

    def test_clean_text_scores_low(self):
        detection = detect(CLEAN_HUMAN_TEXT)
        score = self.scorer.score(CLEAN_HUMAN_TEXT, detection)
        assert score < 30, f"Expected score < 30 for clean text, got {score}"

    def test_moderate_text_scores_in_middle(self):
        detection = detect(MODERATE_AI_TEXT)
        score = self.scorer.score(MODERATE_AI_TEXT, detection)
        # Moderate text should score between the extremes
        assert score < 60, f"Expected score < 60 for moderate text, got {score}"

    def test_score_is_bounded_0_to_100(self):
        for text in [HEAVY_AI_TEXT, CLEAN_HUMAN_TEXT, MODERATE_AI_TEXT, "", "Hello."]:
            detection = detect(text)
            score = self.scorer.score(text, detection)
            assert 0 <= score <= 100, f"Score {score} out of bounds for text: {text[:50]}"

    def test_score_is_float(self):
        detection = detect(CLEAN_HUMAN_TEXT)
        score = self.scorer.score(CLEAN_HUMAN_TEXT, detection)
        assert isinstance(score, float)

    def test_empty_text_does_not_crash(self):
        detection = detect("")
        score = self.scorer.score("", detection)
        assert 0 <= score <= 100

    def test_single_word_does_not_crash(self):
        text = "Hello."
        detection = detect(text)
        score = self.scorer.score(text, detection)
        assert 0 <= score <= 100

    def test_ai_text_scores_higher_than_clean(self):
        ai_detection = detect(HEAVY_AI_TEXT)
        clean_detection = detect(CLEAN_HUMAN_TEXT)
        ai_score = self.scorer.score(HEAVY_AI_TEXT, ai_detection)
        clean_score = self.scorer.score(CLEAN_HUMAN_TEXT, clean_detection)
        assert ai_score > clean_score, (
            f"AI text should score higher than clean text. AI: {ai_score}, Clean: {clean_score}"
        )


class TestAIScorerGrade:
    def setup_method(self):
        self.scorer = AIScorer()

    @pytest.mark.parametrize(
        "score, expected_grade",
        [
            (80.0, "Very AI-like"),
            (75.0, "Very AI-like"),
            (60.0, "Moderately AI-like"),
            (50.0, "Moderately AI-like"),
            (30.0, "Slightly AI-like"),
            (25.0, "Slightly AI-like"),
            (10.0, "Mostly human"),
            (0.0, "Mostly human"),
        ],
    )
    def test_grade_labels(self, score, expected_grade):
        assert self.scorer.grade(score) == expected_grade


class TestAIScorerComponents:
    def setup_method(self):
        self.scorer = AIScorer()

    def test_describe_components_returns_all_keys(self):
        detection = detect(HEAVY_AI_TEXT)
        components = self.scorer.describe_components(HEAVY_AI_TEXT, detection)
        expected_keys = {
            "vocabulary_density",
            "filler_phrases",
            "sentence_uniformity",
            "structural_patterns",
            "ai_openers",
        }
        assert set(components.keys()) == expected_keys

    def test_component_scores_are_non_negative(self):
        detection = detect(HEAVY_AI_TEXT)
        components = self.scorer.describe_components(HEAVY_AI_TEXT, detection)
        for name, val in components.items():
            assert val >= 0, f"Component {name} has negative score: {val}"

    def test_component_scores_within_max(self):
        detection = detect(HEAVY_AI_TEXT)
        components = self.scorer.describe_components(HEAVY_AI_TEXT, detection)
        max_vals = {
            "vocabulary_density": 30,
            "filler_phrases": 25,
            "sentence_uniformity": 20,
            "structural_patterns": 15,
            "ai_openers": 10,
        }
        for name, val in components.items():
            assert val <= max_vals[name], (
                f"Component {name} score {val} exceeds max {max_vals[name]}"
            )

    def test_ai_text_has_high_vocabulary_density(self):
        detection = detect(HEAVY_AI_TEXT)
        components = self.scorer.describe_components(HEAVY_AI_TEXT, detection)
        assert components["vocabulary_density"] > 5, "Expected high vocabulary density for AI text"

    def test_clean_text_has_low_filler_phrases(self):
        detection = detect(CLEAN_HUMAN_TEXT)
        components = self.scorer.describe_components(CLEAN_HUMAN_TEXT, detection)
        assert components["filler_phrases"] == 0, "Expected zero filler phrases in clean human text"
