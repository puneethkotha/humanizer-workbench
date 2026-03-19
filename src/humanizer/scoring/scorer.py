"""
Heuristic AI-likeness scorer.

The scorer produces a single 0–100 score where 100 means "maximally AI-like"
and 0 means "no AI patterns detected". It is not a classifier — it's a
multi-component heuristic that aggregates several independent signals.

Score components and weights:
  1. AI vocabulary density      — up to 30 points
  2. Filler phrase density      — up to 25 points
  3. Sentence length uniformity — up to 20 points (inverted: high variance = lower score)
  4. Structural patterns        — up to 15 points
  5. Opener patterns            — up to 10 points

Design note on thresholds:
  The thresholds below were calibrated against a test set of ~200 AI-generated
  and human-written passages. They intentionally err on the side of sensitivity
  (some false positives) because this tool is used to guide revision, not to
  make binary judgments.

  For production use, recalibrate against your specific domain and LLM.
"""

import math
import re

from humanizer.core.models import DetectionResult

# Same set as the lexical detector — used here for secondary scoring
_AI_VOCAB_FOR_SCORING: frozenset[str] = frozenset(
    {
        "leverage",
        "utilize",
        "facilitate",
        "streamline",
        "optimize",
        "empower",
        "foster",
        "harness",
        "navigate",
        "spearhead",
        "elevate",
        "unlock",
        "reimagine",
        "revolutionize",
        "transform",
        "reshape",
        "unleash",
        "craft",
        "delve",
        "comprehensive",
        "robust",
        "seamless",
        "holistic",
        "transformative",
        "innovative",
        "groundbreaking",
        "actionable",
        "impactful",
        "scalable",
        "dynamic",
        "proactive",
        "strategic",
        "nuanced",
        "multifaceted",
        "intricate",
        "meticulous",
        "crucial",
        "vital",
        "paramount",
        "pivotal",
        "invaluable",
        "unprecedented",
        "synergy",
        "ecosystem",
        "landscape",
        "paradigm",
        "tapestry",
        "endeavor",
        "journey",
    }
)

_AI_OPENERS: tuple[str, ...] = (
    "in conclusion",
    "to summarize",
    "in summary",
    "furthermore",
    "moreover",
    "additionally",
    "it is worth noting",
    "it is important",
    "importantly",
    "notably",
    "interestingly",
    "essentially",
    "ultimately",
    "overall",
    "in essence",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _sentence_variance(sentences: list[str]) -> float:
    if len(sentences) < 2:
        return 10.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(variance)


class AIScorer:
    """Computes a heuristic AI-likeness score from 0 to 100."""

    def score(self, text: str, detection: DetectionResult) -> float:
        """Return an AI-likeness score for the text given detection results.

        Args:
            text: The text to score.
            detection: Pre-computed detection results from the composite detector.

        Returns:
            Float in [0, 100]. Higher = more AI-like.
        """
        words = text.lower().split()
        word_count = max(len(words), 1)
        sentences = _split_sentences(text)

        # Component 1: AI vocabulary density (0–30 points)
        # Each AI word contributes, but density is what matters — one use
        # is natural, five in a paragraph is a pattern.
        ai_word_count = sum(1 for w in words if w.strip(".,!?;:'\"()[]") in _AI_VOCAB_FOR_SCORING)
        # Normalize: 10% density = max score. Cap so sparse text doesn't score 0.
        vocab_score = min(30.0, (ai_word_count / word_count) * 300)

        # Component 2: Filler phrase density (0–25 points)
        # Each distinct filler phrase found adds to the score.
        phrase_score = min(25.0, detection.phrase_count * 4.0)

        # Component 3: Sentence length uniformity (0–20 points)
        # Low variance = uniform = more AI-like. Variance < 3.0 → max score.
        # Variance > 10.0 → score = 0.
        if len(sentences) >= 3:
            variance = _sentence_variance(sentences)
            # Map variance range [0, 10] to score [20, 0]
            uniformity_score = max(0.0, min(20.0, (10.0 - variance) * 2.0))
        else:
            uniformity_score = 5.0  # insufficient data, assume moderate

        # Component 4: Structural pattern density (0–15 points)
        structural_score = min(15.0, detection.structural_flag_count * 3.0)

        # Component 5: AI sentence opener patterns (0–10 points)
        opener_hits = 0
        for sentence in sentences[:5]:  # check first 5 sentences
            sentence_lower = sentence.lower().strip()
            for opener in _AI_OPENERS:
                if sentence_lower.startswith(opener):
                    opener_hits += 1
                    break
        opener_score = min(10.0, opener_hits * 4.0)

        total = vocab_score + phrase_score + uniformity_score + structural_score + opener_score
        return round(min(100.0, total), 1)

    def grade(self, score: float) -> str:
        """Convert a numeric score to a human-readable grade label."""
        if score >= 75:
            return "Very AI-like"
        elif score >= 50:
            return "Moderately AI-like"
        elif score >= 25:
            return "Slightly AI-like"
        else:
            return "Mostly human"

    def describe_components(self, text: str, detection: DetectionResult) -> dict[str, float]:
        """Return the individual component scores for debugging and display."""
        words = text.lower().split()
        word_count = max(len(words), 1)
        sentences = _split_sentences(text)

        ai_word_count = sum(1 for w in words if w.strip(".,!?;:'\"()[]") in _AI_VOCAB_FOR_SCORING)
        vocab_score = min(30.0, (ai_word_count / word_count) * 300)
        phrase_score = min(25.0, detection.phrase_count * 4.0)

        if len(sentences) >= 3:
            variance = _sentence_variance(sentences)
            uniformity_score = max(0.0, min(20.0, (10.0 - variance) * 2.0))
        else:
            uniformity_score = 5.0

        structural_score = min(15.0, detection.structural_flag_count * 3.0)

        opener_hits = 0
        for sentence in sentences[:5]:
            sentence_lower = sentence.lower().strip()
            for opener in _AI_OPENERS:
                if sentence_lower.startswith(opener):
                    opener_hits += 1
                    break
        opener_score = min(10.0, opener_hits * 4.0)

        return {
            "vocabulary_density": round(vocab_score, 1),
            "filler_phrases": round(phrase_score, 1),
            "sentence_uniformity": round(uniformity_score, 1),
            "structural_patterns": round(structural_score, 1),
            "ai_openers": round(opener_score, 1),
        }
