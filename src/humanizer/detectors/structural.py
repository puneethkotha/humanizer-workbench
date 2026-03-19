"""
Structural detector: finds AI-like patterns in text organization and rhythm.

Lexical patterns catch individual words and phrases. Structural patterns catch
how the text is *arranged* — sentence length uniformity, excessive parallelism,
formulaic openers, and list-heavy formatting that reads more like a slide deck
than a piece of writing.

Sentence segmentation uses a simple regex that handles most English sentence
endings. It's deliberately not using NLTK or spaCy to avoid heavy dependencies;
for the detection quality we need here, regex is sufficient.
"""

import math
import re

from humanizer.core.models import DetectionResult, PatternMatch
from humanizer.detectors.base import BaseDetector

# Patterns that appear at the start of sentences in AI-generated text
AI_SENTENCE_OPENERS: tuple[str, ...] = (
    "in conclusion,",
    "to summarize,",
    "in summary,",
    "furthermore,",
    "moreover,",
    "additionally,",
    "it is worth noting",
    "it is important",
    "importantly,",
    "notably,",
    "interestingly,",
    "essentially,",
    "ultimately,",
    "overall,",
    "in essence,",
    "as a result,",
    "consequently,",
    "therefore,",
    "thus,",
)

# Regex to split text into sentences. Handles . ! ? as terminators,
# accounts for abbreviations by requiring sentence-start capitalization.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    # Filter empty strings and very short fragments (< 3 words)
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _sentence_length_variance(sentences: list[str]) -> float:
    """Compute standard deviation of sentence word counts.

    Low variance (< 4.0) indicates uniform sentence lengths, a hallmark of
    AI-generated text. High variance (> 8.0) indicates natural writing rhythm.
    """
    if len(sentences) < 2:
        return 10.0  # Can't measure variance — assume not AI-like

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
    return math.sqrt(variance)


def _count_list_items(text: str) -> int:
    """Count markdown-style bullet points and numbered list items."""
    bullet_pattern = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
    numbered_pattern = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
    return len(bullet_pattern.findall(text)) + len(numbered_pattern.findall(text))


def _count_em_dashes(text: str) -> int:
    """Count em dashes — a pattern LLMs use to excess."""
    return text.count("—") + text.count(" -- ")


def _has_rule_of_three(text: str) -> bool:
    """Detect the 'rule of three' — AI's tendency to list exactly three items.

    We look for the pattern 'X, Y, and Z' appearing more than once, which
    suggests templated enumeration rather than organic writing.
    """
    pattern = re.compile(r"\b\w[\w\s]+,\s+\w[\w\s]+,\s+and\s+\w[\w\s]+")
    return len(pattern.findall(text)) >= 2


class StructuralDetector(BaseDetector):
    """Detects AI-like structural patterns: rhythm, formatting, and organization."""

    @property
    def name(self) -> str:
        return "StructuralDetector"

    def detect(self, text: str) -> DetectionResult:
        patterns: list[PatternMatch] = []
        structural_flags: list[str] = []

        sentences = _split_sentences(text)
        text_lower = text.lower()

        # --- Sentence length uniformity ---
        variance = _sentence_length_variance(sentences)
        if variance < 3.5 and len(sentences) >= 4:
            structural_flags.append(
                f"low_sentence_variance:{variance:.1f} (uniform sentence lengths)"
            )

        # --- AI sentence openers ---
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            for opener in AI_SENTENCE_OPENERS:
                if sentence_lower.startswith(opener):
                    idx = text_lower.find(sentence_lower[:30])
                    if idx >= 0:
                        patterns.append(
                            PatternMatch(
                                pattern_type="structural_opener",
                                matched_text=sentence[:60],
                                start=idx,
                                end=idx + len(sentence),
                                severity=0.7,
                            )
                        )
                    structural_flags.append(f"ai_opener:{opener.rstrip(',')}")
                    break  # one flag per sentence is enough

        # --- Excessive list formatting ---
        list_items = _count_list_items(text)
        total_sentences = max(len(sentences), 1)
        list_ratio = list_items / total_sentences
        if list_ratio > 0.4:
            structural_flags.append(
                f"heavy_list_formatting:{list_items}_items (list-to-sentence ratio {list_ratio:.1f})"
            )

        # --- Em dash overuse ---
        em_dash_count = _count_em_dashes(text)
        if em_dash_count >= 3:
            structural_flags.append(f"em_dash_overuse:{em_dash_count}_instances")

        # --- Rule of three ---
        if _has_rule_of_three(text):
            structural_flags.append("rule_of_three:multiple_'x,_y,_and_z'_patterns")

        # --- Uniform paragraph length ---
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            para_lengths = [len(p.split()) for p in paragraphs]
            para_mean = sum(para_lengths) / len(para_lengths)
            para_variance = sum((n - para_mean) ** 2 for n in para_lengths) / len(para_lengths)
            if math.sqrt(para_variance) < 10 and para_mean > 30:
                structural_flags.append("uniform_paragraph_lengths:blocks_of_similar_size")

        return DetectionResult(
            patterns=patterns,
            ai_vocabulary_hits=[],
            filler_phrase_hits=[],
            structural_flags=structural_flags,
            sentence_length_variance=variance,
        )
