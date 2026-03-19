"""
Lexical detector: finds AI-like words and phrases in text.

This is a knowledge-base approach — we maintain curated lists of vocabulary
and phrases that LLMs use disproportionately. The lists are drawn from:
  - Empirical analysis of GPT/Claude output patterns
  - Wikipedia's "Signs of AI writing" article
  - Human editorial observation

Detection is case-insensitive and matches whole words only for vocabulary
(to avoid flagging "streamlined" inside "downstream"), but substring matches
for multi-word phrases.
"""

import re

from humanizer.core.models import DetectionResult, PatternMatch
from humanizer.detectors.base import BaseDetector

# Words that appear at statistically elevated rates in LLM output.
# These are not inherently bad words — they're flags for *density*.
# A single use of "leverage" is fine; three in one paragraph is a signal.
AI_VOCABULARY: frozenset[str] = frozenset(
    {
        # Overused verbs
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
        "redefine",
        "unleash",
        "craft",
        "delve",
        # Overused adjectives
        "comprehensive",
        "robust",
        "seamless",
        "holistic",
        "transformative",
        "innovative",
        "groundbreaking",
        "cutting-edge",
        "state-of-the-art",
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
        "rigorous",
        "substantial",
        "crucial",
        "vital",
        "paramount",
        "pivotal",
        "invaluable",
        "indispensable",
        "unprecedented",
        "commendable",
        "exceptional",
        "remarkable",
        "captivating",
        "compelling",
        "vibrant",
        "thriving",
        # Overused nouns
        "synergy",
        "ecosystem",
        "landscape",
        "paradigm",
        "tapestry",
        "endeavor",
        "journey",
        # Inflated adverbs
        "seamlessly",
        "holistically",
        "proactively",
        "strategically",
    }
)

# Multi-word filler phrases. These are matched as substrings (lowercased)
# because they appear inline. Order longest-first so overlapping matches
# don't suppress longer ones.
AI_FILLER_PHRASES: tuple[str, ...] = (
    "it is worth noting that",
    "it is important to note that",
    "it is important to note",
    "it should be noted that",
    "it should be noted",
    "it goes without saying",
    "needless to say",
    "as previously mentioned",
    "as mentioned above",
    "as noted above",
    "as noted earlier",
    "with that being said",
    "that being said",
    "with that said",
    "at the end of the day",
    "in today's fast-paced world",
    "in today's world",
    "in the digital age",
    "in the realm of",
    "in the field of",
    "when it comes to",
    "in terms of",
    "plays a crucial role",
    "plays a vital role",
    "plays a key role",
    "plays an important role",
    "a wide range of",
    "a wide variety of",
    "a multitude of",
    "a plethora of",
    "not only",
    "first and foremost",
    "last but not least",
    "in conclusion",
    "to summarize",
    "in summary",
    "furthermore",
    "moreover",
    "additionally",
    "in addition",
)


class LexicalDetector(BaseDetector):
    """Detects AI-like vocabulary and filler phrases."""

    @property
    def name(self) -> str:
        return "LexicalDetector"

    def detect(self, text: str) -> DetectionResult:
        patterns: list[PatternMatch] = []
        vocab_hits: list[str] = []
        phrase_hits: list[str] = []

        text_lower = text.lower()

        # Vocabulary scan — whole-word matching
        for word in AI_VOCABULARY:
            # Use word-boundary regex so "leverage" doesn't match inside
            # "leveraged" in some contexts — actually we want both.
            # Use \b for cleaner matching.
            pattern = rf"\b{re.escape(word)}\b"
            for match in re.finditer(pattern, text_lower):
                patterns.append(
                    PatternMatch(
                        pattern_type="ai_vocabulary",
                        matched_text=text[match.start() : match.end()],
                        start=match.start(),
                        end=match.end(),
                        severity=0.6,
                    )
                )
                if word not in vocab_hits:
                    vocab_hits.append(word)

        # Filler phrase scan — substring matching
        for phrase in AI_FILLER_PHRASES:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                patterns.append(
                    PatternMatch(
                        pattern_type="filler_phrase",
                        matched_text=text[idx : idx + len(phrase)],
                        start=idx,
                        end=idx + len(phrase),
                        severity=0.8,
                    )
                )
                if phrase not in phrase_hits:
                    phrase_hits.append(phrase)
                start = idx + len(phrase)

        return DetectionResult(
            patterns=patterns,
            ai_vocabulary_hits=vocab_hits,
            filler_phrase_hits=phrase_hits,
            structural_flags=[],
            sentence_length_variance=0.0,
        )
