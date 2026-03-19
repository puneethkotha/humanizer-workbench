"""
HumanizerEngine: the central orchestrator.

The engine owns the transformation lifecycle:
  1. Run the composite detector on the input text.
  2. Score the input with the heuristic scorer.
  3. Build a pipeline appropriate for the requested intensity.
  4. Execute each pipeline stage in sequence, passing output forward.
  5. Re-run detection after the final stage.
  6. Score the output and return a complete HumanizerResult.

The engine does not know about prompts, API calls, or style definitions. It
assembles the components and moves data between them. Business logic lives in
the components; the engine provides the wiring.

Design note on re-detection:
  We re-detect before the AUDIT stage (if it runs) so the audit prompt has
  accurate information about what patterns still remain. This costs one extra
  detection pass for aggressive intensity — it's worth it for audit quality.
"""

import os

import anthropic

from humanizer.core.models import (
    HumanizerResult,
    Intensity,
    PipelineStage,
    StageResult,
    StyleName,
)
from humanizer.core.pipeline import Pipeline
from humanizer.detectors.base import CompositeDetector
from humanizer.detectors.lexical import LexicalDetector
from humanizer.detectors.structural import StructuralDetector
from humanizer.scoring.scorer import AIScorer
from humanizer.styles.presets import STYLE_REGISTRY
from humanizer.transformers.llm import LLMTransformer

DEFAULT_MODEL = "claude-sonnet-4-6"

# Warn (but don't fail) if input text exceeds this word count.
# Claude handles very long contexts, but humanization quality degrades for
# very long passages. Recommend chunking above this threshold.
LONG_TEXT_THRESHOLD = 1500


class HumanizerEngine:
    """Orchestrates the full humanization pipeline.

    Usage:
        engine = HumanizerEngine()  # reads ANTHROPIC_API_KEY from environment
        result = engine.humanize("Your AI-generated text here.")
        print(result.output)
        print(f"Score: {result.before_score:.0f} → {result.after_score:.0f}")

    Args:
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable.
        model: Claude model to use for transformations. Defaults to claude-sonnet-4-6.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key is required. Set the ANTHROPIC_API_KEY environment "
                "variable or pass api_key= to HumanizerEngine()."
            )

        client = anthropic.Anthropic(api_key=resolved_key)

        self._transformer = LLMTransformer(client=client, model=model)
        self._detector = CompositeDetector(detectors=[LexicalDetector(), StructuralDetector()])
        self._scorer = AIScorer()
        self._model = model

    def humanize(
        self,
        text: str,
        style: StyleName = StyleName.PROFESSIONAL,
        intensity: Intensity = Intensity.MEDIUM,
    ) -> HumanizerResult:
        """Humanize a piece of text.

        Args:
            text: The text to transform.
            style: The writing style preset to apply.
            intensity: How aggressively to transform the text.

        Returns:
            HumanizerResult with the transformed text, scores, and stage details.

        Raises:
            ValueError: If text is empty.
            anthropic.APIError: If the Anthropic API call fails.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        word_count = len(text.split())
        if word_count > LONG_TEXT_THRESHOLD:
            import warnings

            warnings.warn(
                f"Input text is {word_count} words. Humanization quality is best below "
                f"{LONG_TEXT_THRESHOLD} words. Consider splitting into sections.",
                UserWarning,
                stacklevel=2,
            )

        style_preset = STYLE_REGISTRY[style]
        pipeline = Pipeline.for_intensity(intensity)

        # --- Initial detection and scoring ---
        initial_detection = self._detector.detect(text)
        before_score = self._scorer.score(text, initial_detection)

        # --- Execute pipeline stages ---
        current_text = text
        current_detection = initial_detection
        stage_results: list[StageResult] = []

        for stage in pipeline.stages:
            # Re-run detection before AUDIT so it has fresh signal.
            # For REWRITE and REFINE, use the detection from the previous iteration
            # (detection on the original is more informative for REWRITE).
            if stage == PipelineStage.AUDIT:
                current_detection = self._detector.detect(current_text)

            stage_result = self._transformer.transform(
                text=current_text,
                stage=stage,
                style=style_preset,
                detection=current_detection,
                intensity=intensity,
            )
            stage_results.append(stage_result)
            current_text = stage_result.output_text

        # --- Final detection and scoring ---
        final_detection = self._detector.detect(current_text)
        after_score = self._scorer.score(current_text, final_detection)

        # --- Collect change summary ---
        changes_summary = self._build_changes_summary(stage_results)

        return HumanizerResult(
            original=text,
            output=current_text,
            style=style,
            intensity=intensity,
            before_score=before_score,
            after_score=after_score,
            stages=stage_results,
            changes_summary=changes_summary,
        )

    def _build_changes_summary(self, stages: list[StageResult]) -> list[str]:
        """Deduplicated list of all changes made across pipeline stages."""
        seen: set[str] = set()
        summary: list[str] = []
        for stage in stages:
            for change in stage.changes_made:
                if change not in seen:
                    seen.add(change)
                    summary.append(change)
        return summary

    @property
    def model(self) -> str:
        return self._model
