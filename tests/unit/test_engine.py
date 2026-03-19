"""
Unit tests for HumanizerEngine and Pipeline.

The engine tests mock the LLM transformer to avoid API calls.
Integration tests (marked with @pytest.mark.integration) use the real API
and require ANTHROPIC_API_KEY to be set.

Run unit tests only:
    pytest tests/unit/test_engine.py -m "not integration"

Run all including integration:
    pytest tests/unit/test_engine.py
"""

import pytest

from humanizer.core.engine import HumanizerEngine
from humanizer.core.models import Intensity, PipelineStage, StageResult, StyleName
from humanizer.core.pipeline import Pipeline

# --- Pipeline ---


class TestPipeline:
    def test_light_pipeline_has_one_stage(self):
        pipeline = Pipeline.for_intensity(Intensity.LIGHT)
        assert pipeline.stage_count == 1
        assert pipeline.stages == (PipelineStage.REWRITE,)

    def test_medium_pipeline_has_two_stages(self):
        pipeline = Pipeline.for_intensity(Intensity.MEDIUM)
        assert pipeline.stage_count == 2
        assert pipeline.stages == (PipelineStage.REWRITE, PipelineStage.REFINE)

    def test_aggressive_pipeline_has_three_stages(self):
        pipeline = Pipeline.for_intensity(Intensity.AGGRESSIVE)
        assert pipeline.stage_count == 3
        assert pipeline.stages == (
            PipelineStage.REWRITE,
            PipelineStage.REFINE,
            PipelineStage.AUDIT,
        )

    def test_pipeline_is_frozen(self):
        pipeline = Pipeline.for_intensity(Intensity.MEDIUM)
        with pytest.raises((AttributeError, TypeError)):
            pipeline.stages = (PipelineStage.REWRITE,)  # type: ignore

    def test_pipeline_repr_includes_stage_names(self):
        pipeline = Pipeline.for_intensity(Intensity.MEDIUM)
        repr_str = repr(pipeline)
        assert "REWRITE" in repr_str
        assert "REFINE" in repr_str

    @pytest.mark.parametrize(
        "intensity, expected_count",
        [
            (Intensity.LIGHT, 1),
            (Intensity.MEDIUM, 2),
            (Intensity.AGGRESSIVE, 3),
        ],
    )
    def test_pipeline_stage_counts(self, intensity, expected_count):
        pipeline = Pipeline.for_intensity(intensity)
        assert pipeline.stage_count == expected_count


# --- HumanizerEngine (mocked) ---


class TestHumanizerEngineInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            HumanizerEngine(api_key=None)

    def test_accepts_explicit_api_key(self, mocker):
        mocker.patch("anthropic.Anthropic")
        engine = HumanizerEngine(api_key="test-key-123")
        assert engine.model == "claude-sonnet-4-6"

    def test_accepts_custom_model(self, mocker):
        mocker.patch("anthropic.Anthropic")
        engine = HumanizerEngine(api_key="test-key", model="claude-opus-4-6")
        assert engine.model == "claude-opus-4-6"


class TestHumanizerEngineHumanize:
    """Tests for engine.humanize() with mocked transformer."""

    @pytest.fixture
    def mock_engine(self, mocker):
        """HumanizerEngine with mocked Anthropic client and transformer."""
        mocker.patch("anthropic.Anthropic")

        engine = HumanizerEngine(api_key="test-key")

        # Mock the transformer so we don't need real API calls.
        # Each call returns the input text transformed slightly.
        def fake_transform(text, stage, style, detection, intensity):
            # Simulate a rewrite by removing one common AI phrase.
            output = text.replace("it is worth noting that", "")
            output = output.replace("furthermore,", "")
            output = output.strip()
            return StageResult(
                stage=stage,
                input_text=text,
                output_text=output if output else text,
                changes_made=["Removed filler phrase"],
            )

        engine._transformer.transform = fake_transform
        return engine

    def test_raises_on_empty_input(self, mock_engine):
        with pytest.raises(ValueError, match="empty"):
            mock_engine.humanize("")

    def test_raises_on_whitespace_only_input(self, mock_engine):
        with pytest.raises(ValueError, match="empty"):
            mock_engine.humanize("   \n\n  ")

    def test_returns_humanizer_result(self, mock_engine):
        text = "It is worth noting that this approach facilitates robust synergy."
        result = mock_engine.humanize(text, style=StyleName.CASUAL, intensity=Intensity.LIGHT)
        assert result.original == text
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    def test_result_has_correct_style(self, mock_engine):
        text = "This approach leverages comprehensive methodologies."
        result = mock_engine.humanize(text, style=StyleName.FOUNDER)
        assert result.style == StyleName.FOUNDER

    def test_result_has_correct_intensity(self, mock_engine):
        text = "This approach leverages comprehensive methodologies."
        result = mock_engine.humanize(text, intensity=Intensity.AGGRESSIVE)
        assert result.intensity == Intensity.AGGRESSIVE

    def test_light_intensity_runs_one_stage(self, mock_engine):
        text = "Furthermore, this leverages robust paradigms."
        result = mock_engine.humanize(text, intensity=Intensity.LIGHT)
        assert len(result.stages) == 1
        assert result.stages[0].stage == PipelineStage.REWRITE

    def test_medium_intensity_runs_two_stages(self, mock_engine):
        text = "Furthermore, this leverages robust paradigms."
        result = mock_engine.humanize(text, intensity=Intensity.MEDIUM)
        assert len(result.stages) == 2
        assert result.stages[0].stage == PipelineStage.REWRITE
        assert result.stages[1].stage == PipelineStage.REFINE

    def test_aggressive_intensity_runs_three_stages(self, mock_engine):
        text = "Furthermore, this leverages robust paradigms."
        result = mock_engine.humanize(text, intensity=Intensity.AGGRESSIVE)
        assert len(result.stages) == 3

    def test_before_and_after_scores_are_bounded(self, mock_engine):
        text = "Furthermore, this comprehensive approach leverages robust synergies."
        result = mock_engine.humanize(text)
        assert 0 <= result.before_score <= 100
        assert 0 <= result.after_score <= 100

    def test_improvement_is_before_minus_after(self, mock_engine):
        text = "Furthermore, this leverages robust paradigms."
        result = mock_engine.humanize(text)
        expected = result.before_score - result.after_score
        assert abs(result.improvement - expected) < 0.01

    def test_changes_summary_is_deduplicated(self, mock_engine):
        text = "Furthermore, this leverages robust paradigms. Moreover, it facilitates growth."
        result = mock_engine.humanize(text, intensity=Intensity.MEDIUM)
        # No duplicate entries in changes_summary
        assert len(result.changes_summary) == len(set(result.changes_summary))

    def test_long_text_warning(self, mock_engine):
        # 8 words × 220 repetitions = 1760 words, which exceeds the 1500-word threshold
        long_text = " ".join(
            ["This comprehensive approach leverages robust synergies and ecosystems."] * 220
        )
        with pytest.warns(UserWarning, match="words"):
            mock_engine.humanize(long_text)


# --- Integration tests (require real API key) ---


@pytest.mark.integration
class TestHumanizerEngineIntegration:
    """Real API tests. Run with: pytest -m integration"""

    def test_humanize_basic_text(self):
        engine = HumanizerEngine()  # reads ANTHROPIC_API_KEY from env
        text = (
            "It is worth noting that this comprehensive approach leverages robust methodologies "
            "to facilitate seamless integration across all organizational stakeholders. "
            "Furthermore, these transformative strategies empower teams to optimize their workflows."
        )
        result = engine.humanize(text, style=StyleName.PROFESSIONAL, intensity=Intensity.LIGHT)
        assert result.output != text
        assert len(result.output) > 50
        assert result.before_score > result.after_score

    def test_humanize_all_styles(self):
        engine = HumanizerEngine()
        text = (
            "Furthermore, this innovative approach leverages comprehensive methodologies "
            "to facilitate seamless collaboration. In conclusion, the transformative "
            "synergies will yield substantial organizational benefits."
        )
        for style in StyleName:
            result = engine.humanize(text, style=style, intensity=Intensity.LIGHT)
            assert len(result.output) > 20, f"Empty output for style {style}"

    def test_aggressive_intensity_improves_score(self):
        engine = HumanizerEngine()
        text = (
            "In today's fast-paced world, it is worth noting that comprehensive "
            "communication strategies play a crucial role in facilitating organizational "
            "success. Furthermore, leveraging robust holistic methodologies empowers teams "
            "to seamlessly navigate complex challenges. Moreover, these transformative "
            "approaches enable stakeholders to optimize performance and streamline workflows. "
            "In conclusion, this paradigm shift represents a paramount step toward achieving "
            "unprecedented synergistic outcomes."
        )
        result = engine.humanize(text, intensity=Intensity.AGGRESSIVE)
        assert result.improvement > 0, (
            f"Expected improvement > 0, got {result.improvement}. "
            f"Before: {result.before_score}, After: {result.after_score}"
        )
