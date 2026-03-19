"""
Unit tests for style presets.

Validates that the style registry is complete and each style has all
required fields. Does not test LLM output — that's in integration tests.
"""

import pytest

from humanizer.core.models import StyleName
from humanizer.styles.base import StylePreset
from humanizer.styles.presets import STYLE_REGISTRY, get_style


class TestStyleRegistry:
    def test_all_style_names_have_presets(self):
        for style_name in StyleName:
            assert style_name in STYLE_REGISTRY, (
                f"StyleName.{style_name.name} has no entry in STYLE_REGISTRY"
            )

    def test_all_presets_are_style_preset_instances(self):
        for name, preset in STYLE_REGISTRY.items():
            assert isinstance(preset, StylePreset), (
                f"STYLE_REGISTRY[{name}] is not a StylePreset instance"
            )

    def test_no_extra_keys_in_registry(self):
        style_name_values = {s for s in StyleName}
        registry_keys = set(STYLE_REGISTRY.keys())
        assert registry_keys == style_name_values


class TestStylePresetFields:
    """Each style preset must have non-empty required fields."""

    @pytest.fixture(params=list(StyleName))
    def style_preset(self, request):
        return STYLE_REGISTRY[request.param]

    def test_name_is_non_empty(self, style_preset):
        assert style_preset.name.strip() != ""

    def test_voice_description_is_substantive(self, style_preset):
        assert len(style_preset.voice_description.split()) >= 10, (
            f"Voice description too short for style '{style_preset.name}'"
        )

    def test_tone_descriptors_are_present(self, style_preset):
        assert len(style_preset.tone_descriptors) >= 2, (
            f"Expected at least 2 tone descriptors for style '{style_preset.name}'"
        )

    def test_structural_guidance_is_substantive(self, style_preset):
        assert len(style_preset.structural_guidance.split()) >= 15, (
            f"Structural guidance too short for style '{style_preset.name}'"
        )

    def test_vocabulary_guidance_is_substantive(self, style_preset):
        assert len(style_preset.vocabulary_guidance.split()) >= 15, (
            f"Vocabulary guidance too short for style '{style_preset.name}'"
        )

    def test_avoided_patterns_are_present(self, style_preset):
        assert len(style_preset.avoided_patterns) >= 2, (
            f"Expected at least 2 avoided patterns for style '{style_preset.name}'"
        )

    def test_rewrite_instruction_is_substantive(self, style_preset):
        assert len(style_preset.rewrite_instruction.split()) >= 10, (
            f"Rewrite instruction too short for style '{style_preset.name}'"
        )

    def test_refine_instruction_is_substantive(self, style_preset):
        assert len(style_preset.refine_instruction.split()) >= 10, (
            f"Refine instruction too short for style '{style_preset.name}'"
        )

    def test_audit_instruction_is_substantive(self, style_preset):
        assert len(style_preset.audit_instruction.split()) >= 10, (
            f"Audit instruction too short for style '{style_preset.name}'"
        )


class TestGetStyle:
    def test_get_style_returns_correct_preset(self):
        preset = get_style(StyleName.CASUAL)
        assert preset.name == "casual"

    def test_all_styles_retrievable(self):
        for style_name in StyleName:
            preset = get_style(style_name)
            assert preset is not None

    def test_styles_are_distinct(self):
        """Ensure each style has a unique voice description."""
        voice_descriptions = [STYLE_REGISTRY[s].voice_description for s in StyleName]
        assert len(set(voice_descriptions)) == len(voice_descriptions), (
            "Two style presets have identical voice descriptions"
        )

    def test_styles_have_distinct_tone_descriptors(self):
        """The sets of tone descriptors should differ between styles."""
        tone_sets = [frozenset(STYLE_REGISTRY[s].tone_descriptors) for s in StyleName]
        # No two styles should have identical tone descriptor sets
        assert len(set(tone_sets)) == len(tone_sets), (
            "Two style presets have identical tone descriptor sets"
        )
