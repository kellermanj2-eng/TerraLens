"""
tests/test_narrate.py
----------------------
Unit tests for narrate.py — template fallback and prompt construction.

No watsonx.ai credentials are required; tests exercise the template path
and structural properties of the prompt/parser only.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force template mode by ensuring credentials are absent
os.environ.pop("WATSONX_API_KEY",    None)
os.environ.pop("WATSONX_PROJECT_ID", None)

from narrate import (
    _build_prompt,
    _template_narrative,
    _parse_classifier_output,
    narrate_with_granite,
    _CHANGE_TYPES,
    _FEW_SHOT_EXAMPLES,
)


SAMPLE_STATS = {
    "change_percent":  18.4,
    "num_regions":     3,
    "changed_pixels":  121_440,
    "total_pixels":    660_000,
    "changed_fraction": 0.184,
    "regions": [{"area_px": 98_210}],
}


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_change_percent(self):
        prompt = _build_prompt(SAMPLE_STATS, "Jan → Jun 2024")
        assert "18.4%" in prompt or "18.4" in prompt

    def test_prompt_contains_date_range(self):
        prompt = _build_prompt(SAMPLE_STATS, "Jan → Jun 2024")
        assert "Jan → Jun 2024" in prompt

    def test_prompt_contains_all_change_types(self):
        prompt = _build_prompt(SAMPLE_STATS, "")
        for ct in _CHANGE_TYPES:
            assert ct in prompt, f"Change type '{ct}' missing from prompt"

    def test_prompt_contains_few_shot_examples(self):
        prompt = _build_prompt(SAMPLE_STATS, "")
        # Verify at least one example heading is present
        assert "### Example" in prompt or "Example 1" in prompt

    def test_prompt_includes_ndvi_hint_when_present(self):
        stats_with_hint = {**SAMPLE_STATS, "ndvi_hint": "NDVI dropped by 0.3"}
        prompt = _build_prompt(stats_with_hint, "")
        assert "NDVI dropped by 0.3" in prompt

    def test_prompt_unspecified_date_range(self):
        prompt = _build_prompt(SAMPLE_STATS, "")
        assert "unspecified" in prompt


# ---------------------------------------------------------------------------
# _template_narrative
# ---------------------------------------------------------------------------

class TestTemplateNarrative:
    def test_returns_string(self):
        result = _template_narrative(SAMPLE_STATS, "Q1 2024")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_contains_change_percent(self):
        result = _template_narrative(SAMPLE_STATS, "")
        assert "18.4" in result

    def test_contains_change_type_line(self):
        result = _template_narrative(SAMPLE_STATS, "")
        assert "Change type:" in result

    def test_contains_confidence_line(self):
        result = _template_narrative(SAMPLE_STATS, "")
        assert "Confidence:" in result

    def test_date_range_in_output(self):
        result = _template_narrative(SAMPLE_STATS, "March 2023")
        assert "March 2023" in result

    def test_zero_change(self):
        zero = {**SAMPLE_STATS, "change_percent": 0.0,
                "changed_pixels": 0, "num_regions": 0, "regions": []}
        result = _template_narrative(zero, "")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _parse_classifier_output
# ---------------------------------------------------------------------------

class TestParseClassifierOutput:
    def test_parses_well_formed_output(self):
        raw = (
            "Narrative: A large burn scar formed over dense forest.\n"
            "Change type: wildfire\n"
            "Confidence: High — Spectral confirmation needed."
        )
        result = _parse_classifier_output(raw)
        assert result["narrative"] == "A large burn scar formed over dense forest."
        assert result["change_type"] == "wildfire"
        assert result["confidence"] == "High"
        assert "Spectral" in result["caveat"]

    def test_parses_medium_confidence(self):
        raw = (
            "Narrative: Something changed.\n"
            "Change type: flooding\n"
            "Confidence: Medium — Field verification needed."
        )
        result = _parse_classifier_output(raw)
        assert result["confidence"] == "Medium"

    def test_unknown_fallback(self):
        raw = "Some unstructured text without headers"
        result = _parse_classifier_output(raw)
        # Should not crash; narrative should be populated
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_empty_input(self):
        result = _parse_classifier_output("")
        assert result["change_type"] == "unknown"
        assert result["confidence"] == "Low"


# ---------------------------------------------------------------------------
# narrate_with_granite (template path — no credentials)
# ---------------------------------------------------------------------------

class TestNarrateWithGranite:
    def test_returns_three_tuple(self):
        result = narrate_with_granite(SAMPLE_STATS, "Q1 2024")
        assert len(result) == 3

    def test_source_is_template_without_credentials(self):
        narrative, source, classified = narrate_with_granite(SAMPLE_STATS, "")
        assert source == "template"

    def test_narrative_is_nonempty_string(self):
        narrative, _, _ = narrate_with_granite(SAMPLE_STATS, "")
        assert isinstance(narrative, str)
        assert len(narrative) > 20

    def test_classified_has_required_keys(self):
        _, _, classified = narrate_with_granite(SAMPLE_STATS, "")
        for key in ("narrative", "change_type", "confidence", "caveat"):
            assert key in classified
