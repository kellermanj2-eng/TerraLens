"""
narrate.py
----------
Generate a plain-language narrative of detected satellite image changes.

Public API
~~~~~~~~~~
narrate_with_granite(stats, date_range)
    Primary entry point.  Builds a context string from *stats*, then either:
      • calls IBM Granite-3 8B Instruct via watsonx.ai (when credentials are
        present in the environment), or
      • returns a deterministic template summary so the app works fully offline.

    Returns (narrative: str, source: str)  where source is "granite" or "template".

generate_narrative(stats, user_context)
    Thin compatibility shim used by app.py — delegates to narrate_with_granite.

Environment variables (loaded from .env automatically)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WATSONX_API_KEY      IBM Cloud API key with watsonx.ai access
WATSONX_PROJECT_ID   watsonx.ai project ID
WATSONX_URL          Service endpoint  (default: https://us-south.ml.cloud.ibm.com)
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if present


# ---------------------------------------------------------------------------
# Granite model settings
# ---------------------------------------------------------------------------

_MODEL_ID   = "ibm/granite-3-8b-instruct"
_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
_GEN_PARAMS = {
    "max_new_tokens": 300,
    "temperature":    0.2,
}

# Change-type taxonomy Granite is asked to classify against
_CHANGE_TYPES = (
    "wildfire",
    "flooding",
    "deforestation",
    "urban growth",
    "glacier retreat",
    "drought / vegetation loss",
    "coastal erosion",
    "unknown",
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(stats: dict, date_range: str) -> str:
    """
    Assemble the instruction prompt sent to Granite.

    The prompt encodes the quantitative change statistics, the observation
    window, and a structured task description so the model reliably produces
    a classified, confidence-annotated narrative.
    """
    change_pct   = stats.get("change_percent",    0.0)
    num_regions  = stats.get("num_regions",        0)
    changed_px   = stats.get("changed_pixels",     0)
    total_px     = stats.get("total_pixels",        0)
    regions      = stats.get("regions",            [])

    # Largest region area (first entry — detect_change sorts descending by area)
    largest_area = regions[0]["area_px"] if regions else 0

    context = (
        f"Observation window : {date_range or 'unspecified'}\n"
        f"Scene change        : {change_pct}% of total pixels ({changed_px:,} / {total_px:,})\n"
        f"Distinct regions    : {num_regions}\n"
        f"Largest region      : {largest_area:,} px²"
    )

    change_type_list = "\n".join(f"  - {t}" for t in _CHANGE_TYPES)

    prompt = f"""You are a professional satellite imagery analyst. \
You have just run an automated change-detection algorithm that compared two \
multispectral satellite images of the same geographic location captured at \
different times. The algorithm flags pixels whose brightness changed by more \
than a set threshold and groups them into contiguous regions.

Below are the quantitative results:

{context}

Your task:
1. In 2-3 plain-English sentences, explain what likely changed on the ground \
and what physical process could explain a change of this magnitude and \
spatial pattern.
2. On a new line, write "Change type: <label>" where <label> is the single \
best-matching category from this list:
{change_type_list}
3. On a new line, write "Confidence: <Low|Medium|High>" based on how \
unambiguous the statistics are, followed by a brief one-sentence caveat about \
what additional data (e.g. spectral bands, field verification) would be needed \
to confirm the classification.

Write only the three parts described above — no preamble, no extra headers.
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Template fallback  (used when credentials are absent or the API errors)
# ---------------------------------------------------------------------------

# Severity buckets so the template text is calibrated to the change magnitude
_SEVERITY = [
    (0.5,  "minimal",   "likely sensor noise or minor seasonal variation"),
    (5.0,  "moderate",  "possible vegetation stress, small flood event, or localised construction"),
    (20.0, "significant", "consistent with a major disturbance such as fire, flood, or rapid deforestation"),
    (float("inf"), "extensive",
     "indicative of a large-scale event such as a wildfire, glacial retreat, or widespread urban expansion"),
]


def _template_narrative(stats: dict, date_range: str) -> str:
    """
    Return a deterministic, stats-calibrated narrative when Granite is unavailable.

    The severity label and change-cause hint are chosen from lookup tables so
    the output is meaningfully different across a range of change magnitudes —
    not just a generic boilerplate sentence.
    """
    change_pct  = stats.get("change_percent",  0.0)
    num_regions = stats.get("num_regions",      0)
    changed_px  = stats.get("changed_pixels",   0)
    total_px    = stats.get("total_pixels",      0)
    regions     = stats.get("regions",          [])
    largest_area = regions[0]["area_px"] if regions else 0
    date_str     = f" during {date_range}" if date_range else ""

    # Pick severity tier
    for threshold, label, cause_hint in _SEVERITY:
        if change_pct <= threshold:
            break  # noqa: F821 — loop always assigns before break

    summary = (
        f"Automated change detection{date_str} flagged {change_pct}% of the scene "
        f"({changed_px:,} of {total_px:,} pixels) spread across {num_regions} "
        f"distinct region(s); the largest contiguous changed area covers "
        f"{largest_area:,} px². "
        f"The magnitude of surface change is {label}, {cause_hint}. "
        f"Without access to additional spectral bands or ground-truth data, "
        f"precise cause attribution is not possible — field verification or "
        f"multi-spectral analysis is recommended to confirm the classification."
    )

    change_type = "unknown"
    confidence  = "Low"
    caveat      = ("Multi-spectral imagery and ground verification are required "
                   "to confirm the change type and raise confidence.")

    summary += (
        f"\nChange type: {change_type}\n"
        f"Confidence: {confidence} — {caveat}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def narrate_with_granite(
    stats: dict,
    date_range: str = "",
) -> tuple[str, str]:
    """
    Generate a plain-language narrative for *stats*.

    Parameters
    ----------
    stats      : dict as returned by change_detection.detect_change().
                 Required keys: change_percent, num_regions, changed_pixels,
                 total_pixels, regions (list of region dicts).
    date_range : human-readable observation window, e.g. "Jan 2023 → Jul 2023".
                 Used in both the Granite prompt and the template fallback.

    Returns
    -------
    (narrative: str, source: str)
        source == "granite"  — response came from IBM Granite via watsonx.ai
        source == "template" — deterministic fallback was used
    """
    # Check credentials before attempting any network call
    api_key    = os.getenv("WATSONX_API_KEY",    "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    url        = os.getenv("WATSONX_URL", _DEFAULT_URL).strip()

    if not api_key or not project_id:
        return _template_narrative(stats, date_range), "template"

    prompt = _build_prompt(stats, date_range)

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(url=url, api_key=api_key)
        model = ModelInference(
            model_id=_MODEL_ID,
            credentials=credentials,
            project_id=project_id,
            params=_GEN_PARAMS,
        )
        response  = model.generate_text(prompt=prompt)
        narrative = response.strip() if isinstance(response, str) else str(response).strip()
        if narrative:
            return narrative, "granite"
    except Exception:
        pass  # fall through to template

    return _template_narrative(stats, date_range), "template"


# ---------------------------------------------------------------------------
# Compatibility shim  (called by app.py as generate_narrative)
# ---------------------------------------------------------------------------

def generate_narrative(stats: dict, user_context: str = "") -> tuple[str, str]:
    """
    Thin wrapper kept for app.py compatibility.

    *user_context* is treated as the *date_range* string; if it contains no
    date information it is still forwarded to narrate_with_granite, which will
    embed it in the prompt / template as the observation window label.
    """
    return narrate_with_granite(stats, date_range=user_context)
