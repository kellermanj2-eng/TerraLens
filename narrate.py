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
    "max_new_tokens": 400,
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
    "agricultural change",
    "unknown",
)

# ---------------------------------------------------------------------------
# Few-shot examples  (used to anchor the classifier)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = """\
### Example 1
Scene change: 18.4% (121,440 / 660,000 px) | Regions: 3 | Largest: 98,210 px²
Narrative: A large contiguous area of surface reflectance changed dramatically, concentrated in three patches that together cover nearly a fifth of the scene. The spatial compactness and high intensity are consistent with a wildfire burn scar that removed canopy abruptly between the two acquisition dates.
Change type: wildfire
Confidence: High — Spectral confirmation (SWIR increase, NIR drop) and field survey would verify the burn extent.

### Example 2
Scene change: 6.1% (40,260 / 660,000 px) | Regions: 14 | Largest: 8,100 px²
Narrative: Changes are scattered across many small regions with no dominant cluster, suggesting incremental surface modifications rather than a single event. This fragmented pattern is typical of active construction, land clearing in peri-urban areas, or progressive agricultural expansion.
Change type: urban growth
Confidence: Medium — High-resolution imagery and cadastral records would distinguish urban construction from agricultural conversion.

### Example 3
Scene change: 0.9% (5,940 / 660,000 px) | Regions: 22 | Largest: 1,200 px²
Narrative: Very minor surface change spread across numerous tiny regions. At this magnitude the signal is likely dominated by sensor noise, atmospheric correction artefacts, or minor phenological variation in vegetation.
Change type: unknown
Confidence: Low — The change magnitude is below the threshold where cause attribution is reliable; additional acquisition dates and multi-spectral bands are needed.
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(stats: dict, date_range: str) -> str:
    """
    Assemble the few-shot instruction prompt sent to Granite.

    The prompt provides three labelled examples that anchor the classifier to
    the taxonomy before presenting the target statistics, significantly
    improving label consistency compared to a zero-shot prompt.
    """
    change_pct   = stats.get("change_percent",    0.0)
    num_regions  = stats.get("num_regions",        0)
    changed_px   = stats.get("changed_pixels",     0)
    total_px     = stats.get("total_pixels",        0)
    regions      = stats.get("regions",            [])
    ndvi_hint    = stats.get("ndvi_hint",          "")

    largest_area = regions[0]["area_px"] if regions else 0

    context = (
        f"Scene change: {change_pct}% ({changed_px:,} / {total_px:,} px) | "
        f"Regions: {num_regions} | Largest: {largest_area:,} px²"
    )

    if ndvi_hint:
        context += f"\nNDVI signal: {ndvi_hint}"

    change_type_list = "\n".join(f"  - {t}" for t in _CHANGE_TYPES)

    prompt = f"""You are a professional satellite imagery analyst using a \
few-shot classification system. Study the examples below, then classify the \
target scene in exactly the same format.

Valid change types:
{change_type_list}

---
{_FEW_SHOT_EXAMPLES}
---

### Target scene
Observation window: {date_range or 'unspecified'}
{context}

Respond with exactly three lines:
Narrative: <2-3 plain-English sentences explaining the most likely ground event>
Change type: <single label from the list above>
Confidence: <Low|Medium|High> — <one sentence: what additional data would confirm the classification>
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Parse structured classifier output
# ---------------------------------------------------------------------------

def _parse_classifier_output(raw: str) -> dict:
    """
    Extract structured fields from Granite's response.

    Returns a dict with keys: narrative, change_type, confidence, caveat.
    Falls back gracefully if any field is missing.
    """
    import re
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]

    narrative   = ""
    change_type = "unknown"
    confidence  = "Low"
    caveat      = ""

    for line in lines:
        if line.lower().startswith("narrative:"):
            narrative = line[len("narrative:"):].strip()
        elif line.lower().startswith("change type:"):
            change_type = line[len("change type:"):].strip().lower()
        elif line.lower().startswith("confidence:"):
            rest = line[len("confidence:"):].strip()
            # Split on " — " or " - " or first comma
            m = re.match(r"^(low|medium|high)[^\w]*(.*)", rest, re.IGNORECASE)
            if m:
                confidence = m.group(1).capitalize()
                caveat     = m.group(2).strip(" —-,")
            else:
                confidence = rest.split()[0].capitalize() if rest else "Low"

    # If Granite returned a monolithic block without headers, keep it as narrative
    if not narrative and lines:
        narrative = " ".join(lines)

    return {
        "narrative":   narrative,
        "change_type": change_type,
        "confidence":  confidence,
        "caveat":      caveat,
    }


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
) -> tuple[str, str, dict]:
    """
    Generate a plain-language narrative for *stats*.

    Parameters
    ----------
    stats      : dict as returned by change_detection.detect_change().
                 Required keys: change_percent, num_regions, changed_pixels,
                 total_pixels, regions (list of region dicts).
                 Optional key: ndvi_hint (str) — brief NDVI signal description
                 forwarded into the few-shot prompt for richer classification.
    date_range : human-readable observation window, e.g. "Jan 2023 → Jul 2023".
                 Used in both the Granite prompt and the template fallback.

    Returns
    -------
    (narrative: str, source: str, classified: dict)
        source     == "granite"  — response came from IBM Granite via watsonx.ai
        source     == "template" — deterministic fallback was used
        classified == dict with keys: narrative, change_type, confidence, caveat
    """
    # Check credentials before attempting any network call
    api_key    = os.getenv("WATSONX_API_KEY",    "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    url        = os.getenv("WATSONX_URL", _DEFAULT_URL).strip()

    if not api_key or not project_id:
        raw = _template_narrative(stats, date_range)
        classified = _parse_classifier_output(raw)
        return raw, "template", classified

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
        raw = response.strip() if isinstance(response, str) else str(response).strip()
        if raw:
            classified = _parse_classifier_output(raw)
            # Reconstruct full narrative string from parsed fields for display
            narrative = classified["narrative"] or raw
            if classified["change_type"] and classified["change_type"] != "unknown":
                narrative += f"\nChange type: {classified['change_type']}"
            if classified["confidence"]:
                caveat_str = f" — {classified['caveat']}" if classified["caveat"] else ""
                narrative += f"\nConfidence: {classified['confidence']}{caveat_str}"
            return narrative, "granite", classified
    except Exception:
        pass  # fall through to template

    raw = _template_narrative(stats, date_range)
    classified = _parse_classifier_output(raw)
    return raw, "template", classified


# ---------------------------------------------------------------------------
# Compatibility shim  (called by app.py as generate_narrative)
# ---------------------------------------------------------------------------

def generate_narrative(stats: dict, user_context: str = "") -> tuple[str, str, dict]:
    """
    Thin wrapper kept for app.py compatibility.

    *user_context* is treated as the *date_range* string; if it contains no
    date information it is still forwarded to narrate_with_granite, which will
    embed it in the prompt / template as the observation window label.
    """
    return narrate_with_granite(stats, date_range=user_context)
