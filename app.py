"""
app.py
------
Streamlit dashboard for TerraLens.

Layout
~~~~~~
  Sidebar  : change threshold slider, date-range text input
  Main     : two file uploaders → three-column image view (Before / After / Overlay)
             → stats metrics row → "Generate AI Narration" button → narrative box
             → download buttons

Usage:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import streamlit as st

from change_detection import load_and_align, detect_change, overlay as make_overlay
from narrate import narrate_with_granite

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TerraLens — Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛰️ TerraLens")
st.caption(
    "Upload two satellite images of the same location taken at different dates. "
    "TerraLens aligns them, detects meaningful surface changes, and uses "
    "**IBM Granite** to explain what likely happened on the ground."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Change threshold",
        min_value=10, max_value=100, value=40, step=5,
        help="Pixel intensity difference (0-255) required to flag a change. "
             "Lower = more sensitive; higher = only major changes.",
    )
    date_range = st.text_input(
        "Date range (optional)",
        placeholder="e.g. June 3 – July 12, 2025",
        help="Observation window shown in the Granite prompt and the offline summary.",
    )

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) (Sentinel-2)\n"
        "- [NASA Worldview / GIBS](https://worldview.earthdata.nasa.gov/) (Landsat)\n"
    )
    st.markdown("**Output directory:** `results/`")

# ---------------------------------------------------------------------------
# File uploaders
# ---------------------------------------------------------------------------

up_col1, up_col2 = st.columns(2)
with up_col1:
    before_file = st.file_uploader(
        "📂 Before image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="before",
        help="Earlier satellite image (GeoTIFF or standard raster).",
    )
with up_col2:
    after_file = st.file_uploader(
        "📂 After image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="after",
        help="Later satellite image of the same location.",
    )

# ---------------------------------------------------------------------------
# No-upload placeholder
# ---------------------------------------------------------------------------

if not before_file or not after_file:
    st.divider()
    st.info(
        "👆 Upload a **Before** image and an **After** image above to begin analysis.\n\n"
        "**Tip:** Use the sidebar slider to tune the change sensitivity, and optionally "
        "enter the date range so Granite can provide better-calibrated commentary.\n\n"
        "Supported formats: PNG, JPEG, GeoTIFF (.tif/.tiff).",
        icon="🛰️",
    )
    st.stop()  # Nothing more to render until both files are present

# ---------------------------------------------------------------------------
# Run change-detection pipeline (cached per uploaded file pair + settings)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_analysis(before_bytes: bytes, before_name: str,
                  after_bytes: bytes,  after_name: str,
                  threshold: int) -> dict:
    """
    Write uploads to temporary files, run load_and_align + detect_change,
    and return a serialisable result dict.

    Results are cached so re-running the page (e.g. opening the narration
    panel) does not redo the expensive CV computation.
    """
    before_suffix = os.path.splitext(before_name)[1]
    after_suffix  = os.path.splitext(after_name)[1]

    with tempfile.NamedTemporaryFile(suffix=before_suffix, delete=False) as f:
        f.write(before_bytes)
        before_path = f.name
    with tempfile.NamedTemporaryFile(suffix=after_suffix, delete=False) as f:
        f.write(after_bytes)
        after_path = f.name

    try:
        before_img, after_aligned = load_and_align(before_path, after_path)
        mask, stats = detect_change(before_img, after_aligned, threshold=threshold)

        # Render overlay in memory; also persist to results/
        os.makedirs("results", exist_ok=True)
        overlay_path = os.path.join("results", "change_overlay.png")
        overlay_img = make_overlay(before_img, mask, overlay_path)
    finally:
        os.unlink(before_path)
        os.unlink(after_path)

    # Convert images to RGB for Streamlit; store as lists so st.cache_data
    # can serialise the dict (NumPy arrays are supported natively).
    return {
        "before_rgb":   cv2.cvtColor(before_img,  cv2.COLOR_BGR2RGB),
        "after_rgb":    cv2.cvtColor(after_aligned, cv2.COLOR_BGR2RGB),
        "overlay_rgb":  cv2.cvtColor(overlay_img,  cv2.COLOR_BGR2RGB),
        "mask":         mask,
        "overlay_bgr":  overlay_img,
        "stats":        stats,
    }


with st.spinner("Aligning images and detecting changes…"):
    result = _run_analysis(
        before_file.read(), before_file.name,
        after_file.read(),  after_file.name,
        threshold,
    )

stats = result["stats"]

# ---------------------------------------------------------------------------
# Three-column image view: Before | After | Overlay
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🖼️ Image Comparison")

img_c1, img_c2, img_c3 = st.columns(3)
with img_c1:
    st.caption("**Before**")
    st.image(result["before_rgb"], use_container_width=True)
with img_c2:
    st.caption("**After** (aligned)")
    st.image(result["after_rgb"], use_container_width=True)
with img_c3:
    st.caption("**Change overlay** — red = changed")
    st.image(result["overlay_rgb"], use_container_width=True)

# ---------------------------------------------------------------------------
# Stats metrics row
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Change Statistics")

largest_area = stats["regions"][0]["area_px"] if stats["regions"] else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Scene changed",     f"{stats['change_percent']}%")
m2.metric("Distinct regions",  stats["num_regions"])
m3.metric("Largest region",    f"{largest_area:,} px²")
m4.metric("Changed pixels",    f"{stats['changed_pixels']:,}")

# ---------------------------------------------------------------------------
# AI Narration — behind an explicit button so it only calls the API on demand
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🤖 AI Narration — IBM Granite")

narrate_btn = st.button("✨ Generate AI Narration", type="primary")

if narrate_btn:
    with st.spinner("Calling IBM Granite via watsonx.ai…"):
        narrative, source = narrate_with_granite(stats, date_range=date_range)

    source_label = (
        "🤖 Generated by **IBM Granite-3 8B Instruct** via watsonx.ai"
        if source == "granite"
        else "📋 **Template summary** (set WATSONX_API_KEY + WATSONX_PROJECT_ID to enable Granite)"
    )
    st.caption(source_label)
    st.info(narrative, icon="🛰️")
else:
    st.caption("Press the button above to generate a plain-language explanation of the detected changes.")

# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------

st.divider()
st.subheader("⬇️ Download Results")

dl1, dl2 = st.columns(2)

mask_png    = cv2.imencode(".png", result["mask"])[1].tobytes()
overlay_png = cv2.imencode(".png", result["overlay_bgr"])[1].tobytes()

dl1.download_button(
    "⬇️ Change mask (PNG)",
    data=mask_png,
    file_name="change_mask.png",
    mime="image/png",
    use_container_width=True,
)
dl2.download_button(
    "⬇️ Change overlay (PNG)",
    data=overlay_png,
    file_name="change_overlay.png",
    mime="image/png",
    use_container_width=True,
)
