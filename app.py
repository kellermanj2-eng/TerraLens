"""
app.py
------
Streamlit dashboard for TerraLens.

Layout
~~~~~~
  Sidebar  : mode toggle (Upload / Fetch), change threshold slider, date inputs
  Main     : Upload mode — two file uploaders
             Fetch mode  — Leaflet AOI map + date range + cloud-cover picker
             → three-column image view (Before / After / Overlay)
             → stats metrics row → "Generate AI Narration" button → narrative box
             → download buttons

Usage:
    streamlit run app.py
"""

import os
import tempfile

import cv2
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

from change_detection import (
    load_and_align, detect_change, overlay as make_overlay, regions_to_geojson,
)
from narrate import narrate_with_granite
from satellite_fetch import fetch_scene_pair, search_scenes, LAYER_MODIS, LAYER_LANDSAT

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
    "Detect meaningful surface changes between two satellite images and use "
    "**IBM Granite** to explain what happened on the ground."
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Image source",
        ["📂 Upload images", "🛰️ Fetch from NASA Worldview"],
        help="Upload your own before/after images, or let TerraLens automatically "
             "retrieve MODIS/Landsat imagery from NASA GIBS (no account needed).",
    )

    threshold = st.slider(
        "Change threshold",
        min_value=10, max_value=100, value=40, step=5,
        help="Pixel intensity difference (0-255) required to flag a change. "
             "Lower = more sensitive; higher = only major changes.",
    )

    if mode == "📂 Upload images":
        date_range = st.text_input(
            "Date range (optional)",
            placeholder="e.g. June 3 – July 12, 2025",
            help="Observation window shown in the Granite prompt.",
        )
    else:
        date_range = ""   # built from the fetch dates below

    st.divider()
    st.markdown(
        "**Data sources**\n"
        "- [NASA Worldview / GIBS](https://worldview.earthdata.nasa.gov/) (MODIS / Landsat)\n"
        "- [NASA CMR](https://cmr.earthdata.nasa.gov/) (granule search)\n"
    )
    st.markdown("**Output directory:** `results/`")

# ---------------------------------------------------------------------------
# Session state — persist fetched paths across Streamlit reruns
# (any button click, slider move, or widget interaction triggers a full rerun;
#  without session_state the fetched paths would reset to None each time)
# ---------------------------------------------------------------------------

for _key in ("fetched_before", "fetched_after",
             "fetched_before_meta", "fetched_after_meta",
             "fetch_date_range",
             "scene_bbox"):          # bbox persisted for GeoJSON + km² calc
    if _key not in st.session_state:
        st.session_state[_key] = None

# ---------------------------------------------------------------------------
# Mode: Upload images
# ---------------------------------------------------------------------------

before_file = after_file = None

if mode == "📂 Upload images":
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

    if not before_file or not after_file:
        st.divider()
        st.info(
            "👆 Upload a **Before** image and an **After** image above to begin analysis.\n\n"
            "**Tip:** Use the sidebar slider to tune sensitivity and optionally enter "
            "the date range for better-calibrated Granite commentary.\n\n"
            "Supported formats: PNG, JPEG, GeoTIFF (.tif/.tiff).",
            icon="🛰️",
        )
        st.stop()

# ---------------------------------------------------------------------------
# Mode: Fetch from NASA Worldview / GIBS
# ---------------------------------------------------------------------------

else:
    st.subheader("🛰️ NASA Worldview Auto-Fetch")
    st.markdown(
        "Draw a bounding box on the map (or enter coordinates manually), "
        "choose a date range, and TerraLens will automatically retrieve "
        "MODIS or Landsat imagery from NASA GIBS — **no account required**."
    )

    # --- Two-column layout: map left, controls right ---
    map_col, ctrl_col = st.columns([3, 2])

    with map_col:
        st.markdown("**Step 1 — Draw your area of interest**")
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
        Draw(
            export=False,
            draw_options={
                "rectangle": True,
                "polygon":   False,
                "circle":    False,
                "marker":    False,
                "circlemarker": False,
                "polyline":  False,
            },
        ).add_to(m)
        map_data = st_folium(m, height=400, width=None, key="aoi_map")

    with ctrl_col:
        st.markdown("**Step 2 — Or enter coordinates manually**")
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            min_lon = st.number_input("Min Lon", value=-10.0, format="%.4f")
            min_lat = st.number_input("Min Lat", value=35.0,  format="%.4f")
        with coord_col2:
            max_lon = st.number_input("Max Lon", value=10.0,  format="%.4f")
            max_lat = st.number_input("Max Lat", value=55.0,  format="%.4f")

        # Override with drawn bbox if available
        drawn = map_data.get("last_active_drawing") if map_data else None
        if drawn and drawn.get("geometry", {}).get("type") == "Polygon":
            coords = drawn["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            st.success(
                f"📍 Drawn bbox: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]"
            )

        st.markdown("**Step 3 — Set date windows**")
        date_from = st.date_input("Before window start", value=None, key="date_from")
        date_mid  = st.date_input("Cutoff (before → after)", value=None, key="date_mid")
        date_to   = st.date_input("After window end", value=None, key="date_to")

        nasa_layer = st.selectbox(
            "Imagery layer",
            options=[LAYER_MODIS, LAYER_LANDSAT],
            format_func=lambda x: "MODIS Terra (daily, global)" if x == LAYER_MODIS
                                   else "Landsat (annual composite)",
            help="MODIS is best for recent events; Landsat annual composite for year-to-year change.",
        )

        # --- Scene preview ---
        preview_btn = st.button("🔍 Preview available granules", use_container_width=True)
        if preview_btn and date_from and date_mid and date_to:
            bbox = (float(min_lon), float(min_lat), float(max_lon), float(max_lat))
            with st.spinner("Querying NASA CMR catalogue…"):
                try:
                    before_scenes = search_scenes(
                        bbox, str(date_from), str(date_mid), nasa_layer, limit=3
                    )
                    after_scenes = search_scenes(
                        bbox, str(date_mid), str(date_to), nasa_layer, limit=3
                    )
                    if before_scenes:
                        st.markdown("**Before window — top matches:**")
                        for s in before_scenes:
                            st.markdown(f"- `{s['date']}` · {s['source']} · `{s['name']}`")
                    else:
                        st.info("No CMR granules found — GIBS direct fetch will be used.")
                    if after_scenes:
                        st.markdown("**After window — top matches:**")
                        for s in after_scenes:
                            st.markdown(f"- `{s['date']}` · {s['source']} · `{s['name']}`")
                    else:
                        st.info("No CMR granules found — GIBS direct fetch will be used.")
                except Exception as exc:
                    st.error(f"Catalogue query failed: {exc}")

        # --- Fetch button ---
        fetch_btn = st.button(
            "⬇️ Fetch scene pair & analyse",
            type="primary",
            use_container_width=True,
            disabled=not (date_from and date_mid and date_to),
        )

    if "fetch_btn" in dir() and fetch_btn:
        bbox = (float(min_lon), float(min_lat), float(max_lon), float(max_lat))
        with st.spinner("Fetching imagery from NASA GIBS…"):
            try:
                (bp, ap, bm, am) = fetch_scene_pair(
                    bbox,
                    str(date_from), str(date_mid), str(date_to),
                    out_dir="data",
                    layer=nasa_layer,
                )
                # Persist to session_state so subsequent button clicks
                # (narrate, download) don't lose the fetched paths
                st.session_state.fetched_before      = bp
                st.session_state.fetched_after       = ap
                st.session_state.fetched_before_meta = bm
                st.session_state.fetched_after_meta  = am
                st.session_state.fetch_date_range    = f"{date_from} → {date_to}"
                st.session_state.scene_bbox          = bbox
                st.success(
                    f"✅ Downloaded:  \n"
                    f"**Before:** `{bm['date']}` ({bm['source']})  \n"
                    f"**After:**  `{am['date']}` ({am['source']})"
                )
            except RuntimeError as exc:
                st.error(f"⚠️ {exc}", icon="🛰️")
                st.stop()
            except Exception as exc:
                st.error(f"⚠️ Fetch failed: {exc}", icon="🛰️")
                st.stop()
    elif mode == "🛰️ Fetch from NASA Worldview" and st.session_state.fetched_before is None:
        st.divider()
        st.info(
            "👆 Draw an area of interest on the map, choose your date windows, "
            "and click **Fetch scene pair & analyse** to begin.",
            icon="🛰️",
        )
        st.stop()
    else:
        # Not a fresh fetch — show a reminder of what's loaded
        bm = st.session_state.fetched_before_meta
        am = st.session_state.fetched_after_meta
        if bm and am:
            st.success(
                f"✅ Using cached fetch:  \n"
                f"**Before:** `{bm['date']}` ({bm['source']})  \n"
                f"**After:**  `{am['date']}` ({am['source']})"
            )

# ---------------------------------------------------------------------------
# Resolve image bytes for the pipeline
# ---------------------------------------------------------------------------

if mode == "📂 Upload images":
    before_bytes = before_file.read()
    before_name  = before_file.name
    after_bytes  = after_file.read()
    after_name   = after_file.name
    date_range   = date_range  # already set from sidebar text input
else:
    # Read from session_state (persisted across all reruns)
    _fb = st.session_state.fetched_before
    _fa = st.session_state.fetched_after
    with open(_fb, "rb") as f:
        before_bytes = f.read()
    with open(_fa, "rb") as f:
        after_bytes = f.read()
    before_name = os.path.basename(_fb)
    after_name  = os.path.basename(_fa)
    date_range  = st.session_state.fetch_date_range or ""

# ---------------------------------------------------------------------------
# Run change-detection pipeline (cached per uploaded file pair + settings)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_analysis(before_bytes: bytes, before_name: str,
                  after_bytes: bytes,  after_name: str,
                  threshold: int,
                  scene_bbox: tuple | None = None) -> dict:
    """
    Write uploads to temporary files, run load_and_align + detect_change,
    and return a serialisable result dict.

    scene_bbox : optional (min_lon, min_lat, max_lon, max_lat) used to
                 compute changed_km2 and build the GeoJSON export.
    """
    import math as _math

    before_suffix = os.path.splitext(before_name)[1]
    after_suffix  = os.path.splitext(after_name)[1]

    before_path = after_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=before_suffix, delete=False) as f:
            f.write(before_bytes)
            before_path = f.name
        with tempfile.NamedTemporaryFile(suffix=after_suffix, delete=False) as f:
            f.write(after_bytes)
            after_path = f.name

        before_img, after_aligned, align_warnings = load_and_align(before_path, after_path)

        if before_img is None or after_aligned is None:
            return {"error": "One or both images could not be loaded. "
                             "Please check that the files are valid PNG, JPEG, or GeoTIFF images."}
        if before_img.shape != after_aligned.shape:
            return {"error": f"Image size mismatch after alignment: "
                             f"before={before_img.shape[:2]}, after={after_aligned.shape[:2]}. "
                             "Upload images of the same geographic area and similar resolution."}

        mask, stats = detect_change(before_img, after_aligned, threshold=threshold)

        # D — compute real-world km² if bbox is available
        if scene_bbox:
            min_lon, min_lat, max_lon, max_lat = scene_bbox
            mid_lat     = (min_lat + max_lat) / 2
            lon_span_km = abs(max_lon - min_lon) * 111.32 * _math.cos(_math.radians(mid_lat))
            lat_span_km = abs(max_lat - min_lat) * 110.574
            total_km2   = lon_span_km * lat_span_km
            stats["changed_km2"] = round(stats["changed_fraction"] * total_km2, 4)

        os.makedirs("results", exist_ok=True)
        overlay_path = os.path.join("results", "change_overlay.png")
        overlay_img = make_overlay(before_img, mask, overlay_path)

    except (FileNotFoundError, ValueError) as exc:
        return {"error": f"Could not read image: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Analysis failed: {exc}"}
    finally:
        for p in (before_path, after_path):
            if p and os.path.exists(p):
                os.unlink(p)

    return {
        "before_rgb":    cv2.cvtColor(before_img,  cv2.COLOR_BGR2RGB),
        "after_rgb":     cv2.cvtColor(after_aligned, cv2.COLOR_BGR2RGB),
        "overlay_rgb":   cv2.cvtColor(overlay_img,  cv2.COLOR_BGR2RGB),
        "mask":          mask,
        "overlay_bgr":   overlay_img,
        "stats":         stats,
        "warnings":      align_warnings,
        "image_shape":   before_img.shape[:2],  # (H, W) for GeoJSON
    }


# Resolve bbox for this run (None in upload mode unless user drew a box)
_scene_bbox = st.session_state.scene_bbox if mode == "🛰️ Fetch from NASA Worldview" else None

with st.spinner("Aligning images and detecting changes…"):
    result = _run_analysis(
        before_bytes, before_name,
        after_bytes,  after_name,
        threshold,
        scene_bbox=_scene_bbox,
    )

if "error" in result:
    st.error(f"⚠️ {result['error']}", icon="🛰️")
    st.stop()

for warn in result.get("warnings", []):
    st.warning(f"⚠️ {warn}", icon="🌥️")

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
changed_km2  = stats.get("changed_km2")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scene changed",    f"{stats['change_percent']}%")
m2.metric("Distinct regions", stats["num_regions"])
m3.metric("Largest region",   f"{largest_area:,} px²")
m4.metric("Changed pixels",   f"{stats['changed_pixels']:,}")
m5.metric("Changed area",     f"{changed_km2:.2f} km²" if changed_km2 is not None else "n/a (upload mode)")

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
# Multi-temporal trend chart  (Fetch mode only, ≥ 3 dates)
# ---------------------------------------------------------------------------

if mode == "🛰️ Fetch from NASA Worldview" and _scene_bbox:
    st.divider()
    st.subheader("📈 Multi-temporal Change Trend")
    st.markdown(
        "Fetch imagery across **3 or more dates** to see how change percentage "
        "evolved over time. Enter comma-separated dates below:"
    )

    trend_dates_input = st.text_input(
        "Dates (YYYY-MM-DD, comma-separated, oldest first)",
        placeholder="e.g. 2023-01-01, 2023-04-01, 2023-07-01, 2023-10-01",
        key="trend_dates",
    )
    trend_layer = st.selectbox(
        "Imagery layer",
        [LAYER_MODIS, LAYER_LANDSAT],
        format_func=lambda x: "MODIS Terra (daily)" if x == LAYER_MODIS else "Landsat (annual)",
        key="trend_layer",
    )
    trend_btn = st.button("📈 Build trend chart", use_container_width=True)

    if trend_btn and trend_dates_input:
        raw_dates = [d.strip() for d in trend_dates_input.split(",") if d.strip()]
        if len(raw_dates) < 3:
            st.warning("Enter at least 3 dates to build a trend.")
        else:
            trend_points: list[dict] = []
            prog = st.progress(0, text="Fetching scenes…")
            total_pairs = len(raw_dates) - 1
            for i in range(total_pairs):
                d_before = raw_dates[i]
                d_after  = raw_dates[i + 1]
                prog.progress((i) / total_pairs, text=f"Fetching {d_before} → {d_after}…")
                try:
                    from satellite_fetch import download_scene
                    b_scene = {"date": d_before, "bbox": _scene_bbox, "layer": trend_layer, "name": f"trend_{d_before}"}
                    a_scene = {"date": d_after,  "bbox": _scene_bbox, "layer": trend_layer, "name": f"trend_{d_after}"}
                    bp = download_scene(b_scene, out_dir="data", layer=trend_layer)
                    ap = download_scene(a_scene, out_dir="data", layer=trend_layer)
                    bimg, aimg, _ = load_and_align(bp, ap)
                    _, tstats = detect_change(bimg, aimg, threshold=threshold)
                    trend_points.append({
                        "period": f"{d_before[:7]} → {d_after[:7]}",
                        "change_pct": tstats["change_percent"],
                    })
                except Exception as exc:
                    st.warning(f"Skipped {d_before} → {d_after}: {exc}")
            prog.progress(1.0, text="Done.")

            if trend_points:
                import pandas as pd
                df = pd.DataFrame(trend_points)
                st.line_chart(df.set_index("period")["change_pct"], use_container_width=True)
                st.caption("Y-axis: % of scene pixels flagged as changed between consecutive date pairs.")

# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------

st.divider()
st.subheader("⬇️ Download Results")

mask_png    = cv2.imencode(".png", result["mask"])[1].tobytes()
overlay_png = cv2.imencode(".png", result["overlay_bgr"])[1].tobytes()

dl1, dl2, dl3 = st.columns(3)

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

# A — GeoJSON download (only when bbox is known)
_geojson_bbox = _scene_bbox
if _geojson_bbox and stats["regions"]:
    import json as _json
    geojson = regions_to_geojson(
        stats["regions"],
        bbox=_geojson_bbox,
        image_shape=result["image_shape"],
    )
    dl3.download_button(
        "⬇️ Changed regions (GeoJSON)",
        data=_json.dumps(geojson, indent=2),
        file_name="changed_regions.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )
else:
    dl3.info("GeoJSON export available in **Fetch mode** after analysis.", icon="🗺️")
