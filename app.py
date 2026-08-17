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
    compute_ndvi_diff, compute_false_colour, FALSE_COLOUR_PRESETS,
)
from narrate import narrate_with_granite
from catalogue import add_entry as _cat_add, list_entries as _cat_list, catalogue_stats as _cat_stats
from satellite_fetch import fetch_scene_pair, search_scenes, LAYER_MODIS, LAYER_LANDSAT
from sentinel2_fetch import (
    search_sentinel2_scenes, fetch_sentinel2_pair,
    credentials_available as s2_credentials_available,
    CopernicusAuthError, CopernicusSearchError,
)

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
        ["📂 Upload images", "🛰️ Fetch from NASA Worldview", "🌍 Fetch from Copernicus (Sentinel-2)"],
        help=(
            "Upload your own images, retrieve MODIS/Landsat from NASA GIBS "
            "(no account), or fetch 10 m Sentinel-2 L2A from Copernicus "
            "(free account required)."
        ),
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

    # Copernicus credential status badge (sidebar)
    if mode == "🌍 Fetch from Copernicus (Sentinel-2)":
        if s2_credentials_available():
            st.success("✅ Copernicus credentials found", icon="🔑")
        else:
            st.warning(
                "⚠️ CDSE_USER / CDSE_PASSWORD not set.  "
                "Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) "
                "and add credentials to your `.env` file.",
                icon="🔑",
            )

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
             "scene_bbox",           # bbox persisted for GeoJSON + km² calc
             "s2_fetched_before", "s2_fetched_after",
             "s2_fetched_before_meta", "s2_fetched_after_meta",
             "s2_fetch_date_range", "s2_scene_bbox",
             "s2_before_scl", "s2_after_scl"):   # SCL sidecar paths for cloud masking
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
# Preset landmark events  (shared across all fetch modes)
# ---------------------------------------------------------------------------

# Each entry: display label → dict with bbox, date_from, date_mid, date_to,
#   layer ("modis"/"landsat"/"sentinel2"), description, and recommended_layer.
_PRESET_EVENTS = {
    "— choose a preset event —": None,
    "🔥 2023 Maui (Lahaina) Wildfire — Hawaii, USA": {
        "bbox":      (-156.72, 20.82, -156.60, 20.92),
        "date_from": "2023-08-01", "date_mid": "2023-08-09", "date_to": "2023-08-20",
        "layer":     "modis",
        "desc":      "The August 8 2023 Lahaina fire destroyed over 2 200 acres and caused 100+ deaths — one of the deadliest US wildfires in a century.",
    },
    "🌊 2011 Tōhoku Tsunami — Sendai coast, Japan": {
        "bbox":      (140.8, 38.0, 141.4, 38.6),
        "date_from": "2011-02-01", "date_mid": "2011-03-11", "date_to": "2011-04-01",
        "layer":     "modis",
        "desc":      "The Mw 9.1 earthquake and subsequent tsunami inundated ~560 km² of coastal Japan on 11 March 2011.",
    },
    "🌲 Amazon Deforestation — Rondônia, Brazil": {
        "bbox":      (-63.5, -11.0, -62.0, -9.5),
        "date_from": "2000-01-01", "date_mid": "2001-01-01", "date_to": "2002-01-01",
        "layer":     "landsat",
        "desc":      "Rondônia lost more primary forest than any other Brazilian state during the early 2000s expansion of soy and cattle farming.",
    },
    "🌋 2018 Kīlauea Lava Flow — Big Island, Hawaii": {
        "bbox":      (-154.97, 19.42, -154.82, 19.52),
        "date_from": "2018-04-15", "date_mid": "2018-05-05", "date_to": "2018-06-15",
        "layer":     "modis",
        "desc":      "The Lower East Rift Zone eruption destroyed 700+ homes in Leilani Estates and covered farmland with fresh lava flows.",
    },
    "🏙️ Dubai Urban Expansion — UAE": {
        "bbox":      (55.0, 25.0, 55.5, 25.4),
        "date_from": "2000-01-01", "date_mid": "2005-01-01", "date_to": "2010-01-01",
        "layer":     "landsat",
        "desc":      "Dubai's built-up area expanded dramatically in the 2000s — a textbook example of rapid urban growth visible from space.",
    },
    "❄️ Jakobshavn Glacier Retreat — Greenland": {
        "bbox":      (-50.5, 69.0, -48.5, 69.5),
        "date_from": "2000-01-01", "date_mid": "2007-01-01", "date_to": "2014-01-01",
        "layer":     "landsat",
        "desc":      "Jakobshavn Isbræ has retreated ~40 km since 2000 and is one of the fastest-moving glaciers on Earth.",
    },
    "💧 2019 Midwest USA Flooding — Missouri River": {
        "bbox":      (-96.5, 41.2, -95.5, 41.8),
        "date_from": "2019-02-01", "date_mid": "2019-03-17", "date_to": "2019-04-30",
        "layer":     "modis",
        "desc":      "Record spring flooding on the Missouri and Platte rivers inundated farmland and infrastructure across Nebraska and Iowa.",
    },
    "🌾 2020 Beirut Explosion — Lebanon": {
        "bbox":      (35.49, 33.88, 35.55, 33.92),
        "date_from": "2020-07-15", "date_mid": "2020-08-04", "date_to": "2020-08-15",
        "layer":     "modis",
        "desc":      "The August 4 2020 port explosion destroyed the harbour and surrounding districts — dramatic before/after surface change.",
    },
}


# ---------------------------------------------------------------------------
# Mode: Fetch from NASA Worldview / GIBS
# ---------------------------------------------------------------------------

if mode == "🛰️ Fetch from NASA Worldview":
    st.subheader("🛰️ NASA Worldview Auto-Fetch")
    st.markdown(
        "Draw a bounding box on the map (or enter coordinates manually), "
        "choose a date range, and TerraLens will automatically retrieve "
        "MODIS or Landsat imagery from NASA GIBS — **no account required**."
    )

    # --- Quick-select preset events ---
    _nasa_preset_label = st.selectbox(
        "⚡ Quick-select a landmark event",
        list(_PRESET_EVENTS.keys()),
        key="nasa_preset",
        help="Fills in coordinates and dates automatically. You can still edit them below.",
    )
    _nasa_preset = _PRESET_EVENTS.get(_nasa_preset_label)
    if _nasa_preset:
        st.info(f"📍 **{_nasa_preset_label.split('—')[0].strip()}** — {_nasa_preset['desc']}", icon="🌍")
        # Inject into session state so the inputs below pick up the values
        _pb = _nasa_preset["bbox"]
        st.session_state["nasa_min_lon"] = float(_pb[0])
        st.session_state["nasa_min_lat"] = float(_pb[1])
        st.session_state["nasa_max_lon"] = float(_pb[2])
        st.session_state["nasa_max_lat"] = float(_pb[3])
        from datetime import date as _date
        st.session_state["date_from"] = _date.fromisoformat(_nasa_preset["date_from"])
        st.session_state["date_mid"]  = _date.fromisoformat(_nasa_preset["date_mid"])
        st.session_state["date_to"]   = _date.fromisoformat(_nasa_preset["date_to"])

    # --- Two-column layout: map left, controls right ---
    map_col, ctrl_col = st.columns([3, 2])

    with map_col:
        st.markdown("**Step 1 — Draw your area of interest**")
        _nasa_map_center = (
            [(_nasa_preset["bbox"][1] + _nasa_preset["bbox"][3]) / 2,
             (_nasa_preset["bbox"][0] + _nasa_preset["bbox"][2]) / 2]
            if _nasa_preset else [20, 0]
        )
        m = folium.Map(location=_nasa_map_center, zoom_start=7 if _nasa_preset else 2,
                       tiles="CartoDB positron")
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
            min_lon = st.number_input("Min Lon", value=st.session_state.get("nasa_min_lon", -10.0), format="%.4f", key="nasa_min_lon")
            min_lat = st.number_input("Min Lat", value=st.session_state.get("nasa_min_lat",  35.0), format="%.4f", key="nasa_min_lat")
        with coord_col2:
            max_lon = st.number_input("Max Lon", value=st.session_state.get("nasa_max_lon",  10.0), format="%.4f", key="nasa_max_lon")
            max_lat = st.number_input("Max Lat", value=st.session_state.get("nasa_max_lat",  55.0), format="%.4f", key="nasa_max_lat")

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
# Mode: Fetch from Copernicus / Sentinel-2
# ---------------------------------------------------------------------------

elif mode == "🌍 Fetch from Copernicus (Sentinel-2)":
    st.subheader("🌍 Sentinel-2 L2A — Copernicus Data Space")
    st.markdown(
        "Search and download **10 m resolution** Sentinel-2 L2A imagery from "
        "[Copernicus Data Space](https://dataspace.copernicus.eu/).  "
        "Requires a **free CDSE account** — add `CDSE_USER` and `CDSE_PASSWORD` "
        "to your `.env` file to enable downloads."
    )

    if not s2_credentials_available():
        st.warning(
            "🔑 **Copernicus credentials required.**  "
            "Register for free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/), "
            "then set `CDSE_USER` and `CDSE_PASSWORD` in your `.env` file.",
            icon="🔑",
        )

    # Sentinel-2 specific presets (post-2015, 10 m resolution events)
    _S2_PRESET_EVENTS = {
        "— choose a preset event —": None,
        "🔥 2021 Dixie Fire — Northern California, USA": {
            "bbox":      (-121.4, 40.0, -120.6, 40.6),
            "date_from": "2021-07-01", "date_mid": "2021-07-14", "date_to": "2021-09-15",
            "desc":      "California's largest single wildfire on record — burned nearly 1 million acres across Plumas and Butte counties.",
        },
        "🌊 2023 Libya Floods — Derna, Libya": {
            "bbox":      (22.55, 32.70, 22.75, 32.82),
            "date_from": "2023-09-01", "date_mid": "2023-09-11", "date_to": "2023-09-25",
            "desc":      "Storm Daniel caused two dams to burst on 11 Sep 2023, wiping out entire neighbourhoods of Derna and killing thousands.",
        },
        "🏗️ 2022 Aral Sea Desiccation — Kazakhstan/Uzbekistan": {
            "bbox":      (58.0, 43.5, 61.0, 46.0),
            "date_from": "2019-06-01", "date_mid": "2020-06-01", "date_to": "2022-06-01",
            "desc":      "The South Aral Sea has nearly vanished — one of the world's most dramatic human-induced landscape changes.",
        },
        "🌲 2019 Amazon Fires — Pará State, Brazil": {
            "bbox":      (-53.5, -4.5, -52.0, -3.0),
            "date_from": "2019-07-01", "date_mid": "2019-08-20", "date_to": "2019-09-30",
            "desc":      "Fires in Pará during the record 2019 Amazon fire season — visible burn scars at 10 m resolution.",
        },
        "🏙️ 2016 Mosul Urban Damage — Iraq": {
            "bbox":      (43.05, 36.28, 43.22, 36.40),
            "date_from": "2016-06-01", "date_mid": "2017-01-01", "date_to": "2017-07-15",
            "desc":      "Satellite imagery captured large-scale urban destruction during the battle to retake Mosul from ISIL.",
        },
        "❄️ 2022 Pine Island Glacier Calving — Antarctica": {
            "bbox":      (-101.0, -75.5, -99.0, -74.8),
            "date_from": "2022-01-01", "date_mid": "2022-06-01", "date_to": "2022-12-01",
            "desc":      "Pine Island Glacier calved a large iceberg in 2022 — Sentinel-2 captures ice-front retreat at 10 m detail.",
        },
        "💥 2020 Beirut Port Explosion — Lebanon": {
            "bbox":      (35.505, 33.895, 35.545, 33.915),
            "date_from": "2020-07-20", "date_mid": "2020-08-04", "date_to": "2020-08-12",
            "desc":      "The August 4 2020 ammonium nitrate explosion — at 10 m resolution Sentinel-2 shows the crater and structural destruction clearly.",
        },
        "🌾 2022 Ukraine Cropland Loss — Kherson Oblast": {
            "bbox":      (33.0, 46.5, 34.5, 47.2),
            "date_from": "2021-06-01", "date_mid": "2022-06-01", "date_to": "2022-09-01",
            "desc":      "Agricultural land abandonment and damage in Kherson Oblast following the 2022 Russian invasion, visible as reduced crop coverage.",
        },
    }

    _s2_preset_label = st.selectbox(
        "⚡ Quick-select a landmark event",
        list(_S2_PRESET_EVENTS.keys()),
        key="s2_preset",
        help="Fills in coordinates and dates for a real-world event. You can edit them below.",
    )
    _s2_preset = _S2_PRESET_EVENTS.get(_s2_preset_label)
    if _s2_preset:
        st.info(f"📍 **{_s2_preset_label.split('—')[0].strip()}** — {_s2_preset['desc']}", icon="🌍")
        from datetime import date as _date2
        _s2pb = _s2_preset["bbox"]
        st.session_state["s2_min_lon"]   = float(_s2pb[0])
        st.session_state["s2_min_lat"]   = float(_s2pb[1])
        st.session_state["s2_max_lon"]   = float(_s2pb[2])
        st.session_state["s2_max_lat"]   = float(_s2pb[3])
        st.session_state["s2_date_from"] = _date2.fromisoformat(_s2_preset["date_from"])
        st.session_state["s2_date_mid"]  = _date2.fromisoformat(_s2_preset["date_mid"])
        st.session_state["s2_date_to"]   = _date2.fromisoformat(_s2_preset["date_to"])

    s2_map_col, s2_ctrl_col = st.columns([3, 2])

    with s2_map_col:
        st.markdown("**Step 1 — Draw your area of interest**")
        _s2_map_center = (
            [(_s2_preset["bbox"][1] + _s2_preset["bbox"][3]) / 2,
             (_s2_preset["bbox"][0] + _s2_preset["bbox"][2]) / 2]
            if _s2_preset else [20, 0]
        )
        s2_m = folium.Map(location=_s2_map_center, zoom_start=9 if _s2_preset else 2,
                          tiles="CartoDB positron")
        Draw(
            export=False,
            draw_options={
                "rectangle": True, "polygon": False, "circle": False,
                "marker": False, "circlemarker": False, "polyline": False,
            },
        ).add_to(s2_m)
        s2_map_data = st_folium(s2_m, height=400, width=None, key="s2_aoi_map")

    with s2_ctrl_col:
        st.markdown("**Step 2 — Or enter coordinates manually**")
        s2_cc1, s2_cc2 = st.columns(2)
        with s2_cc1:
            s2_min_lon = st.number_input("Min Lon", value=-2.0,  format="%.4f", key="s2_min_lon")
            s2_min_lat = st.number_input("Min Lat", value=51.0,  format="%.4f", key="s2_min_lat")
        with s2_cc2:
            s2_max_lon = st.number_input("Max Lon", value=0.0,   format="%.4f", key="s2_max_lon")
            s2_max_lat = st.number_input("Max Lat", value=52.5,  format="%.4f", key="s2_max_lat")

        s2_drawn = s2_map_data.get("last_active_drawing") if s2_map_data else None
        if s2_drawn and s2_drawn.get("geometry", {}).get("type") == "Polygon":
            s2_coords = s2_drawn["geometry"]["coordinates"][0]
            s2_lons = [c[0] for c in s2_coords]
            s2_lats = [c[1] for c in s2_coords]
            s2_min_lon, s2_max_lon = min(s2_lons), max(s2_lons)
            s2_min_lat, s2_max_lat = min(s2_lats), max(s2_lats)
            st.success(
                f"📍 Drawn bbox: [{s2_min_lon:.3f}, {s2_min_lat:.3f}, "
                f"{s2_max_lon:.3f}, {s2_max_lat:.3f}]"
            )

        st.markdown("**Step 3 — Set date windows**")
        s2_date_from = st.date_input("Before window start", value=None, key="s2_date_from")
        s2_date_mid  = st.date_input("Cutoff (before → after)", value=None, key="s2_date_mid")
        s2_date_to   = st.date_input("After window end", value=None, key="s2_date_to")

        s2_cloud = st.slider(
            "Max cloud cover %", min_value=0, max_value=100, value=30, step=5,
            key="s2_cloud",
            help="Only return scenes with cloud cover below this threshold.",
        )

        s2_bands_options = {
            "B04 + B08 (Red + NIR — NDVI ready)": ["B04", "B08"],
            "B02 + B03 + B04 + B08 (RGB + NIR)":  ["B02", "B03", "B04", "B08"],
        }
        s2_band_choice = st.selectbox(
            "Bands to download",
            list(s2_bands_options.keys()),
            key="s2_bands",
            help="B04+B08 is fastest and sufficient for NDVI.  "
                 "Add B02/B03 for a true-colour composite.",
        )
        s2_bands = s2_bands_options[s2_band_choice]

        s2_use_scl = st.checkbox(
            "☁️ Apply SCL cloud mask",
            value=True,
            key="s2_use_scl",
            help=(
                "Download the Scene Classification Layer (SCL) sidecar and use it "
                "to zero out cloud, cloud-shadow, cirrus, and saturated pixels in "
                "the difference image before thresholding.  Reduces false-positive "
                "change detections in scenes with partial cloud cover."
            ),
        )

        # Search preview
        s2_preview_btn = st.button("🔍 Preview available scenes", use_container_width=True, key="s2_preview")
        if s2_preview_btn and s2_date_from and s2_date_mid and s2_date_to:
            s2_bbox = (float(s2_min_lon), float(s2_min_lat), float(s2_max_lon), float(s2_max_lat))
            with st.spinner("Searching Copernicus catalogue (no auth required)…"):
                try:
                    s2_before_scenes = search_sentinel2_scenes(
                        s2_bbox, str(s2_date_from), str(s2_date_mid),
                        max_cloud_pct=s2_cloud, limit=3
                    )
                    s2_after_scenes = search_sentinel2_scenes(
                        s2_bbox, str(s2_date_mid), str(s2_date_to),
                        max_cloud_pct=s2_cloud, limit=3
                    )
                    if s2_before_scenes:
                        st.markdown("**Before window — top matches:**")
                        for s in s2_before_scenes:
                            st.markdown(
                                f"- `{s['date']}` · cloud {s['cloud_pct']:.0f}% · `{s['name'][:40]}…`"
                                if s['cloud_pct'] is not None
                                else f"- `{s['date']}` · `{s['name'][:40]}…`"
                            )
                    else:
                        st.info("No scenes found in before window — try different dates or cloud limit.")
                    if s2_after_scenes:
                        st.markdown("**After window — top matches:**")
                        for s in s2_after_scenes:
                            st.markdown(
                                f"- `{s['date']}` · cloud {s['cloud_pct']:.0f}% · `{s['name'][:40]}…`"
                                if s['cloud_pct'] is not None
                                else f"- `{s['date']}` · `{s['name'][:40]}…`"
                            )
                    else:
                        st.info("No scenes found in after window — try different dates or cloud limit.")
                except CopernicusSearchError as exc:
                    st.error(f"Catalogue search failed: {exc}")

        # Download button — disabled without credentials
        s2_fetch_btn = st.button(
            "⬇️ Fetch Sentinel-2 pair & analyse",
            type="primary",
            use_container_width=True,
            key="s2_fetch",
            disabled=not (s2_date_from and s2_date_mid and s2_date_to
                          and s2_credentials_available()),
        )

    if "s2_fetch_btn" in dir() and s2_fetch_btn:
        s2_bbox = (float(s2_min_lon), float(s2_min_lat), float(s2_max_lon), float(s2_max_lat))
        with st.spinner("Authenticating with Copernicus and downloading scenes…"):
            try:
                (s2_bp, s2_ap, s2_bm, s2_am) = fetch_sentinel2_pair(
                    s2_bbox,
                    str(s2_date_from), str(s2_date_mid), str(s2_date_to),
                    out_dir="data",
                    max_cloud_pct=float(s2_cloud),
                    bands=s2_bands,
                    include_scl=s2_use_scl,
                )
                st.session_state.s2_fetched_before      = s2_bp
                st.session_state.s2_fetched_after       = s2_ap
                st.session_state.s2_fetched_before_meta = s2_bm
                st.session_state.s2_fetched_after_meta  = s2_am
                st.session_state.s2_fetch_date_range    = f"{s2_date_from} → {s2_date_to}"
                st.session_state.s2_scene_bbox          = s2_bbox
                # Store SCL sidecar paths (None when include_scl=False)
                st.session_state.s2_before_scl = s2_bm.get("scl_path")
                st.session_state.s2_after_scl  = s2_am.get("scl_path")
                cloud_b = f"{s2_bm['cloud_pct']:.0f}%" if s2_bm.get("cloud_pct") is not None else "n/a"
                cloud_a = f"{s2_am['cloud_pct']:.0f}%" if s2_am.get("cloud_pct") is not None else "n/a"
                scl_note = " · SCL cloud mask ✅" if s2_use_scl else ""
                st.success(
                    f"✅ Downloaded Sentinel-2 pair{scl_note}:  \n"
                    f"**Before:** `{s2_bm['date']}` · cloud {cloud_b}  \n"
                    f"**After:**  `{s2_am['date']}` · cloud {cloud_a}"
                )
            except CopernicusAuthError as exc:
                st.error(f"🔑 {exc}", icon="🔑")
                st.stop()
            except RuntimeError as exc:
                st.error(f"⚠️ {exc}", icon="🌍")
                st.stop()
            except Exception as exc:
                st.error(f"⚠️ Sentinel-2 fetch failed: {exc}", icon="🌍")
                st.stop()
    elif st.session_state.s2_fetched_before is None:
        st.divider()
        st.info(
            "👆 Draw an AOI, set date windows, and click **Fetch Sentinel-2 pair & analyse**.  \n"
            "ℹ️ The catalogue search (Preview button) works without credentials.  "
            "Only the download step requires a Copernicus account.",
            icon="🌍",
        )
        st.stop()
    else:
        s2_bm = st.session_state.s2_fetched_before_meta
        s2_am = st.session_state.s2_fetched_after_meta
        if s2_bm and s2_am:
            st.success(
                f"✅ Using cached Sentinel-2 fetch:  \n"
                f"**Before:** `{s2_bm['date']}` ({s2_bm['source']})  \n"
                f"**After:**  `{s2_am['date']}` ({s2_am['source']})"
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
elif mode == "🛰️ Fetch from NASA Worldview":
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
else:
    # Sentinel-2 mode — read from s2_* session state keys
    _fb = st.session_state.s2_fetched_before
    _fa = st.session_state.s2_fetched_after
    with open(_fb, "rb") as f:
        before_bytes = f.read()
    with open(_fa, "rb") as f:
        after_bytes = f.read()
    before_name = os.path.basename(_fb)
    after_name  = os.path.basename(_fa)
    date_range  = st.session_state.s2_fetch_date_range or ""

# ---------------------------------------------------------------------------
# Run change-detection pipeline (cached per uploaded file pair + settings)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_analysis(before_bytes: bytes, before_name: str,
                  after_bytes: bytes,  after_name: str,
                  threshold: int,
                  scene_bbox: tuple | None = None,
                  before_scl_path: str | None = None,
                  after_scl_path:  str | None = None) -> dict:
    """
    Write uploads to temporary files, run load_and_align + detect_change,
    and return a serialisable result dict.

    scene_bbox      : optional (min_lon, min_lat, max_lon, max_lat) used to
                      compute changed_km2 and build the GeoJSON export.
    before_scl_path : optional path to the before-scene SCL GeoTIFF sidecar.
                      When provided, apply_scl_mask() is called and the resulting
                      cloud mask is passed to detect_change() so that cloudy/
                      shadowed pixels cannot trigger false-positive detections.
    after_scl_path  : same for the after scene.  The union of both masks is used.
    """
    import math as _math
    from change_detection import apply_scl_mask

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

        # --- Cloud mask (SCL) ---
        import numpy as _np
        cloud_mask   = None
        scl_warnings = []
        if before_scl_path or after_scl_path:
            target_shape = before_img.shape[:2]  # (H, W)
            masks = []
            for scl_p, label in ((before_scl_path, "before"), (after_scl_path, "after")):
                if scl_p and os.path.exists(scl_p):
                    try:
                        cm, scl_stats = apply_scl_mask(scl_p, target_shape)
                        masks.append(cm)
                        if scl_stats["cloud_fraction"] > 0.5:
                            scl_warnings.append(
                                f"SCL ({label}): {scl_stats['masked_pct']:.1f}% of pixels masked "
                                f"— scene is heavily cloud-covered."
                            )
                    except Exception as _e:
                        scl_warnings.append(f"SCL ({label}) could not be loaded: {_e}")
                elif scl_p:
                    scl_warnings.append(f"SCL ({label}) file not found: {scl_p}")
            if masks:
                # Union of both masks — a pixel excluded in either scene is suppressed
                cloud_mask = _np.zeros(target_shape, dtype=_np.uint8)
                for m in masks:
                    cloud_mask = _np.where(m == 255, _np.uint8(255), cloud_mask)

        mask, stats = detect_change(before_img, after_aligned, threshold=threshold,
                                    cloud_mask=cloud_mask)

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
        "warnings":      align_warnings + scl_warnings,
        "image_shape":   before_img.shape[:2],  # (H, W) for GeoJSON
    }


# Resolve bbox and SCL paths for this run
if mode == "🛰️ Fetch from NASA Worldview":
    _scene_bbox = st.session_state.scene_bbox
    _before_scl = None
    _after_scl  = None
elif mode == "🌍 Fetch from Copernicus (Sentinel-2)":
    _scene_bbox = st.session_state.s2_scene_bbox
    _before_scl = st.session_state.s2_before_scl
    _after_scl  = st.session_state.s2_after_scl
else:
    _scene_bbox = None
    _before_scl = None
    _after_scl  = None

with st.spinner("Aligning images and detecting changes…"):
    result = _run_analysis(
        before_bytes, before_name,
        after_bytes,  after_name,
        threshold,
        scene_bbox=_scene_bbox,
        before_scl_path=_before_scl,
        after_scl_path=_after_scl,
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
# Swipe viewer — drag the divider to compare Before vs After
# ---------------------------------------------------------------------------

import base64 as _base64

def _numpy_to_b64png(arr) -> str:
    """Encode an H×W×3 uint8 RGB numpy array as a base64 PNG data-URI."""
    import cv2 as _cv2
    ok, buf = _cv2.imencode(".png", _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + _base64.b64encode(buf.tobytes()).decode()

_b64_before = _numpy_to_b64png(result["before_rgb"])
_b64_after  = _numpy_to_b64png(result["after_rgb"])

_swipe_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e1117; font-family: system-ui, sans-serif; }}
  #swipe-container {{
    position: relative;
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
    overflow: hidden;
    cursor: ew-resize;
    border-radius: 6px;
    border: 1px solid #30363d;
    user-select: none;
    -webkit-user-select: none;
  }}
  #swipe-container img {{
    display: block;
    width: 100%;
    height: auto;
    pointer-events: none;
  }}
  #after-img {{
    position: absolute;
    top: 0; left: 0;
    clip-path: inset(0 50% 0 0);
    transition: none;
  }}
  #divider {{
    position: absolute;
    top: 0; bottom: 0;
    left: 50%;
    width: 3px;
    background: #ffffff;
    cursor: ew-resize;
    transform: translateX(-50%);
    z-index: 10;
  }}
  #handle {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 36px; height: 36px;
    background: #ffffff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    color: #0e1117;
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    cursor: ew-resize;
    pointer-events: none;
  }}
  .label {{
    position: absolute;
    top: 8px;
    padding: 3px 8px;
    background: rgba(0,0,0,0.55);
    color: #fff;
    font-size: 12px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 5;
  }}
  #label-before {{ left: 8px; }}
  #label-after  {{ left: 8px; display: none; }}
</style>
</head>
<body>
<div id="swipe-container">
  <img id="before-img" src="{_b64_before}" alt="Before"/>
  <img id="after-img"  src="{_b64_after}"  alt="After (aligned)"/>
  <div id="divider"><div id="handle">⇔</div></div>
  <span class="label" id="label-before">Before</span>
  <span class="label" id="label-after">After</span>
</div>
<p style="text-align:center; color:#8b949e; font-size:11px; margin-top:6px;">
  Drag the divider ⇔ to compare Before vs After (aligned)
</p>
<script>
(function() {{
  var container  = document.getElementById("swipe-container");
  var afterImg   = document.getElementById("after-img");
  var dividerEl  = document.getElementById("divider");
  var labelAfter = document.getElementById("label-after");
  var dragging   = false;

  function setPos(fraction) {{
    fraction = Math.max(0.02, Math.min(0.98, fraction));
    var pct = (fraction * 100).toFixed(2);
    afterImg.style.clipPath = "inset(0 " + (100 - fraction * 100).toFixed(2) + "% 0 0)";
    dividerEl.style.left    = pct + "%";
    // Show "After" label only when after panel is wide enough
    labelAfter.style.display = fraction > 0.12 ? "block" : "none";
    labelAfter.style.left    = "8px";
  }}

  function getX(e) {{
    return (e.touches ? e.touches[0].clientX : e.clientX);
  }}

  container.addEventListener("mousedown",  function(e) {{ dragging = true; e.preventDefault(); }});
  container.addEventListener("touchstart", function(e) {{ dragging = true; }}, {{passive: true}});

  window.addEventListener("mousemove", function(e) {{
    if (!dragging) return;
    var rect = container.getBoundingClientRect();
    setPos((getX(e) - rect.left) / rect.width);
  }});
  window.addEventListener("touchmove", function(e) {{
    if (!dragging) return;
    var rect = container.getBoundingClientRect();
    setPos((getX(e) - rect.left) / rect.width);
  }}, {{passive: true}});

  window.addEventListener("mouseup",  function() {{ dragging = false; }});
  window.addEventListener("touchend", function() {{ dragging = false; }});

  setPos(0.5);
}})();
</script>
</body>
</html>"""

st.subheader("🔀 Swipe Comparison")
st.caption("Drag the ⇔ handle left or right to reveal the **Before** image underneath the **After** image.")
import streamlit.components.v1 as _components
_components.html(_swipe_html, height=520, scrolling=False)

# ---------------------------------------------------------------------------
# Stats metrics row
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Change Statistics")

largest_area     = stats["regions"][0]["area_px"] if stats["regions"] else 0
changed_km2      = stats.get("changed_km2")
cloud_masked_pct = stats.get("cloud_masked_pct", 0.0)

if cloud_masked_pct > 0:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
else:
    m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scene changed",    f"{stats['change_percent']}%")
m2.metric("Distinct regions", stats["num_regions"])
m3.metric("Largest region",   f"{largest_area:,} px²")
m4.metric("Changed pixels",   f"{stats['changed_pixels']:,}")
m5.metric("Changed area",     f"{changed_km2:.2f} km²" if changed_km2 is not None else "n/a (upload mode)")
if cloud_masked_pct > 0:
    m6.metric("Cloud masked", f"{cloud_masked_pct:.1f}%",
              help="Percentage of pixels suppressed by the SCL cloud mask before change detection.")

# ---------------------------------------------------------------------------
# AI Narration — behind an explicit button so it only calls the API on demand
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🤖 AI Narration — IBM Granite (few-shot classifier)")

narrate_btn = st.button("✨ Generate AI Narration", type="primary")

if narrate_btn:
    with st.spinner("Calling IBM Granite via watsonx.ai…"):
        narrative, source, classified = narrate_with_granite(stats, date_range=date_range)

    source_label = (
        "🤖 Generated by **IBM Granite-3 8B Instruct** via watsonx.ai"
        if source == "granite"
        else "📋 **Template summary** (set WATSONX_API_KEY + WATSONX_PROJECT_ID to enable Granite)"
    )
    st.caption(source_label)

    # Classifier badge row
    _ct = classified.get("change_type", "unknown")
    _cf = classified.get("confidence", "Low")
    _badge_colour = {"High": "green", "Medium": "orange", "Low": "red"}.get(_cf, "grey")
    _caveat = classified.get("caveat", "")
    cls_col1, cls_col2 = st.columns(2)
    cls_col1.metric("Detected change type", _ct.title() if _ct else "Unknown")
    cls_col2.metric("Confidence", _cf)
    if _caveat:
        st.caption(f"💡 {_caveat}")

    st.info(narrative, icon="🛰️")

    # Auto-save to catalogue
    try:
        _mode_label = (
            "nasa"      if mode == "🛰️ Fetch from NASA Worldview" else
            "sentinel2" if mode == "🌍 Fetch from Copernicus (Sentinel-2)" else
            "upload"
        )
        _cat_add({
            "before_name":    before_name,
            "after_name":     after_name,
            "date_range":     date_range,
            "mode":           _mode_label,
            "change_percent": stats.get("change_percent"),
            "num_regions":    stats.get("num_regions"),
            "changed_pixels": stats.get("changed_pixels"),
            "total_pixels":   stats.get("total_pixels"),
            "changed_km2":    stats.get("changed_km2"),
            "change_type":    classified.get("change_type"),
            "confidence":     classified.get("confidence"),
            "narrative":      narrative,
            "bbox_json":      list(_scene_bbox) if _scene_bbox else None,
        })
    except Exception:
        pass  # catalogue write is best-effort, never block the UI
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
# NDVI Vegetation Health Analysis
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🌿 NDVI Vegetation Health Analysis")

# Determine whether the current inputs are GeoTIFFs we can run NDVI on.
_is_tif_before = before_name.lower().endswith((".tif", ".tiff"))
_is_tif_after  = after_name.lower().endswith((".tif", ".tiff"))

if not (_is_tif_before and _is_tif_after):
    st.info(
        "NDVI analysis requires **multi-band GeoTIFF** input (Sentinel-2 or Landsat). "
        "Upload `.tif` files with NIR and Red bands, or use **Fetch mode** "
        "with a Sentinel-2 dataset to enable this section.",
        icon="🌱",
    )
else:
    with st.expander("⚙️ Band configuration", expanded=False):
        band_col1, band_col2 = st.columns(2)
        with band_col1:
            nir_band_sel = st.number_input(
                "NIR band index (1-based)",
                min_value=1, max_value=20, value=8,
                help="Sentinel-2: band 8 (833 nm).  Landsat 8/9: band 5.",
            )
        with band_col2:
            red_band_sel = st.number_input(
                "Red band index (1-based)",
                min_value=1, max_value=20, value=4,
                help="Sentinel-2: band 4 (664 nm).  Landsat 8/9: band 4.",
            )

    ndvi_btn = st.button("🌿 Compute NDVI Difference", type="secondary")

    if ndvi_btn:
        @st.cache_data(show_spinner=False)
        def _run_ndvi(b_bytes: bytes, b_name: str,
                      a_bytes: bytes, a_name: str,
                      nir_b: int, red_b: int) -> dict:
            """Write uploads to temp files and run compute_ndvi_diff."""
            import tempfile as _tf
            b_suf = os.path.splitext(b_name)[1]
            a_suf = os.path.splitext(a_name)[1]
            bp = ap = None
            try:
                with _tf.NamedTemporaryFile(suffix=b_suf, delete=False) as f:
                    f.write(b_bytes); bp = f.name
                with _tf.NamedTemporaryFile(suffix=a_suf, delete=False) as f:
                    f.write(a_bytes); ap = f.name
                nb, na, diff, ndvi_stats = compute_ndvi_diff(
                    bp, ap, nir_band=nir_b, red_band=red_b
                )
                return {"ndvi_before": nb, "ndvi_after": na,
                        "ndvi_diff": diff, "ndvi_stats": ndvi_stats}
            except ValueError as exc:
                return {"error": str(exc)}
            except Exception as exc:
                return {"error": f"NDVI computation failed: {exc}"}
            finally:
                for p in (bp, ap):
                    if p and os.path.exists(p):
                        os.unlink(p)

        with st.spinner("Computing NDVI…"):
            ndvi_result = _run_ndvi(
                before_bytes, before_name,
                after_bytes,  after_name,
                int(nir_band_sel), int(red_band_sel),
            )

        if "error" in ndvi_result:
            st.error(f"⚠️ {ndvi_result['error']}", icon="🌿")
        else:
            import matplotlib.pyplot as _plt
            import matplotlib.colors as _mcolors
            import io as _io

            ndvi_stats = ndvi_result["ndvi_stats"]
            ndvi_diff  = ndvi_result["ndvi_diff"]

            # --- Metrics row ---
            nm1, nm2, nm3, nm4, nm5 = st.columns(5)
            nm1.metric("NDVI before",   f"{ndvi_stats['mean_ndvi_before']:.3f}")
            nm2.metric("NDVI after",    f"{ndvi_stats['mean_ndvi_after']:.3f}",
                       delta=f"{ndvi_stats['mean_ndvi_diff']:+.3f}")
            nm3.metric("Mean Δ NDVI",   f"{ndvi_stats['mean_ndvi_diff']:+.3f}")
            nm4.metric("Vegetation gain", f"{ndvi_stats['gain_area_pct']:.1f}%",
                       help="Pixels with NDVI increase > +0.05")
            nm5.metric("Vegetation loss", f"{ndvi_stats['loss_area_pct']:.1f}%",
                       delta=f"-{ndvi_stats['loss_area_pct']:.1f}%",
                       delta_color="inverse",
                       help="Pixels with NDVI decrease < −0.05")

            # --- Colour-mapped NDVI diff image ---
            # Diverging RdYlGn: red = vegetation loss, green = vegetation gain
            fig, ax = _plt.subplots(figsize=(8, 4))
            im = ax.imshow(
                ndvi_diff,
                cmap="RdYlGn",
                vmin=-0.5, vmax=0.5,
                interpolation="nearest",
            )
            _plt.colorbar(im, ax=ax, label="ΔNDVI (after − before)", fraction=0.03)
            ax.set_title("NDVI Change Map  (green = gain · red = loss)", fontsize=11)
            ax.axis("off")
            _plt.tight_layout()

            buf = _io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            _plt.close(fig)
            buf.seek(0)
            ndvi_png_bytes = buf.read()

            st.image(ndvi_png_bytes, use_container_width=True)
            st.caption(
                "Colour scale clamped to ±0.5 NDVI units.  "
                "Green pixels gained vegetation; red pixels lost it.  "
                "Threshold for gain/loss classification: |ΔNDVI| > 0.05."
            )

            # --- Download button for the NDVI diff PNG ---
            st.download_button(
                "⬇️ NDVI change map (PNG)",
                data=ndvi_png_bytes,
                file_name="ndvi_change_map.png",
                mime="image/png",
            )

# ---------------------------------------------------------------------------
# False-colour composites
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🎨 False-Colour Composites")

if not _is_tif_before:
    st.info(
        "False-colour composites require a **multi-band GeoTIFF** before-image "
        "(Sentinel-2 L2A or Landsat).  Upload a `.tif` file or use Fetch mode "
        "with the Sentinel-2 source and select additional bands.",
        icon="🎨",
    )
else:
    _fc_preset_name = st.selectbox(
        "Composite preset",
        list(FALSE_COLOUR_PRESETS.keys()),
        help="Choose a standard band combination optimised for a specific analysis goal.",
        key="fc_preset",
    )
    _fc_r, _fc_g, _fc_b, _fc_desc = FALSE_COLOUR_PRESETS[_fc_preset_name]

    with st.expander("⚙️ Custom band indices", expanded=False):
        _fc_col1, _fc_col2, _fc_col3 = st.columns(3)
        _fc_r = _fc_col1.number_input("Red channel band",   min_value=1, max_value=30, value=_fc_r, key="fc_r")
        _fc_g = _fc_col2.number_input("Green channel band", min_value=1, max_value=30, value=_fc_g, key="fc_g")
        _fc_b = _fc_col3.number_input("Blue channel band",  min_value=1, max_value=30, value=_fc_b, key="fc_b")

    st.caption(f"**{_fc_preset_name}** — {_fc_desc}")

    fc_btn = st.button("🎨 Render False-Colour Composites", type="secondary", key="fc_btn")

    if fc_btn:
        @st.cache_data(show_spinner=False)
        def _run_fc(b_bytes: bytes, b_name: str,
                    a_bytes: bytes, a_name: str,
                    r_b: int, g_b: int, bl_b: int) -> dict:
            """Write uploads to temp files and render false-colour composites."""
            import tempfile as _tf2
            b_suf = os.path.splitext(b_name)[1]
            a_suf = os.path.splitext(a_name)[1]
            bp = ap = None
            try:
                with _tf2.NamedTemporaryFile(suffix=b_suf, delete=False) as f:
                    f.write(b_bytes); bp = f.name
                if a_bytes:
                    with _tf2.NamedTemporaryFile(suffix=a_suf, delete=False) as f:
                        f.write(a_bytes); ap = f.name
                fc_before = compute_false_colour(bp, r_b, g_b, bl_b)
                fc_after  = compute_false_colour(ap, r_b, g_b, bl_b) if ap else None
                return {"fc_before": fc_before, "fc_after": fc_after}
            except (ValueError, FileNotFoundError) as exc:
                return {"error": str(exc)}
            except Exception as exc:
                return {"error": f"False-colour render failed: {exc}"}
            finally:
                for p in (bp, ap):
                    if p and os.path.exists(p):
                        os.unlink(p)

        with st.spinner("Rendering false-colour composites…"):
            _fc_result = _run_fc(
                before_bytes, before_name,
                after_bytes if _is_tif_after else b"",
                after_name,
                int(_fc_r), int(_fc_g), int(_fc_b),
            )

        if "error" in _fc_result:
            st.error(f"⚠️ {_fc_result['error']}", icon="🎨")
        else:
            _fc_cols = st.columns(2) if _fc_result["fc_after"] is not None else [st.container()]
            with _fc_cols[0]:
                st.caption(f"**Before** — {_fc_preset_name}")
                st.image(_fc_result["fc_before"], use_container_width=True)
                _fc_b_bytes = cv2.imencode(".png", cv2.cvtColor(_fc_result["fc_before"], cv2.COLOR_RGB2BGR))[1].tobytes()
                st.download_button(
                    "⬇️ Before composite (PNG)", data=_fc_b_bytes,
                    file_name="fc_before.png", mime="image/png",
                )
            if _fc_result["fc_after"] is not None:
                with _fc_cols[1]:
                    st.caption(f"**After** — {_fc_preset_name}")
                    st.image(_fc_result["fc_after"], use_container_width=True)
                    _fc_a_bytes = cv2.imencode(".png", cv2.cvtColor(_fc_result["fc_after"], cv2.COLOR_RGB2BGR))[1].tobytes()
                    st.download_button(
                        "⬇️ After composite (PNG)", data=_fc_a_bytes,
                        file_name="fc_after.png", mime="image/png",
                    )

# ---------------------------------------------------------------------------
# Scene Catalogue
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🗄️ Scene Catalogue")
st.caption(
    "Every analysis with AI narration is automatically saved here.  "
    "Filter, browse, and re-read past results."
)

try:
    _cstats = _cat_stats()
    _cc1, _cc2, _cc3, _cc4 = st.columns(4)
    _cc1.metric("Total analyses",     _cstats["total_analyses"])
    _cc2.metric("Avg change %",       f"{_cstats['avg_change_pct']:.1f}%" if _cstats["avg_change_pct"] is not None else "—")
    _cc3.metric("Change types seen",  len(_cstats["by_change_type"]))
    _cc4.metric("Last analysed",      (_cstats["last_analysed_at"] or "—")[:10])
except Exception:
    pass

with st.expander("Browse / filter catalogue", expanded=False):
    _f_col1, _f_col2, _f_col3 = st.columns(3)
    _cat_limit        = _f_col1.number_input("Max rows", min_value=5, max_value=200, value=20, key="cat_limit")
    _cat_type_filter  = _f_col2.text_input("Change type filter", placeholder="e.g. wildfire", key="cat_type")
    _cat_pct_filter   = _f_col3.number_input("Min change %", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="cat_pct")

    _cat_refresh = st.button("🔄 Refresh catalogue", key="cat_refresh")
    if _cat_refresh or True:   # always render on first load
        try:
            import pandas as _pd
            _entries = _cat_list(
                limit=int(_cat_limit),
                change_type=_cat_type_filter.strip() or None,
                min_change_pct=float(_cat_pct_filter) if _cat_pct_filter else None,
            )
            if _entries:
                _cat_df = _pd.DataFrame([{
                    "ID":           e["id"],
                    "Date":         (e.get("analysed_at") or "")[:10],
                    "Before":       e.get("before_name", ""),
                    "After":        e.get("after_name", ""),
                    "Mode":         e.get("mode", ""),
                    "Change %":     e.get("change_percent"),
                    "Regions":      e.get("num_regions"),
                    "km²":          e.get("changed_km2"),
                    "Type":         e.get("change_type", ""),
                    "Confidence":   e.get("confidence", ""),
                    "Observation":  e.get("date_range", ""),
                } for e in _entries])
                st.dataframe(_cat_df, use_container_width=True, hide_index=True)

                # Detail expander for individual entries
                _sel_id = st.number_input("Show full narrative for entry ID", min_value=0, value=0, step=1, key="cat_sel_id")
                if _sel_id > 0:
                    _sel_entry = next((e for e in _entries if e["id"] == _sel_id), None)
                    if _sel_entry and _sel_entry.get("narrative"):
                        st.info(_sel_entry["narrative"], icon="🛰️")
                    elif _sel_entry:
                        st.caption("No narrative saved for this entry.")
                    else:
                        st.caption(f"Entry {_sel_id} not found in the current filtered view.")
            else:
                st.caption("No catalogue entries yet — run an analysis and generate AI narration to populate it.")
        except Exception as _exc:
            st.caption(f"Catalogue unavailable: {_exc}")

# ---------------------------------------------------------------------------
# Scheduled Monitoring (AOI Watcher)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📡 Scheduled Monitoring")
st.caption(
    "Register this AOI with the scheduler so TerraLens automatically checks for "
    "new satellite acquisitions and runs change detection + narration.  "
    "Then run `python scheduler.py` (or `--daemon`) to start monitoring."
)

with st.expander("➕ Watch this AOI", expanded=False):
    _sch_col1, _sch_col2 = st.columns(2)
    _aoi_name   = _sch_col1.text_input("AOI name", value=before_name.split(".")[0], key="aoi_name")
    _aoi_source = _sch_col2.selectbox(
        "Imagery source",
        ["modis", "landsat", "sentinel2"],
        index=2 if mode == "🌍 Fetch from Copernicus (Sentinel-2)" else 0,
        key="aoi_source",
    )
    _aoi_cloud     = st.slider("Max cloud % (Sentinel-2)", 0, 100, 30, key="aoi_cloud")
    _aoi_threshold = st.slider("Change threshold", 10, 100, threshold, key="aoi_threshold")
    _aoi_notes     = st.text_area("Notes (optional)", key="aoi_notes", height=60)

    _watch_btn = st.button("📡 Register AOI for monitoring", key="watch_btn")
    if _watch_btn:
        try:
            from scheduler import add_watched_aoi as _add_aoi
            _aoi_bbox = _scene_bbox or (-10.0, 35.0, 10.0, 55.0)
            _new_id = _add_aoi(
                name=_aoi_name or "Unnamed AOI",
                bbox=_aoi_bbox,
                source=_aoi_source,
                max_cloud_pct=float(_aoi_cloud),
                threshold=int(_aoi_threshold),
                notes=_aoi_notes,
            )
            st.success(
                f"✅ AOI registered (id={_new_id}).  "
                "Run `python scheduler.py` to trigger the first poll, "
                "or `python scheduler.py --daemon --interval 360` for continuous monitoring.",
                icon="📡",
            )
        except Exception as _exc:
            st.error(f"Failed to register AOI: {_exc}")

with st.expander("📋 Monitored AOIs", expanded=False):
    try:
        from scheduler import list_watched_aois as _list_aois
        import pandas as _pd2
        _aois = _list_aois()
        if _aois:
            _aoi_df = _pd2.DataFrame([{
                "ID":           a["id"],
                "Name":         a["name"],
                "Source":       a["source"],
                "Enabled":      "✅" if a["enabled"] else "⏸",
                "Last checked": (a.get("last_checked_at") or "never")[:19],
                "Last scene":   a.get("last_scene_date") or "—",
                "Polls":        a.get("check_count", 0),
            } for a in _aois])
            st.dataframe(_aoi_df, use_container_width=True, hide_index=True)
            st.caption(
                "Manage AOIs from the CLI:  \n"
                "`python scheduler.py --list-aois`  \n"
                "`python scheduler.py --disable-aoi <ID>`  \n"
                "`python scheduler.py --delete-aoi <ID>`"
            )
        else:
            st.caption("No AOIs registered yet.  Use the form above to add one.")
    except Exception as _exc:
        st.caption(f"Scheduler unavailable: {_exc}")

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
