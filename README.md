# TerraLens 🛰️

> **IBM AI Builders August Challenge — Space Exploration Theme**

Detect meaningful change between two satellite images of the same location
taken at different dates, then use **IBM Granite** to narrate what happened on
the ground — in plain language any non-specialist can act on.

---

## Problem Statement

Satellite imagery is collected continuously — Sentinel-2 revisits any point on
Earth every five days, Landsat every sixteen. Yet turning those raw images into
actionable intelligence still requires expert analysts. Automated change
detection exists, but its output (a binary pixel mask and a change percentage)
is opaque to decision-makers, disaster responders, and conservation teams who
need to know *what* changed, *why* it matters, and *how confident* we should be.

TerraLens closes that gap: it pairs a classical computer-vision pipeline with an
IBM Granite language model so that the same workflow that produces a pixel mask
also produces a human-readable briefing.

---

## Solution Description

Given a **before** image and an **after** image of the same geographic area:

1. **Alignment** — ORB feature matching + RANSAC homography corrects for
   small viewpoint differences between satellite passes.
2. **Change detection** — grayscale absolute differencing, Gaussian blur,
   binary thresholding, morphological opening, and contour extraction isolate
   meaningful changed regions and filter sensor noise.
3. **Narration** — quantitative stats (fraction changed, region count, largest
   region area) are passed as a structured prompt to IBM Granite, which
   classifies the probable change type and produces a confidence-annotated
   plain-language summary.
4. **Dashboard** — a Streamlit app ties everything together: upload, analyse,
   compare, and download results in a single browser session.

---

## AI Approach & Architecture

### Pipeline diagram

```
  ┌──────────────┐     ┌──────────────┐
  │ before.tif   │     │  after.tif   │
  └──────┬───────┘     └──────┬───────┘
         │                    │
         ▼                    ▼
  ┌──────────────────────────────────────┐
  │   load_and_align()                   │
  │   • rasterio / OpenCV image load     │
  │   • resize "after" → "before" dims   │
  │   • ORB keypoints + BF Hamming match │
  │   • Lowe ratio test (0.75)           │
  │   • RANSAC homography + warpPerspect │
  └──────────────────┬───────────────────┘
                     │ (before, after_aligned)
                     ▼
  ┌──────────────────────────────────────┐
  │   detect_change()                    │
  │   • grayscale absdiff                │
  │   • Gaussian blur (5×5)              │
  │   • binary threshold                 │
  │   • morphological opening            │
  │   • contour filter (min area)        │
  └──────────┬──────────────┬────────────┘
             │              │
             ▼              ▼
       change mask        stats dict
       (binary PNG)   changed_fraction
                      num_regions
                      top-10 regions
                      largest area px²
             │              │
             ▼              ▼
  ┌──────────────┐   ┌──────────────────────────────────┐
  │  overlay()   │   │  narrate_with_granite()           │
  │  40 % red    │   │  • build structured prompt        │
  │  blend PNG   │   │  • POST → watsonx.ai              │
  └──────────────┘   │    model: granite-3-8b-instruct   │
                     │    max_new_tokens: 300             │
                     │    temperature: 0.2                │
                     │  • classify change type            │
                     │  • state confidence + caveat       │
                     │  • template fallback if offline    │
                     └──────────────────────────────────┘
                                    │
                                    ▼
                          Streamlit dashboard
                         (app.py — browser UI)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| ORB + RANSAC over ECC | ORB is robust to non-uniform brightness shifts across satellite passes; ECC assumes photometric consistency |
| Gaussian blur before threshold | Suppresses JPEG/sensor noise without morphological closing artefacts |
| Prompt encodes largest-region area | Single-band pixel count is ambiguous; spatial concentration changes the likely cause classification |
| `temperature=0.2` | Low temperature produces consistent, factual classifications over creative paraphrasing |
| Template fallback | Keeps the app fully functional at demo time without live credentials |
| `@st.cache_data` on CV pipeline | Prevents expensive re-computation on every Streamlit re-render |

---

## Selected Challenge Theme

**Space Exploration** — TerraLens uses Earth-observation satellite data
(Sentinel-2, Landsat) to monitor surface change: deforestation, wildfire
extent, flood inundation, glacier retreat, and urban growth. These are exactly
the problems that remote-sensing satellites were built to track, and where
AI-powered interpretation can accelerate human response.

---

## How IBM Bob Was Used

IBM Bob (the AI coding assistant) was used throughout the entire development
lifecycle of TerraLens:

- **Architecture design** — Bob proposed the ORB → RANSAC → absdiff → Granite
  pipeline and advised on the tradeoffs between ECC and feature-matching
  alignment approaches.
- **Code generation** — all three Python modules (`change_detection.py`,
  `narrate.py`, `app.py`) were written iteratively through Bob, with each
  module generated from a precise natural-language specification.
- **Prompt engineering** — Bob designed the structured three-part Granite prompt
  (narrative / change-type classification / confidence + caveat) and advised on
  setting `temperature=0.2` for consistent factual output.
- **Debugging & refinement** — Bob identified that `st.cache_data` was needed
  to avoid re-running the CV pipeline on every Streamlit widget interaction,
  and corrected the BGR→RGB conversion path for the three-column image view.
- **Documentation** — this README was drafted by Bob to satisfy the IBM AI
  Builders challenge judge requirements.

---

## Data Sources

| Source | Dataset | Access |
|--------|---------|--------|
| [Copernicus Open Access Hub](https://scihub.copernicus.eu/) | **Sentinel-2** L2A (10 m resolution, 13 spectral bands, 5-day revisit) | Free registration required |
| [NASA Worldview / GIBS](https://worldview.earthdata.nasa.gov/) | **Landsat 8/9** (30 m, 16-day revisit), MODIS true-colour | Public, no registration |
| [USGS Earth Explorer](https://earthexplorer.usgs.gov/) | Landsat Collection 2 Level-2 products | Free registration required |

TerraLens accepts any co-registered before/after pair as PNG, JPEG, or GeoTIFF.
The ORB alignment step compensates for small registration errors between
different scene acquisitions, but images should cover the same geographic
bounding box.

---

## Getting Started

### Prerequisites

- Python 3.11+
- A watsonx.ai account with a project ID (optional — app runs offline without it)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kellermanj2-eng/TerraLens.git
cd terralens

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Set watsonx.ai credentials

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

If these variables are not set, TerraLens runs fully offline with a
template-based narrative fallback.

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a before/after image pair, adjust the
threshold slider, and click **✨ Generate AI Narration**.

### Run the CLI

```bash
python change_detection.py \
  --before  data/before.tif \
  --after   data/after.tif  \
  --out     results/overlay.png \
  --threshold 40
```

Stats are printed to stdout as formatted JSON:

```json
{
  "changed_fraction": 0.073,
  "change_percent": 7.3,
  "changed_pixels": 48203,
  "total_pixels": 660000,
  "num_regions": 12,
  "top_regions": [
    { "x": 142, "y": 88, "w": 310, "h": 204, "area_px": 47821 },
    ...
  ]
}
```

---

## Roadmap

### Phase 1 — Foundation ✅ Complete

| Status | Feature |
|--------|---------|
| ✅ Done | Streamlit dashboard — upload, align, detect, overlay, narrate, download |
| ✅ Done | ORB + RANSAC alignment pipeline (`change_detection.py`) |
| ✅ Done | IBM Granite narration with template fallback (`narrate.py`) |
| ✅ Done | No-credentials offline mode (template narrative fallback) |
| ✅ Done | CLI interface (`python change_detection.py --before … --after …`) |

---

### Phase 2 — Live Satellite Image Acquisition ✅ Complete

The goal of this phase is to eliminate the manual upload step entirely.
Users will enter a location and date range; TerraLens fetches the imagery automatically.

| Status | Feature | Notes |
|--------|---------|-------|
| ✅ Done | **NASA Worldview / GIBS auto-fetch** — `satellite_fetch.py` queries NASA CMR + stitches GIBS WMTS tiles; **no account required** | MODIS Terra (daily) and Landsat annual composites via [NASA GIBS](https://worldview.earthdata.nasa.gov/) |
| ✅ Done | **Map-based AOI picker** — Leaflet.js draw widget in Fetch mode; coordinate bbox fallback; session-state persisted across reruns | Powered by `streamlit-folium` + `folium.plugins.Draw` |
| ✅ Done | **MCP satellite tool server** — `mcp_server.py` exposes `search_satellite_scenes`, `fetch_scene_pair`, `run_change_detection`, `narrate_change` as MCP tools | Register with: `bob mcp add --name terralens --command "python mcp_server.py"` |
| ✅ Done | **Copernicus / Sentinel-2 integration** — `sentinel2_fetch.py` searches CDSE OData catalogue (no auth) and downloads 10 m L2A band stacks (CDSE_USER/CDSE_PASSWORD); third mode in app.py with AOI map, cloud-cover filter, band selector, and credential badge; `search_sentinel2_scenes` + `fetch_sentinel2_pair` MCP tools added | Free registration at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) required |
| ✅ Done | **Automated scene scheduling** — `scheduler.py`; `watched_aois` SQLite table; `poll_once()` / `run_daemon()` loop; downloads pair, runs change detection + narration, writes plain-text report, saves to catalogue; CLI (`--add-aoi`, `--daemon`, `--list-aois`); `schedule_aoi` + `list_watched_aois` MCP tools; "Watch this AOI" UI in app | Sentinel-2 5-day revisit cadence |
| ✅ Done | **Cloud-mask filtering** — `include_scl=True` wired through `fetch_sentinel2_pair` → `download_sentinel2_scene`; SCL paths stored in session state; union cloud mask fed to `detect_change(cloud_mask=…)`; "Cloud masked %" metric in stats row; ☁️ toggle in Sentinel-2 sidebar | Reduces false-positive change detections in scenes with partial cloud cover |

---

### Phase 3 — Analysis Depth

| Status | Feature |
|--------|---------|
| ✅ Done | Multi-temporal analysis — date-series input → consecutive-pair change % → `st.line_chart` trend |
| ✅ Done | GeoJSON export of changed-region bounding boxes (`regions_to_geojson()` in `change_detection.py`) |
| ✅ Done | Change area in real-world km² — computed from bbox lon/lat span, shown in metrics row |
| ✅ Done | NDVI differencing for vegetation health — `compute_ndvi_diff()` in `change_detection.py`; diverging RdYlGn change map + gain/loss metrics in `app.py`; configurable NIR/Red band indices (Sentinel-2 defaults: band 8/4; Landsat 8: band 5/4) |

---

### Phase 4 — AI & Visualisation ✅ Complete

| Status | Feature |
|--------|---------|
| ✅ Done | **Few-shot Granite classifier** — `narrate.py` upgraded with three labelled examples that anchor the taxonomy, structured `Narrative / Change type / Confidence` output parsed into a classified dict; classifier badge shown in app UI; `agricultural change` label added |
| ✅ Done | **Side-by-side swipe viewer** — drag-divider before/after comparison embedded in the Streamlit dashboard (`app.py`) |
| ✅ Done | **Scene catalogue** — `catalogue.py` persists every narrated analysis to a local SQLite database; `list_catalogue` and `get_catalogue_entry` MCP tools; searchable dataframe in app UI; Watsonx.data JDBC mirror hook via `WATSONX_DATA_CONNECTION_URL` env var |
| ✅ Done | **Multi-spectral false-colour composites** — `compute_false_colour()` in `change_detection.py` with 5 presets (CIR, Urban/SWIR, Agriculture, Geology, Bathymetric); side-by-side before/after composite view + download in app |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built for the IBM AI Builders August Challenge — Space Exploration theme.*  
*Satellite imagery courtesy of the European Space Agency (Copernicus) and NASA.*
