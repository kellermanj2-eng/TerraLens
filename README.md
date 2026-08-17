# TerraLens 🛰️

> **IBM AI Builders August Challenge — Space Exploration Theme**

Detect meaningful change between two satellite images of the same location
taken at different dates, then use **IBM Granite** to narrate what happened on
the ground — in plain language any non-specialist can act on.

[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Docker](https://img.shields.io/badge/docker-one--command-blue)](#docker-one-command)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Quick Start

### Option A — Docker (one command, no Python required)

```bash
git clone https://github.com/kellermanj2-eng/TerraLens.git
cd TerraLens/terralens
docker compose up
```

Open **http://localhost:8501** — the app runs fully offline with template
narration. Add watsonx.ai / Copernicus credentials to a `.env` file to unlock
live Granite narration and Sentinel-2 downloads (see [`.env.example`](.env.example)).

### Option B — Python

```bash
git clone https://github.com/kellermanj2-eng/TerraLens.git
cd TerraLens/terralens
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

### Try it instantly — bundled sample images

The repo includes a synthetic before/after pair in `data/` that works
out-of-the-box without any credentials or downloads:

1. In the app, select **📂 Upload images**
2. Upload `data/sample_before.png` as **Before**
3. Upload `data/sample_after.png` as **After**
4. Click **✨ Generate AI Narration**

Expected result: ~14 % scene change with a large burn-scar region detected.

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
   region area) are passed as a few-shot prompt to IBM Granite, which
   classifies the probable change type and produces a confidence-annotated
   plain-language summary.
4. **Dashboard** — a Streamlit app ties everything together: upload or
   auto-fetch, analyse, swipe-compare, NDVI, false-colour, catalogue,
   and download results in a single browser session.

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
  │   • SCL cloud-mask suppression       │
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
                      cloud_masked_pct
             │              │
             ▼              ▼
  ┌──────────────┐   ┌──────────────────────────────────┐
  │  overlay()   │   │  narrate_with_granite()           │
  │  40 % red    │   │  • few-shot prompt (3 examples)   │
  │  blend PNG   │   │  • POST → watsonx.ai              │
  └──────────────┘   │    model: granite-3-8b-instruct   │
                     │    max_new_tokens: 400             │
                     │    temperature: 0.2                │
                     │  • classify change type (9 labels) │
                     │  • state confidence + caveat       │
                     │  • template fallback if offline    │
                     └──────────────────────────────────┘
                                    │
                                    ▼
                          Streamlit dashboard
                         (app.py — browser UI)
                                    │
                                    ▼
                          catalogue.py (SQLite)
                         ← Watsonx.data mirror →
```

### Key design decisions

| Decision | Rationale |
|---|---|
| ORB + RANSAC over ECC | ORB is robust to non-uniform brightness shifts across satellite passes; ECC assumes photometric consistency |
| Gaussian blur before threshold | Suppresses JPEG/sensor noise without morphological closing artefacts |
| Few-shot prompt with 3 anchor examples | Dramatically improves label consistency vs zero-shot; examples span wildfire, urban growth, and noise magnitude |
| Prompt encodes largest-region area | Single-band pixel count is ambiguous; spatial concentration changes the likely cause classification |
| `temperature=0.2` | Low temperature produces consistent, factual classifications over creative paraphrasing |
| Template fallback | Keeps the app fully functional at demo time without live credentials |
| `@st.cache_data` on CV pipeline | Prevents expensive re-computation on every Streamlit re-render |
| SCL cloud-mask union | Union of before+after SCL masks suppresses cloud/shadow pixels in *either* scene before thresholding |

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
- **Code generation** — all Python modules were written iteratively through Bob,
  with each module generated from a precise natural-language specification.
- **Prompt engineering** — Bob designed the few-shot Granite prompt
  (three labelled anchor examples + `Narrative / Change type / Confidence`
  structured output) and advised on setting `temperature=0.2` for consistent
  factual output.
- **Debugging & refinement** — Bob identified that `st.cache_data` was needed
  to avoid re-running the CV pipeline on every Streamlit widget interaction,
  and corrected the BGR→RGB conversion path for the three-column image view.
- **Feature expansion** — Bob implemented the complete Phase 2–4 roadmap:
  NASA GIBS auto-fetch, Copernicus Sentinel-2 integration, SCL cloud-mask
  filtering, automated scene scheduler, scene catalogue, false-colour
  composites, swipe viewer, few-shot classifier, MCP tool server, and
  quick-select preset landmark events.
- **Documentation & tests** — this README and the 66-test pytest suite were
  drafted by Bob to satisfy the IBM AI Builders challenge judge requirements.

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

- Python 3.11+  **or** Docker (for the one-command option)
- A watsonx.ai account with a project ID (optional — app runs offline without it)

### Docker (one command) {#docker-one-command}

```bash
# Build and start (first run takes ~3 min to pull base image + install deps)
docker compose up

# Background mode
docker compose up -d

# Stop
docker compose down
```

Open http://localhost:8501.

### Python installation

```bash
# 1. Clone
git clone https://github.com/kellermanj2-eng/TerraLens.git
cd TerraLens/terralens

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Set credentials (all optional)

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

CDSE_USER=your_copernicus_email@example.com
CDSE_PASSWORD=your_copernicus_password
```

If these variables are not set, TerraLens runs fully offline with a
template-based narrative fallback and NASA GIBS imagery (no account required).

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501.

### Run the CLI

```bash
python change_detection.py \
  --before  data/sample_before.png \
  --after   data/sample_after.png  \
  --out     results/overlay.png \
  --threshold 40
```

### Run the automated scheduler

```bash
# Register an AOI to monitor
python scheduler.py --add-aoi \
    --name "Amazon Watch" --source sentinel2 \
    --bbox "-63.5,-11.0,-62.0,-9.5" --max-cloud 30

# Poll all AOIs once (cron-friendly)
python scheduler.py

# Run as a daemon (polls every 6 hours)
python scheduler.py --daemon --interval 360
```

---

## Test Suite

```bash
pytest tests/ -v
# 66 tests, ~6 seconds, fully offline
```

Coverage:
- `tests/test_change_detection.py` — CV pipeline: detect_change, overlay, GeoJSON, NDVI, false-colour
- `tests/test_catalogue.py` — SQLite catalogue: CRUD, filtering, stats
- `tests/test_narrate.py` — prompt construction, template fallback, classifier output parsing
- `tests/test_scheduler.py` — AOI management: add, list, enable/disable, delete, poll lifecycle

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

| Status | Feature | Notes |
|--------|---------|-------|
| ✅ Done | **NASA Worldview / GIBS auto-fetch** | MODIS Terra (daily) and Landsat annual composites via [NASA GIBS](https://worldview.earthdata.nasa.gov/) — no account required |
| ✅ Done | **Map-based AOI picker** | Leaflet.js draw widget; coordinate bbox fallback; session-state persisted |
| ✅ Done | **MCP satellite tool server** | `mcp_server.py` — register with `bob mcp add --name terralens --command "python mcp_server.py"` |
| ✅ Done | **Copernicus / Sentinel-2 integration** | `sentinel2_fetch.py`; 10 m L2A; CDSE_USER/CDSE_PASSWORD; `search_sentinel2_scenes` + `fetch_sentinel2_pair` MCP tools |
| ✅ Done | **Automated scene scheduling** | `scheduler.py`; `watched_aois` SQLite table; `poll_once()` / `run_daemon()`; `schedule_aoi` + `list_watched_aois` MCP tools |
| ✅ Done | **Cloud-mask filtering** | SCL sidecar wired through pipeline; union mask fed to `detect_change(cloud_mask=…)`; ☁️ toggle in sidebar |

---

### Phase 3 — Analysis Depth ✅ Complete

| Status | Feature |
|--------|---------|
| ✅ Done | Multi-temporal analysis — date-series → consecutive-pair change % → `st.line_chart` trend |
| ✅ Done | GeoJSON export of changed-region bounding boxes |
| ✅ Done | Change area in real-world km² |
| ✅ Done | NDVI differencing — diverging RdYlGn change map + gain/loss metrics |

---

### Phase 4 — AI & Visualisation ✅ Complete

| Status | Feature |
|--------|---------|
| ✅ Done | **Few-shot Granite classifier** — 3 anchor examples; structured Narrative/Change type/Confidence output; 9-label taxonomy |
| ✅ Done | **Side-by-side swipe viewer** — drag-divider before/after comparison |
| ✅ Done | **Scene catalogue** — SQLite persistence; Watsonx.data JDBC mirror hook; browse/filter UI |
| ✅ Done | **Multi-spectral false-colour composites** — 5 presets (CIR, Urban/SWIR, Agriculture, Geology, Bathymetric) |

---

### Submission Extras ✅ Complete

| Status | Item |
|--------|------|
| ✅ Done | **Docker** — `Dockerfile` + `docker-compose.yml`; one-command startup |
| ✅ Done | **Sample images** — `data/sample_before.png` + `data/sample_after.png` bundled in repo |
| ✅ Done | **Test suite** — 66 pytest tests, fully offline, < 10 s runtime |
| ✅ Done | **Quick-select preset events** — 16 landmark events pre-loaded (8 NASA, 8 Sentinel-2) |
| ✅ Done | **MCP tools** — 10 tools total: search, fetch, analyse, narrate, catalogue, scheduler |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built for the IBM AI Builders August Challenge — Space Exploration theme.*
*Satellite imagery courtesy of the European Space Agency (Copernicus) and NASA.*
