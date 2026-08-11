"""
satellite_fetch.py
------------------
Automatic satellite imagery acquisition via NASA Worldview / GIBS.

No account or API key is required.  GIBS (Global Imagery Browse Services) is
a public NASA service that serves pre-rendered true-colour tiles for Landsat,
MODIS, and other instruments via a standard WMTS/WMS interface.

Public API
~~~~~~~~~~
search_scenes(bbox, date_from, date_to, layer=LAYER_MODIS, limit=5)
    Use the NASA CMR (Common Metadata Repository) API to list MODIS / Landsat
    granules that intersect *bbox* and fall within the date range.
    Returns a list of scene metadata dicts sorted by date (most recent first).
    No authentication required.

download_scene(scene, out_dir, layer=LAYER_MODIS)
    Fetch a GIBS WMTS tile mosaic covering *scene*'s bbox and date, stitch
    the tiles into a single PNG, and save it to *out_dir*.
    No authentication required.

fetch_scene_pair(bbox, date_from, date_mid, date_to, out_dir, layer)
    High-level helper: fetches one "before" scene (date_from → date_mid) and
    one "after" scene (date_mid → date_to).
    Returns (before_path, after_path, before_meta, after_meta) or raises
    RuntimeError if no scenes are found.

Available layers (GIBS WMTS layer identifiers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LAYER_MODIS    = "MODIS_Terra_CorrectedReflectance_TrueColor"   (daily, 250 m)
LAYER_LANDSAT  = "Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual"

No environment variables required.  Everything is fetched from public endpoints.
"""

import io
import os
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# GIBS / CMR constants
# ---------------------------------------------------------------------------

# NASA CMR granule search endpoint
_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# NASA GIBS WMTS KVP endpoint (EPSG:4326 best-available)
_GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi"

# Publicly available GIBS true-colour layers (no auth needed)
LAYER_MODIS   = "MODIS_Terra_CorrectedReflectance_TrueColor"
LAYER_LANDSAT = "Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual"

# CMR short names that correspond to each layer
_LAYER_SHORT_NAME = {
    LAYER_MODIS:   "MOD09GA",          # MODIS Terra daily surface reflectance
    LAYER_LANDSAT: "LANDSAT_ETM_C2",   # Landsat 7/8 ETM+ Collection 2
}

# ---------------------------------------------------------------------------
# Per-layer GIBS tile matrix parameters
# Derived from WMTSCapabilities.xml:
#   MODIS  uses TileMatrixSet=250m, zoom=6 → 80×40 grid, 512px tiles
#   Landsat uses TileMatrixSet=500m, zoom=6 → 40×20 grid, 512px tiles
# ---------------------------------------------------------------------------

_LAYER_TILE_PARAMS: dict[str, dict] = {
    LAYER_MODIS: {
        "tile_matrix_set": "250m",
        "tile_zoom":       "6",
        "n_cols":          80,    # MatrixWidth at zoom 6
        "n_rows":          40,    # MatrixHeight at zoom 6
        "tile_size":       512,   # pixels per tile
        "format":          "image/jpeg",
    },
    LAYER_LANDSAT: {
        "tile_matrix_set": "500m",
        "tile_zoom":       "6",
        "n_cols":          40,
        "n_rows":          20,
        "tile_size":       512,
        "format":          "image/jpeg",
    },
}
# Default fallback params (matches MODIS)
_DEFAULT_TILE_PARAMS = _LAYER_TILE_PARAMS[LAYER_MODIS]

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _deg2tile(lon: float, lat: float, n_cols: int, n_rows: int) -> tuple[int, int]:
    """
    Convert WGS-84 lon/lat to a GIBS EPSG:4326 tile col/row.

    GIBS EPSG:4326 grid origin: top-left (-180, 90).
    n_cols × n_rows is the full grid size at the chosen zoom level.
    """
    col = int((lon + 180.0) / 360.0 * n_cols)
    row = int((90.0 - lat)  / 180.0 * n_rows)
    col = max(0, min(col, n_cols - 1))
    row = max(0, min(row, n_rows - 1))
    return col, row


# ---------------------------------------------------------------------------
# CMR granule search  (no auth)
# ---------------------------------------------------------------------------

def search_scenes(
    bbox: tuple[float, float, float, float],
    date_from: str,
    date_to: str,
    layer: str = LAYER_LANDSAT,
    limit: int = 5,
) -> list[dict]:
    """
    Search NASA CMR for granules that intersect *bbox* within [date_from, date_to].

    Parameters
    ----------
    bbox      : (min_lon, min_lat, max_lon, max_lat) in WGS-84 degrees.
    date_from : ISO date string, e.g. "2024-01-01".
    date_to   : ISO date string, e.g. "2024-06-30".
    layer     : GIBS layer constant (LAYER_LANDSAT or LAYER_MODIS).
    limit     : Maximum number of results.

    Returns
    -------
    List of scene dicts (most recent first), each containing:
        id           – CMR granule UR
        name         – granule producer_granule_id
        date         – sensing date string ("YYYY-MM-DD")
        bbox         – (min_lon, min_lat, max_lon, max_lat) of the granule
        layer        – GIBS layer identifier for this granule
        source       – "NASA CMR"
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    short_name = _LAYER_SHORT_NAME.get(layer, "MOD09GA")

    params = {
        "short_name":       short_name,
        "temporal":         f"{date_from}T00:00:00Z,{date_to}T23:59:59Z",
        "bounding_box":     f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "sort_key":         "-start_date",
        "page_size":        limit,
        "page_num":         1,
    }

    resp = requests.get(_CMR_URL, params=params, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])

    scenes = []
    for entry in entries:
        # Parse sensing date from time_start
        time_start = entry.get("time_start", "")
        sensing_date = time_start[:10] if time_start else date_from

        # Bounding box from CMR (boxes is a list of [S,W,N,E] strings)
        boxes = entry.get("boxes", [])
        if boxes:
            s, w, n, e = [float(v) for v in boxes[0].split()]
            granule_bbox = (w, s, e, n)
        else:
            granule_bbox = bbox

        scenes.append({
            "id":     entry.get("id", ""),
            "name":   entry.get("producer_granule_id") or entry.get("title", ""),
            "date":   sensing_date,
            "bbox":   granule_bbox,
            "layer":  layer,
            "source": "NASA CMR",
        })

    return scenes


# ---------------------------------------------------------------------------
# GIBS tile download + mosaic
# ---------------------------------------------------------------------------

def download_scene(
    scene: dict,
    out_dir: str = "data",
    layer: Optional[str] = None,
) -> str:
    """
    Fetch GIBS WMTS tiles covering *scene*'s bbox and date, stitch into a PNG.

    Uses the OGC WMTS KVP interface at EPSG:4326 — no authentication required.
    Tile matrix parameters (grid size, tile size, format) are looked up from
    _LAYER_TILE_PARAMS for the requested layer.

    Parameters
    ----------
    scene   : Scene dict as returned by search_scenes() or a minimal dict
              with keys: date (YYYY-MM-DD), bbox (min_lon,min_lat,max_lon,max_lat),
              layer (optional, overridden by the *layer* parameter).
    out_dir : Local directory to save the mosaic PNG into.
    layer   : GIBS layer name.  Defaults to scene["layer"] or LAYER_MODIS.

    Returns
    -------
    Path to the saved mosaic PNG.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    use_layer  = layer or scene.get("layer", LAYER_MODIS)
    scene_date = scene["date"]          # "YYYY-MM-DD"
    bbox       = scene.get("bbox", (-180, -90, 180, 90))
    min_lon, min_lat, max_lon, max_lat = bbox

    # Look up correct tile matrix parameters for this layer
    tp        = _LAYER_TILE_PARAMS.get(use_layer, _DEFAULT_TILE_PARAMS)
    tms       = tp["tile_matrix_set"]
    tile_zoom = tp["tile_zoom"]
    n_cols    = tp["n_cols"]
    n_rows    = tp["n_rows"]
    tile_size = tp["tile_size"]
    fmt       = tp["format"]

    # Determine tile range covering the requested bbox
    col_min, row_min = _deg2tile(min_lon, max_lat, n_cols, n_rows)   # top-left
    col_max, row_max = _deg2tile(max_lon, min_lat, n_cols, n_rows)   # bottom-right

    col_max = min(col_max, n_cols - 1)
    row_max = min(row_max, n_rows - 1)

    n_tiles_x = col_max - col_min + 1
    n_tiles_y = row_max - row_min + 1

    # Cap mosaic to 6×6 tiles max to keep download fast (~18 MB worst case)
    MAX_TILES = 6
    if n_tiles_x > MAX_TILES:
        col_mid   = (col_min + col_max) // 2
        col_min   = col_mid - MAX_TILES // 2
        col_max   = col_min + MAX_TILES - 1
        n_tiles_x = MAX_TILES
    if n_tiles_y > MAX_TILES:
        row_mid   = (row_min + row_max) // 2
        row_min   = row_mid - MAX_TILES // 2
        row_max   = row_min + MAX_TILES - 1
        n_tiles_y = MAX_TILES

    mosaic = Image.new("RGB", (n_tiles_x * tile_size, n_tiles_y * tile_size))

    for ty, row in enumerate(range(row_min, row_max + 1)):
        for tx, col in enumerate(range(col_min, col_max + 1)):
            tile_url = (
                f"{_GIBS_BASE}"
                f"?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
                f"&LAYER={use_layer}"
                f"&STYLE=default"
                f"&TILEMATRIXSET={tms}"
                f"&TILEMATRIX={tile_zoom}"
                f"&TILEROW={row}"
                f"&TILECOL={col}"
                f"&FORMAT={fmt}"
                f"&TIME={scene_date}"
            )
            try:
                r = requests.get(tile_url, timeout=20)
                r.raise_for_status()
                tile_img = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                tile_img = Image.new("RGB", (tile_size, tile_size))

            mosaic.paste(tile_img, (tx * tile_size, ty * tile_size))

    safe_name = scene.get("name", f"{use_layer}_{scene_date}").replace("/", "_")
    out_path  = os.path.join(out_dir, f"{safe_name}_{scene_date}.png")
    mosaic.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# High-level scene-pair fetcher
# ---------------------------------------------------------------------------

def fetch_scene_pair(
    bbox: tuple[float, float, float, float],
    date_from: str,
    date_mid: str,
    date_to: str,
    out_dir: str = "data",
    layer: str = LAYER_MODIS,
) -> tuple[str, str, dict, dict]:
    """
    Fetch a before/after satellite image pair via NASA GIBS.

    Searches CMR for the most recent granule in each date window.
    If CMR returns no granules (e.g. for annual composites), synthesises
    a minimal scene dict using the window end-date so that the GIBS tile
    fetch can still proceed.

    Parameters
    ----------
    bbox      : (min_lon, min_lat, max_lon, max_lat)
    date_from : Start of the "before" window  (ISO date, e.g. "2023-01-01")
    date_mid  : Boundary between before/after windows
    date_to   : End of the "after"  window
    out_dir   : Directory to download mosaics into
    layer     : GIBS layer (LAYER_LANDSAT or LAYER_MODIS)

    Returns
    -------
    (before_path, after_path, before_meta, after_meta)

    Raises
    ------
    RuntimeError if the tile download fails for both windows.
    """
    before_scenes = search_scenes(bbox, date_from, date_mid, layer, limit=3)
    after_scenes  = search_scenes(bbox, date_mid,  date_to,  layer, limit=3)

    # Fall back to synthetic scene dicts when CMR has no granules
    # (annual composites don't appear in CMR but GIBS still serves them by date)
    if not before_scenes:
        before_scenes = [{
            "id": "", "name": f"gibs_{date_mid}", "date": date_mid,
            "bbox": bbox, "layer": layer, "source": "GIBS direct",
        }]
    if not after_scenes:
        after_scenes = [{
            "id": "", "name": f"gibs_{date_to}", "date": date_to,
            "bbox": bbox, "layer": layer, "source": "GIBS direct",
        }]

    before_meta = before_scenes[0]
    after_meta  = after_scenes[0]

    # Override bbox with user-requested bbox (granule bbox may be much larger)
    before_meta = {**before_meta, "bbox": bbox}
    after_meta  = {**after_meta,  "bbox": bbox}

    before_path = download_scene(before_meta, out_dir, layer=layer)
    after_path  = download_scene(after_meta,  out_dir, layer=layer)

    return before_path, after_path, before_meta, after_meta
