"""
mcp_server.py
-------------
TerraLens MCP (Model Context Protocol) server.

Exposes TerraLens capabilities as MCP tools so any MCP-aware client
(IBM Bob, Claude Desktop, etc.) can fetch satellite imagery and run
change detection through natural-language tool calls — no browser needed.

Tools
~~~~~
search_satellite_scenes
    Query NASA CMR for MODIS or Landsat granules by bbox + date range.

fetch_scene_pair
    Download a before/after GIBS tile mosaic for a location and dates.

search_sentinel2_scenes
    Search the Copernicus CDSE catalogue for Sentinel-2 L2A products.
    No authentication required for catalogue search.

fetch_sentinel2_pair
    Download a before/after Sentinel-2 L2A pair (10 m resolution) from CDSE.
    Requires CDSE_USER / CDSE_PASSWORD credentials (free account).
    Output GeoTIFFs are multi-band and ready for NDVI via compute_ndvi_diff().

run_change_detection
    Align two local image files and return change statistics.

narrate_change
    Generate a plain-language summary of change statistics via IBM Granite
    (falls back to template if watsonx credentials are absent).

Usage
~~~~~
Register with Bob:

    bob mcp add --name terralens --command "python mcp_server.py"

Or run standalone for testing:

    python mcp_server.py

Requires: mcp  (pip install mcp)
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# MCP server bootstrap
# ---------------------------------------------------------------------------

try:
    from mcp.server.mcpserver.server import MCPServer
except ImportError:
    print(
        "ERROR: 'mcp' package not installed.  Run:  pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

mcp = MCPServer(name="TerraLens")

# ---------------------------------------------------------------------------
# Tool: search_satellite_scenes
# ---------------------------------------------------------------------------

@mcp.tool()
def search_satellite_scenes(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    date_from: str,
    date_to: str,
    layer: str = "MODIS",
    limit: int = 5,
) -> str:
    """
    Search NASA CMR for satellite granules within a bounding box and date range.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : WGS-84 bounding box in decimal degrees.
    date_from : Start date in YYYY-MM-DD format (e.g. "2024-01-01").
    date_to   : End date   in YYYY-MM-DD format (e.g. "2024-06-30").
    layer     : "MODIS" for daily MODIS Terra true-colour, or "LANDSAT" for
                annual Landsat composite.  Default: "MODIS".
    limit     : Maximum number of results to return (default 5).

    Returns
    -------
    JSON string listing matching scenes with date, name, and source.
    """
    from satellite_fetch import search_scenes, LAYER_MODIS, LAYER_LANDSAT

    use_layer = LAYER_LANDSAT if layer.upper() == "LANDSAT" else LAYER_MODIS
    bbox      = (min_lon, min_lat, max_lon, max_lat)

    try:
        scenes = search_scenes(bbox, date_from, date_to, use_layer, limit)
        if not scenes:
            return json.dumps({"scenes": [], "message": "No granules found for this area and date range."})
        return json.dumps({"scenes": [
            {"date": s["date"], "name": s["name"], "source": s["source"]}
            for s in scenes
        ]}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: fetch_scene_pair
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_scene_pair(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    date_from: str,
    date_mid: str,
    date_to: str,
    layer: str = "MODIS",
    out_dir: str = "data",
) -> str:
    """
    Download a before/after satellite image pair for a location and date range.

    Fetches the most recent granule in [date_from, date_mid) as the "before"
    image and in [date_mid, date_to] as the "after" image using NASA GIBS.
    No credentials required.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : WGS-84 bounding box in decimal degrees.
    date_from : Start of the "before" window (YYYY-MM-DD).
    date_mid  : Boundary between before and after windows (YYYY-MM-DD).
    date_to   : End of the "after" window (YYYY-MM-DD).
    layer     : "MODIS" (daily) or "LANDSAT" (annual composite).
    out_dir   : Local directory to save downloaded images (default: "data").

    Returns
    -------
    JSON string with before_path, after_path, and scene metadata.
    """
    from satellite_fetch import (
        fetch_scene_pair as _fetch,
        LAYER_MODIS, LAYER_LANDSAT,
    )

    use_layer = LAYER_LANDSAT if layer.upper() == "LANDSAT" else LAYER_MODIS
    bbox      = (min_lon, min_lat, max_lon, max_lat)

    try:
        before_path, after_path, before_meta, after_meta = _fetch(
            bbox, date_from, date_mid, date_to,
            out_dir=out_dir, layer=use_layer,
        )
        return json.dumps({
            "before_path": before_path,
            "after_path":  after_path,
            "before_date": before_meta["date"],
            "after_date":  after_meta["date"],
            "source":      before_meta["source"],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: run_change_detection
# ---------------------------------------------------------------------------

@mcp.tool()
def run_change_detection(
    before_path: str,
    after_path: str,
    threshold: int = 40,
    bbox_json: str = "",
) -> str:
    """
    Align two satellite images and detect surface changes between them.

    Parameters
    ----------
    before_path : Local path to the "before" image (PNG, JPEG, or GeoTIFF).
    after_path  : Local path to the "after"  image.
    threshold   : Pixel-intensity difference (0-255) to flag as changed.
                  Lower = more sensitive.  Default: 40.
    bbox_json   : Optional JSON string "[min_lon, min_lat, max_lon, max_lat]"
                  used to convert pixel areas to km².

    Returns
    -------
    JSON string with change statistics: change_percent, num_regions,
    changed_km2 (if bbox provided), and alignment warnings.
    """
    from change_detection import load_and_align, detect_change

    if not Path(before_path).exists():
        return json.dumps({"error": f"before_path not found: {before_path}"})
    if not Path(after_path).exists():
        return json.dumps({"error": f"after_path not found: {after_path}"})

    try:
        before, after_aligned, warnings = load_and_align(before_path, after_path)
        mask, stats = detect_change(before, after_aligned, threshold=threshold)

        # Compute km² if bbox provided
        if bbox_json:
            try:
                min_lon, min_lat, max_lon, max_lat = json.loads(bbox_json)
                lon_span_km = abs(max_lon - min_lon) * 111.32 * \
                              __import__("math").cos(
                                  __import__("math").radians((min_lat + max_lat) / 2)
                              )
                lat_span_km = abs(max_lat - min_lat) * 110.574
                total_km2   = lon_span_km * lat_span_km
                stats["changed_km2"] = round(
                    stats["changed_fraction"] * total_km2, 4
                )
            except Exception:
                pass

        result = {
            "change_percent":  stats["change_percent"],
            "changed_km2":     stats["changed_km2"],
            "num_regions":     stats["num_regions"],
            "changed_pixels":  stats["changed_pixels"],
            "total_pixels":    stats["total_pixels"],
            "warnings":        warnings,
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: search_sentinel2_scenes
# ---------------------------------------------------------------------------

@mcp.tool()
def search_sentinel2_scenes(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    date_from: str,
    date_to: str,
    max_cloud_pct: float = 30.0,
    limit: int = 5,
) -> str:
    """
    Search the Copernicus Data Space Ecosystem (CDSE) catalogue for
    Sentinel-2 L2A products within a bounding box and date range.

    No authentication is required for catalogue search.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : WGS-84 bounding box in decimal degrees.
    date_from     : Start date in YYYY-MM-DD format (e.g. "2024-01-01").
    date_to       : End date   in YYYY-MM-DD format (e.g. "2024-06-30").
    max_cloud_pct : Maximum acceptable cloud cover percentage (0-100). Default: 30.
    limit         : Maximum number of results to return (default 5).

    Returns
    -------
    JSON string listing matching Sentinel-2 scenes with date, name, cloud
    cover %, and source.
    """
    from sentinel2_fetch import (
        search_sentinel2_scenes as _search,
        CopernicusSearchError,
    )

    bbox = (min_lon, min_lat, max_lon, max_lat)
    try:
        scenes = _search(bbox, date_from, date_to,
                         max_cloud_pct=max_cloud_pct, limit=limit)
        if not scenes:
            return json.dumps({
                "scenes": [],
                "message": "No Sentinel-2 scenes found for this area and date range.",
            })
        return json.dumps({"scenes": [
            {
                "date":      s["date"],
                "name":      s["name"],
                "cloud_pct": s["cloud_pct"],
                "source":    s["source"],
            }
            for s in scenes
        ]}, indent=2)
    except CopernicusSearchError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: fetch_sentinel2_pair
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_sentinel2_pair(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    date_from: str,
    date_mid: str,
    date_to: str,
    max_cloud_pct: float = 30.0,
    bands: str = "B04,B08",
    out_dir: str = "data",
) -> str:
    """
    Download a before/after Sentinel-2 L2A image pair from the Copernicus
    Data Space Ecosystem.

    Requires CDSE_USER and CDSE_PASSWORD environment variables (free account
    at https://dataspace.copernicus.eu/).

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : WGS-84 bounding box in decimal degrees.
    date_from     : Start of the "before" window (YYYY-MM-DD).
    date_mid      : Boundary between before and after windows (YYYY-MM-DD).
    date_to       : End of the "after" window (YYYY-MM-DD).
    max_cloud_pct : Maximum cloud cover accepted (0-100).  Default: 30.
    bands         : Comma-separated band names to download.
                    Default "B04,B08" (Red + NIR, 10 m — sufficient for NDVI).
                    Other options: B02, B03, B05, B06, B07, B8A, B11, B12, SCL.
    out_dir       : Local directory to save downloaded GeoTIFFs (default: "data").

    Returns
    -------
    JSON string with before_path, after_path, scene dates, cloud cover, and source.
    The output GeoTIFFs have one band per requested band in the order specified.
    For default B04,B08: band 1 = Red, band 2 = NIR.
    Use compute_ndvi_diff(before_path, after_path, nir_band=2, red_band=1) for NDVI.
    """
    from sentinel2_fetch import (
        fetch_sentinel2_pair as _fetch,
        CopernicusAuthError, CopernicusSearchError,
    )

    bbox       = (min_lon, min_lat, max_lon, max_lat)
    bands_list = [b.strip() for b in bands.split(",") if b.strip()]

    try:
        before_path, after_path, before_meta, after_meta = _fetch(
            bbox, date_from, date_mid, date_to,
            out_dir=out_dir,
            max_cloud_pct=max_cloud_pct,
            bands=bands_list,
            include_scl=True,   # always download SCL sidecar via MCP for cloud masking
        )
        return json.dumps({
            "before_path":     before_path,
            "after_path":      after_path,
            "before_scl_path": before_meta.get("scl_path"),
            "after_scl_path":  after_meta.get("scl_path"),
            "before_date":     before_meta["date"],
            "after_date":      after_meta["date"],
            "before_cloud":    before_meta.get("cloud_pct"),
            "after_cloud":     after_meta.get("cloud_pct"),
            "source":          before_meta["source"],
            "bands":           bands_list,
            "cloud_mask_hint": (
                "SCL sidecars downloaded. Pass before_scl_path/after_scl_path "
                "to run_change_detection as bbox_scl_before / bbox_scl_after."
            ),
            "ndvi_hint":    (
                "For NDVI, run: compute_ndvi_diff(before_path, after_path, "
                "nir_band=2, red_band=1)"
                if bands_list == ["B04", "B08"] else ""
            ),
        }, indent=2)
    except CopernicusAuthError as exc:
        return json.dumps({"error": f"Authentication failed: {exc}"})
    except CopernicusSearchError as exc:
        return json.dumps({"error": f"Catalogue search failed: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_catalogue
# ---------------------------------------------------------------------------

@mcp.tool()
def list_catalogue(
    limit: int = 20,
    change_type: str = "",
    min_change_pct: float = 0.0,
    mode: str = "",
) -> str:
    """
    List past TerraLens change-detection analyses from the local scene catalogue.

    Parameters
    ----------
    limit          : Maximum number of records to return (default 20, max 200).
    change_type    : Optional filter — return only records matching this label
                     (e.g. "wildfire", "flooding", "deforestation").
    min_change_pct : Return only records where the scene changed by at least
                     this percentage (0-100).  Default: 0 (no filter).
    mode           : Optional filter by acquisition mode: "upload", "nasa",
                     or "sentinel2".

    Returns
    -------
    JSON string with a list of catalogue entries (id, date, before_name,
    after_name, change_percent, change_type, confidence, narrative excerpt).
    """
    from catalogue import list_entries, catalogue_stats

    try:
        entries = list_entries(
            limit=min(max(1, limit), 200),
            change_type=change_type.strip() or None,
            min_change_pct=min_change_pct if min_change_pct > 0 else None,
            mode=mode.strip() or None,
        )
        stats = catalogue_stats()
        return json.dumps({
            "catalogue_stats": stats,
            "entries": [
                {
                    "id":             e["id"],
                    "analysed_at":    e.get("analysed_at", "")[:10],
                    "before_name":    e.get("before_name"),
                    "after_name":     e.get("after_name"),
                    "mode":           e.get("mode"),
                    "change_percent": e.get("change_percent"),
                    "change_type":    e.get("change_type"),
                    "confidence":     e.get("confidence"),
                    "date_range":     e.get("date_range"),
                    "narrative_excerpt": (e.get("narrative") or "")[:200],
                }
                for e in entries
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_catalogue_entry
# ---------------------------------------------------------------------------

@mcp.tool()
def get_catalogue_entry(entry_id: int) -> str:
    """
    Retrieve the full details of a single catalogue entry by its numeric ID.

    Parameters
    ----------
    entry_id : The integer ID of the catalogue entry to retrieve.
               Use list_catalogue to find IDs.

    Returns
    -------
    JSON string with all stored fields for the entry, including the full
    narrative text, bbox coordinates, and all change statistics.
    """
    from catalogue import get_entry

    try:
        entry = get_entry(entry_id)
        if entry is None:
            return json.dumps({"error": f"No catalogue entry with id={entry_id}."})
        return json.dumps(entry, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: narrate_change
# ---------------------------------------------------------------------------

@mcp.tool()
def narrate_change(
    change_percent: float,
    num_regions: int,
    changed_pixels: int,
    total_pixels: int,
    date_range: str = "",
    largest_area_px: int = 0,
) -> str:
    """
    Generate a plain-language explanation of satellite change-detection results.

    Uses IBM Granite via watsonx.ai if WATSONX_API_KEY and WATSONX_PROJECT_ID
    are set in the environment; otherwise returns a template-based summary.

    Parameters
    ----------
    change_percent  : Percentage of the scene that changed (0-100).
    num_regions     : Number of distinct changed regions detected.
    changed_pixels  : Total number of changed pixels.
    total_pixels    : Total pixels in the scene.
    date_range      : Human-readable observation window (e.g. "Jan → Jun 2024").
    largest_area_px : Area in pixels of the largest changed region.

    Returns
    -------
    JSON string with "narrative" (plain text) and "source" ("granite" or "template").
    """
    from narrate import narrate_with_granite

    stats = {
        "change_percent":  change_percent,
        "num_regions":     num_regions,
        "changed_pixels":  changed_pixels,
        "total_pixels":    total_pixels,
        "changed_fraction": changed_pixels / total_pixels if total_pixels else 0.0,
        "regions": [{"area_px": largest_area_px}] if largest_area_px else [],
    }

    try:
        narrative, source, classified = narrate_with_granite(stats, date_range=date_range)
        return json.dumps({
            "narrative":   narrative,
            "source":      source,
            "change_type": classified.get("change_type"),
            "confidence":  classified.get("confidence"),
            "caveat":      classified.get("caveat"),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: schedule_aoi
# ---------------------------------------------------------------------------

@mcp.tool()
def schedule_aoi(
    name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    source: str = "modis",
    max_cloud_pct: float = 30.0,
    bands: str = "B04,B08",
    threshold: int = 40,
    notes: str = "",
) -> str:
    """
    Register a new watched Area of Interest (AOI) with the TerraLens scheduler.

    Once registered, running ``python scheduler.py`` (or the daemon) will
    automatically check for new satellite acquisitions over this AOI, run
    change detection + Granite narration, and save results to the catalogue.

    Parameters
    ----------
    name          : Human-readable label for the AOI.
    min_lon, min_lat, max_lon, max_lat : WGS-84 bounding box.
    source        : Imagery source — "modis" (daily, no auth),
                    "landsat" (annual, no auth), or "sentinel2" (10 m, requires
                    CDSE_USER/CDSE_PASSWORD).
    max_cloud_pct : Sentinel-2 only — reject scenes with cloud cover above this
                    value (0-100).  Default: 30.
    bands         : Sentinel-2 only — comma-separated band names to download.
                    Default "B04,B08" (Red + NIR, sufficient for NDVI + change).
    threshold     : Pixel-intensity difference threshold for change detection
                    (0-255).  Lower = more sensitive.  Default: 40.
    notes         : Optional free-text notes stored with the AOI.

    Returns
    -------
    JSON string with the new AOI id and a confirmation message.
    """
    from scheduler import add_watched_aoi

    bbox       = (min_lon, min_lat, max_lon, max_lat)
    bands_list = [b.strip() for b in bands.split(",") if b.strip()]

    try:
        aoi_id = add_watched_aoi(
            name=name,
            bbox=bbox,
            source=source.lower(),
            max_cloud_pct=max_cloud_pct,
            bands=bands_list,
            threshold=threshold,
            notes=notes,
        )
        return json.dumps({
            "aoi_id":  aoi_id,
            "name":    name,
            "source":  source,
            "bbox":    list(bbox),
            "message": (
                f"AOI '{name}' registered with id={aoi_id}.  "
                "Run 'python scheduler.py' to trigger the first poll, "
                "or 'python scheduler.py --daemon' for continuous monitoring."
            ),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_watched_aois
# ---------------------------------------------------------------------------

@mcp.tool()
def list_watched_aois(enabled_only: bool = False) -> str:
    """
    List all Areas of Interest registered with the TerraLens scheduler.

    Parameters
    ----------
    enabled_only : If true, only return AOIs that are currently active (not paused).

    Returns
    -------
    JSON string with a list of AOI records, each including id, name, source,
    bbox, last check time, last scene date, and check count.
    """
    from scheduler import list_watched_aois as _list

    try:
        aois = _list(enabled_only=enabled_only)
        return json.dumps({"aois": aois, "total": len(aois)}, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
