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
        narrative, source = narrate_with_granite(stats, date_range=date_range)
        return json.dumps({"narrative": narrative, "source": source}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())
