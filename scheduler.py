"""
scheduler.py
------------
TerraLens automated scene scheduler.

Polls NASA GIBS/MODIS and Copernicus Sentinel-2 for new satellite acquisitions
over saved Areas of Interest (AOIs), runs change detection + Granite narration
automatically, and stores every result in the scene catalogue.

A "watched AOI" is a named bounding box + source combination stored in the
same SQLite database used by catalogue.py.  Every time the scheduler runs, it
checks whether a new scene has been acquired since the AOI was last processed,
downloads the pair, analyses it, and saves the result.

Usage
~~~~~
Run once (e.g. from a cron job or CI pipeline):

    python scheduler.py

Run as a persistent daemon polling every N minutes:

    python scheduler.py --daemon --interval 360

Register a new AOI from the command line:

    python scheduler.py --add-aoi \
        --name "Amazon deforestation watch" \
        --source sentinel2 \
        --bbox "-62.5,-4.0,-60.0,-2.0" \
        --max-cloud 30

List all watched AOIs:

    python scheduler.py --list-aois

Environment variables
~~~~~~~~~~~~~~~~~~~~~
TERRALENS_DB_PATH       SQLite database path (shared with catalogue.py)
WATSONX_API_KEY         }
WATSONX_PROJECT_ID      } passed through to narrate_with_granite()
WATSONX_URL             }
CDSE_USER               } required for Sentinel-2 source
CDSE_PASSWORD           }
SCHEDULER_REPORT_DIR    directory to write plain-text reports (default: results/scheduler)
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared database handle (same DB as catalogue.py)
# ---------------------------------------------------------------------------

_DEFAULT_DB   = os.path.join(os.path.dirname(__file__), "terralens_catalogue.db")
_DB_PATH      = os.getenv("TERRALENS_DB_PATH", _DEFAULT_DB)
_REPORT_DIR   = os.getenv("SCHEDULER_REPORT_DIR",
                           os.path.join(os.path.dirname(__file__), "results", "scheduler"))

# ---------------------------------------------------------------------------
# Schema — watched_aois table
# ---------------------------------------------------------------------------

_CREATE_WATCHED = """
CREATE TABLE IF NOT EXISTS watched_aois (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'modis',  -- 'modis' | 'landsat' | 'sentinel2'
    min_lon         REAL    NOT NULL,
    min_lat         REAL    NOT NULL,
    max_lon         REAL    NOT NULL,
    max_lat         REAL    NOT NULL,
    max_cloud_pct   REAL    NOT NULL DEFAULT 30.0,
    bands           TEXT    NOT NULL DEFAULT 'B04,B08', -- comma-separated, Sentinel-2 only
    threshold       INTEGER NOT NULL DEFAULT 40,
    enabled         INTEGER NOT NULL DEFAULT 1,         -- 0 = paused
    created_at      TEXT    NOT NULL,
    last_checked_at TEXT,                               -- ISO-8601 UTC
    last_scene_date TEXT,                               -- sensing date of the last processed scene
    check_count     INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);
"""


# ---------------------------------------------------------------------------
# Internal DB helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    db_dir = os.path.dirname(_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(_CREATE_WATCHED)
    con.commit()
    return con


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Public AOI management API
# ---------------------------------------------------------------------------

def add_watched_aoi(
    name: str,
    bbox: tuple[float, float, float, float],
    source: str = "modis",
    max_cloud_pct: float = 30.0,
    bands: Optional[list[str]] = None,
    threshold: int = 40,
    notes: str = "",
) -> int:
    """
    Register a new watched AOI in the database.

    Parameters
    ----------
    name          : Human-readable label, e.g. "Amazon deforestation watch".
    bbox          : (min_lon, min_lat, max_lon, max_lat) in WGS-84 degrees.
    source        : "modis", "landsat", or "sentinel2".
    max_cloud_pct : Sentinel-2 only — maximum acceptable cloud cover (0-100).
    bands         : Sentinel-2 only — list of band names (default ["B04","B08"]).
    threshold     : Pixel-intensity difference threshold for change detection (0-255).
    notes         : Optional free-text notes stored with the AOI record.

    Returns
    -------
    int — the auto-assigned AOI ID.
    """
    source = source.lower()
    if source not in ("modis", "landsat", "sentinel2"):
        raise ValueError(f"source must be 'modis', 'landsat', or 'sentinel2'; got '{source}'")

    bands_str = ",".join(bands) if bands else "B04,B08"
    min_lon, min_lat, max_lon, max_lat = bbox

    with _conn() as con:
        cur = con.execute(
            """INSERT INTO watched_aois
               (name, source, min_lon, min_lat, max_lon, max_lat,
                max_cloud_pct, bands, threshold, enabled, created_at, notes)
               VALUES (?,?,?,?,?,?,?,?,?,1,?,?)""",
            (name, source, min_lon, min_lat, max_lon, max_lat,
             max_cloud_pct, bands_str, threshold, _now_utc(), notes),
        )
        aoi_id = cur.lastrowid
        con.commit()
    return aoi_id


def list_watched_aois(enabled_only: bool = False) -> list[dict]:
    """Return all (or only enabled) watched AOIs."""
    sql = "SELECT * FROM watched_aois"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY id"
    with _conn() as con:
        rows = con.execute(sql).fetchall()
    return [dict(r) for r in rows]


def disable_aoi(aoi_id: int) -> None:
    """Pause an AOI (set enabled=0)."""
    with _conn() as con:
        con.execute("UPDATE watched_aois SET enabled=0 WHERE id=?", (aoi_id,))
        con.commit()


def enable_aoi(aoi_id: int) -> None:
    """Resume a paused AOI."""
    with _conn() as con:
        con.execute("UPDATE watched_aois SET enabled=1 WHERE id=?", (aoi_id,))
        con.commit()


def delete_aoi(aoi_id: int) -> bool:
    """Remove an AOI permanently. Returns True if a row was deleted."""
    with _conn() as con:
        cur = con.execute("DELETE FROM watched_aois WHERE id=?", (aoi_id,))
        con.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Core poll logic
# ---------------------------------------------------------------------------

def _poll_aoi(aoi: dict, verbose: bool = True) -> Optional[dict]:
    """
    Check one AOI for a new acquisition since *last_scene_date*.

    If a new scene pair is found, runs change detection + narration, saves to
    catalogue, and writes a plain-text report.

    Returns a summary dict on success, None if no new scene was available.
    """
    from change_detection import load_and_align, detect_change, apply_scl_mask
    from catalogue import add_entry
    from narrate import narrate_with_granite

    aoi_id   = aoi["id"]
    name     = aoi["name"]
    source   = aoi["source"]
    bbox     = (aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"])
    threshold = aoi["threshold"]

    # Date windows: "before" ends where we last stopped; "after" is today
    today       = datetime.now(timezone.utc).date()
    last_date   = aoi.get("last_scene_date")  # YYYY-MM-DD or None

    # On first run: use 30 days ago as "before" window
    if last_date:
        date_from = last_date
    else:
        date_from = _date_str(today - timedelta(days=30))

    # Split roughly in the middle so we always have a before/after pair
    try:
        before_dt = datetime.strptime(date_from, "%Y-%m-%d").date()
    except ValueError:
        before_dt = today - timedelta(days=30)

    midpoint  = before_dt + (today - before_dt) // 2
    date_mid  = _date_str(midpoint)
    date_to   = _date_str(today)

    if midpoint >= today:
        if verbose:
            print(f"  [skip] AOI {aoi_id} '{name}': midpoint ({date_mid}) >= today, "
                  "need a wider date window.")
        return None

    if verbose:
        print(f"  AOI {aoi_id} '{name}' | source={source} | "
              f"{date_from} → {date_mid} → {date_to}")

    before_path = after_path = before_scl = after_scl = None

    try:
        if source in ("modis", "landsat"):
            from satellite_fetch import (
                fetch_scene_pair, LAYER_MODIS, LAYER_LANDSAT,
            )
            layer = LAYER_LANDSAT if source == "landsat" else LAYER_MODIS
            before_path, after_path, before_meta, after_meta = fetch_scene_pair(
                bbox, date_from, date_mid, date_to,
                out_dir="data", layer=layer,
            )
            new_scene_date = after_meta.get("date", date_to)
            mode_label     = source

        elif source == "sentinel2":
            from sentinel2_fetch import (
                fetch_sentinel2_pair,
                CopernicusAuthError, CopernicusSearchError,
            )
            bands = [b.strip() for b in aoi.get("bands", "B04,B08").split(",") if b.strip()]
            max_cloud = float(aoi.get("max_cloud_pct", 30.0))
            before_path, after_path, before_meta, after_meta = fetch_sentinel2_pair(
                bbox, date_from, date_mid, date_to,
                out_dir="data",
                max_cloud_pct=max_cloud,
                bands=bands,
                include_scl=True,
            )
            before_scl     = before_meta.get("scl_path")
            after_scl      = after_meta.get("scl_path")
            new_scene_date = after_meta.get("date", date_to)
            mode_label     = "sentinel2"
        else:
            if verbose:
                print(f"  [skip] Unknown source '{source}'")
            return None

    except Exception as exc:
        if verbose:
            print(f"  [error] AOI {aoi_id}: fetch failed — {exc}")
        _update_aoi_checked(aoi_id)
        return None

    # --- Load, align, detect ---
    try:
        before_img, after_aligned, warnings = load_and_align(before_path, after_path)
        if before_img is None or after_aligned is None:
            raise RuntimeError("Image load returned None")

        # Cloud masking
        cloud_mask = None
        if before_scl or after_scl:
            import numpy as np
            target = before_img.shape[:2]
            masks  = []
            for p in (before_scl, after_scl):
                if p and os.path.exists(p):
                    try:
                        cm, _ = apply_scl_mask(p, target)
                        masks.append(cm)
                    except Exception:
                        pass
            if masks:
                cloud_mask = np.zeros(target, dtype=np.uint8)
                for m in masks:
                    cloud_mask = np.where(m == 255, np.uint8(255), cloud_mask)

        mask, stats = detect_change(before_img, after_aligned,
                                    threshold=threshold, cloud_mask=cloud_mask)

        # Real-world km²
        import math
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lat     = (min_lat + max_lat) / 2
        lon_span_km = abs(max_lon - min_lon) * 111.32 * math.cos(math.radians(mid_lat))
        lat_span_km = abs(max_lat - min_lat) * 110.574
        stats["changed_km2"] = round(stats["changed_fraction"] * lon_span_km * lat_span_km, 4)

        date_range = f"{date_from} → {date_to}"
        narrative, narr_source, classified = narrate_with_granite(stats, date_range=date_range)

        # Write plain-text report
        report_text = _build_report(name, aoi, date_from, date_to, stats, narrative, classified, warnings)
        report_path = _write_report(aoi_id, name, date_to, report_text)

        # Save to catalogue
        cat_id = add_entry({
            "before_name":    before_meta.get("name") or before_path,
            "after_name":     after_meta.get("name")  or after_path,
            "date_range":     date_range,
            "mode":           mode_label,
            "change_percent": stats.get("change_percent"),
            "num_regions":    stats.get("num_regions"),
            "changed_pixels": stats.get("changed_pixels"),
            "total_pixels":   stats.get("total_pixels"),
            "changed_km2":    stats.get("changed_km2"),
            "change_type":    classified.get("change_type"),
            "confidence":     classified.get("confidence"),
            "narrative":      narrative,
            "bbox_json":      list(bbox),
            "extra_json":     {"aoi_id": aoi_id, "aoi_name": name,
                               "cloud_masked_pct": stats.get("cloud_masked_pct", 0.0)},
        })

        _update_aoi_checked(aoi_id, new_scene_date=new_scene_date)

        summary = {
            "aoi_id":         aoi_id,
            "aoi_name":       name,
            "catalogue_id":   cat_id,
            "change_percent": stats["change_percent"],
            "change_type":    classified.get("change_type"),
            "confidence":     classified.get("confidence"),
            "report_path":    report_path,
        }
        if verbose:
            print(f"  ✅ AOI {aoi_id}: {stats['change_percent']}% change "
                  f"({classified.get('change_type', 'unknown')}, "
                  f"{classified.get('confidence', '?')} confidence) → report: {report_path}")
        return summary

    except Exception as exc:
        if verbose:
            print(f"  [error] AOI {aoi_id}: analysis failed — {exc}")
        _update_aoi_checked(aoi_id)
        return None
    finally:
        # Temp files are managed by fetch functions; SCL sidecars live in data/
        pass


def _update_aoi_checked(aoi_id: int, new_scene_date: Optional[str] = None) -> None:
    """Bump last_checked_at and check_count; optionally update last_scene_date."""
    with _conn() as con:
        if new_scene_date:
            con.execute(
                "UPDATE watched_aois SET last_checked_at=?, last_scene_date=?, "
                "check_count=check_count+1 WHERE id=?",
                (_now_utc(), new_scene_date, aoi_id),
            )
        else:
            con.execute(
                "UPDATE watched_aois SET last_checked_at=?, check_count=check_count+1 WHERE id=?",
                (_now_utc(), aoi_id),
            )
        con.commit()


def _build_report(name, aoi, date_from, date_to, stats, narrative, classified, warnings) -> str:
    lines = [
        "=" * 72,
        f"TerraLens Scheduled Change Report",
        f"AOI     : {name} (id={aoi['id']})",
        f"Source  : {aoi['source']}",
        f"Window  : {date_from} → {date_to}",
        f"Generated: {_now_utc()}",
        "=" * 72,
        "",
        "── Change Statistics ──────────────────────────────────────────────",
        f"  Change %       : {stats['change_percent']}%",
        f"  Changed pixels : {stats['changed_pixels']:,} / {stats['total_pixels']:,}",
        f"  Distinct regions: {stats['num_regions']}",
        f"  Changed area    : {stats.get('changed_km2', 'n/a')} km²",
        f"  Cloud masked    : {stats.get('cloud_masked_pct', 0.0):.1f}%",
        "",
        "── Classifier Output ──────────────────────────────────────────────",
        f"  Change type : {classified.get('change_type', 'unknown')}",
        f"  Confidence  : {classified.get('confidence', 'Low')}",
        f"  Caveat      : {classified.get('caveat', '')}",
        "",
        "── AI Narrative ───────────────────────────────────────────────────",
        narrative,
        "",
    ]
    if warnings:
        lines += ["── Warnings ────────────────────────────────────────────────────"]
        lines += [f"  {w}" for w in warnings]
        lines += [""]
    return "\n".join(lines)


def _write_report(aoi_id: int, name: str, date: str, text: str) -> str:
    Path(_REPORT_DIR).mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)[:40].strip()
    filename  = f"aoi_{aoi_id}_{safe_name}_{date}.txt".replace(" ", "_")
    path      = os.path.join(_REPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ---------------------------------------------------------------------------
# Poll all enabled AOIs
# ---------------------------------------------------------------------------

def poll_once(verbose: bool = True) -> list[dict]:
    """
    Check every enabled AOI once and return a list of result summaries.

    Safe to run from a cron job:  ``0 */6 * * * python scheduler.py``
    """
    aois = list_watched_aois(enabled_only=True)
    if not aois:
        if verbose:
            print("No enabled AOIs to check.")
        return []

    if verbose:
        print(f"[TerraLens Scheduler] Checking {len(aois)} AOI(s) — {_now_utc()}")

    results = []
    for aoi in aois:
        result = _poll_aoi(aoi, verbose=verbose)
        if result:
            results.append(result)

    if verbose:
        print(f"[TerraLens Scheduler] Done — {len(results)} new result(s).")
    return results


def run_daemon(interval_minutes: int = 360, verbose: bool = True) -> None:
    """
    Run the scheduler as a persistent background daemon.

    Calls poll_once() every *interval_minutes* minutes until interrupted.

    Parameters
    ----------
    interval_minutes : Poll interval in minutes (default 360 = 6 hours).
                       Sentinel-2 has a 5-day revisit; polling more often than
                       once per day wastes bandwidth without finding new scenes.
    """
    if verbose:
        print(f"[TerraLens Scheduler daemon] starting — polling every {interval_minutes} min")
    try:
        while True:
            poll_once(verbose=verbose)
            if verbose:
                print(f"  Sleeping {interval_minutes} min until next poll…")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        if verbose:
            print("\n[TerraLens Scheduler] stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scheduler",
        description="TerraLens automated scene scheduler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--daemon",     action="store_true",
                   help="Run as persistent daemon (polls repeatedly until killed).")
    p.add_argument("--interval",   type=int, default=360,
                   help="Daemon poll interval in minutes.")
    p.add_argument("--add-aoi",    action="store_true",
                   help="Register a new watched AOI and exit.")
    p.add_argument("--list-aois",  action="store_true",
                   help="Print all watched AOIs and exit.")
    p.add_argument("--disable-aoi", type=int, metavar="ID",
                   help="Pause (disable) the AOI with this ID.")
    p.add_argument("--enable-aoi",  type=int, metavar="ID",
                   help="Resume the AOI with this ID.")
    p.add_argument("--delete-aoi",  type=int, metavar="ID",
                   help="Permanently remove the AOI with this ID.")
    # AOI fields for --add-aoi
    p.add_argument("--name",        default="Unnamed AOI")
    p.add_argument("--source",      default="modis",
                   choices=["modis", "landsat", "sentinel2"])
    p.add_argument("--bbox",        default="-10,35,10,55",
                   help="min_lon,min_lat,max_lon,max_lat")
    p.add_argument("--max-cloud",   type=float, default=30.0)
    p.add_argument("--bands",       default="B04,B08",
                   help="Comma-separated Sentinel-2 band names.")
    p.add_argument("--threshold",   type=int, default=40)
    p.add_argument("--notes",       default="")
    p.add_argument("--quiet",       action="store_true")
    return p


if __name__ == "__main__":
    args   = _build_parser().parse_args()
    verbose = not args.quiet

    if args.list_aois:
        aois = list_watched_aois()
        if not aois:
            print("No watched AOIs registered.")
        else:
            print(f"{'ID':>4}  {'Enabled':>7}  {'Source':>10}  {'Last checked':>20}  Name")
            print("-" * 72)
            for a in aois:
                print(f"{a['id']:>4}  {'yes' if a['enabled'] else 'no':>7}  "
                      f"{a['source']:>10}  "
                      f"{(a['last_checked_at'] or 'never')[:19]:>20}  "
                      f"{a['name']}")
        sys.exit(0)

    if args.add_aoi:
        try:
            parts = [float(x) for x in args.bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
        except ValueError:
            print("ERROR: --bbox must be 'min_lon,min_lat,max_lon,max_lat'", file=sys.stderr)
            sys.exit(1)
        aoi_id = add_watched_aoi(
            name=args.name,
            bbox=tuple(parts),
            source=args.source,
            max_cloud_pct=args.max_cloud,
            bands=[b.strip() for b in args.bands.split(",") if b.strip()],
            threshold=args.threshold,
            notes=args.notes,
        )
        print(f"✅ AOI registered with id={aoi_id}: {args.name}")
        sys.exit(0)

    if args.disable_aoi:
        disable_aoi(args.disable_aoi)
        print(f"AOI {args.disable_aoi} paused.")
        sys.exit(0)

    if args.enable_aoi:
        enable_aoi(args.enable_aoi)
        print(f"AOI {args.enable_aoi} resumed.")
        sys.exit(0)

    if args.delete_aoi:
        if delete_aoi(args.delete_aoi):
            print(f"AOI {args.delete_aoi} deleted.")
        else:
            print(f"AOI {args.delete_aoi} not found.")
        sys.exit(0)

    if args.daemon:
        run_daemon(interval_minutes=args.interval, verbose=verbose)
    else:
        poll_once(verbose=verbose)
