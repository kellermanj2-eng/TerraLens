"""
catalogue.py
------------
TerraLens scene catalogue — persistent storage for change-detection analyses.

Every time a before/after pair is analysed in the Streamlit app or via the
MCP server, a summary record is appended to a local SQLite database
(``terralens_catalogue.db`` by default).  Records can be searched, listed,
and replayed from the app's "Catalogue" tab or via the ``list_catalogue`` and
``get_catalogue_entry`` MCP tools.

When the environment variable ``WATSONX_DATA_CONNECTION_URL`` is set, the
module additionally mirrors records to a Watsonx.data (Lakehouse) Iceberg
table via a JDBC-compatible connection string, enabling enterprise-scale
catalogue management and SQL analytics over the stored events.

Public API
~~~~~~~~~~
add_entry(entry)
    Persist a single analysis record.  Returns the assigned integer ID.

list_entries(limit=50, change_type=None, min_change_pct=None)
    Return a list of stored records, newest first, with optional filters.

get_entry(entry_id)
    Retrieve a single record by its integer ID.

delete_entry(entry_id)
    Remove a record from the catalogue.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
TERRALENS_DB_PATH           Local SQLite path (default: terralens_catalogue.db)
WATSONX_DATA_CONNECTION_URL Optional JDBC URL for Watsonx.data Iceberg mirror
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "terralens_catalogue.db")
_DB_PATH    = os.getenv("TERRALENS_DB_PATH", _DEFAULT_DB)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysed_at     TEXT    NOT NULL,   -- ISO-8601 UTC timestamp
    before_name     TEXT,               -- original filename / scene name
    after_name      TEXT,
    date_range      TEXT,               -- human-readable observation window
    mode            TEXT,               -- "upload" | "nasa" | "sentinel2"
    change_percent  REAL,
    num_regions     INTEGER,
    changed_pixels  INTEGER,
    total_pixels    INTEGER,
    changed_km2     REAL,
    change_type     TEXT,               -- classifier label
    confidence      TEXT,               -- Low / Medium / High
    narrative       TEXT,               -- full Granite / template narrative
    bbox_json       TEXT,               -- "[min_lon, min_lat, max_lon, max_lat]" or NULL
    extra_json      TEXT                -- arbitrary extra fields as JSON
);
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    """Open (and if necessary create) the catalogue database."""
    db_dir = os.path.dirname(_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(_CREATE_TABLE)
    con.commit()
    return con


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Deserialise JSON fields
    for key in ("bbox_json", "extra_json"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (ValueError, TypeError):
                pass
    return d


def _mirror_to_watsonx_data(entry: dict) -> None:
    """
    Optional: write the entry to a Watsonx.data Iceberg table.

    Only executed when WATSONX_DATA_CONNECTION_URL is set.  Errors are
    silently swallowed so a missing/misconfigured connection never blocks
    the local catalogue write.
    """
    url = os.getenv("WATSONX_DATA_CONNECTION_URL", "").strip()
    if not url:
        return
    try:
        # JayDeBeApi or ibm_db could be used here; we use a best-effort
        # approach that does nothing if the driver is not installed.
        import jaydebeapi  # type: ignore
        cols = (
            "analysed_at", "before_name", "after_name", "date_range", "mode",
            "change_percent", "num_regions", "changed_pixels", "total_pixels",
            "changed_km2", "change_type", "confidence", "narrative",
            "bbox_json", "extra_json",
        )
        vals = tuple(
            json.dumps(entry.get(c)) if isinstance(entry.get(c), (dict, list)) else entry.get(c)
            for c in cols
        )
        placeholders = ", ".join(["?"] * len(cols))
        sql = (
            f"INSERT INTO terralens_analyses ({', '.join(cols)}) "
            f"VALUES ({placeholders})"
        )
        with jaydebeapi.connect(url) as wcon:
            with wcon.cursor() as cur:
                cur.execute(sql, vals)
        wcon.commit()
    except Exception:
        pass  # Watsonx.data mirror is best-effort


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_entry(entry: dict) -> int:
    """
    Persist an analysis record to the catalogue.

    Parameters
    ----------
    entry : dict with any subset of the schema columns.  At minimum, supply
            ``change_percent`` and one of ``before_name`` / ``after_name``.
            ``analysed_at`` is set automatically to the current UTC time if
            not provided.

    Returns
    -------
    int — the auto-assigned record ID.
    """
    entry = dict(entry)  # don't mutate caller's dict
    entry.setdefault("analysed_at", datetime.now(timezone.utc).isoformat())

    # Serialise list/dict fields
    for key in ("bbox_json", "extra_json"):
        if isinstance(entry.get(key), (dict, list)):
            entry[key] = json.dumps(entry[key])

    columns = [
        "analysed_at", "before_name", "after_name", "date_range", "mode",
        "change_percent", "num_regions", "changed_pixels", "total_pixels",
        "changed_km2", "change_type", "confidence", "narrative",
        "bbox_json", "extra_json",
    ]
    present = {k: entry[k] for k in columns if k in entry}

    sql = (
        f"INSERT INTO analyses ({', '.join(present.keys())}) "
        f"VALUES ({', '.join(['?'] * len(present))})"
    )

    with _conn() as con:
        cur = con.execute(sql, list(present.values()))
        row_id = cur.lastrowid
        con.commit()

    _mirror_to_watsonx_data(entry)
    return row_id


def list_entries(
    limit: int = 50,
    change_type: Optional[str] = None,
    min_change_pct: Optional[float] = None,
    mode: Optional[str] = None,
) -> list[dict]:
    """
    Return catalogue records, newest first.

    Parameters
    ----------
    limit          : Maximum number of records to return (default 50).
    change_type    : Filter to a specific classifier label (e.g. "wildfire").
    min_change_pct : Only return records where change_percent ≥ this value.
    mode           : Filter by acquisition mode ("upload", "nasa", "sentinel2").

    Returns
    -------
    list of dicts, each representing one analysis record.
    """
    conditions: list[str] = []
    params:     list      = []

    if change_type:
        conditions.append("LOWER(change_type) = LOWER(?)")
        params.append(change_type)
    if min_change_pct is not None:
        conditions.append("change_percent >= ?")
        params.append(min_change_pct)
    if mode:
        conditions.append("LOWER(mode) = LOWER(?)")
        params.append(mode)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql   = f"SELECT * FROM analyses {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _conn() as con:
        rows = con.execute(sql, params).fetchall()

    return [_row_to_dict(r) for r in rows]


def get_entry(entry_id: int) -> Optional[dict]:
    """
    Retrieve a single catalogue record by ID.

    Returns None if the ID does not exist.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM analyses WHERE id = ?", (entry_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_entry(entry_id: int) -> bool:
    """
    Delete a catalogue record.

    Returns True if a row was deleted, False if the ID was not found.
    """
    with _conn() as con:
        cur = con.execute("DELETE FROM analyses WHERE id = ?", (entry_id,))
        con.commit()
    return cur.rowcount > 0


def catalogue_stats() -> dict:
    """
    Return aggregate statistics about the catalogue contents.

    Returns a dict with total_analyses, by_change_type, by_mode,
    avg_change_pct, and last_analysed_at.
    """
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        by_type = {
            row[0] or "unknown": row[1]
            for row in con.execute(
                "SELECT change_type, COUNT(*) FROM analyses GROUP BY change_type"
            ).fetchall()
        }
        by_mode = {
            row[0] or "unknown": row[1]
            for row in con.execute(
                "SELECT mode, COUNT(*) FROM analyses GROUP BY mode"
            ).fetchall()
        }
        avg_row = con.execute(
            "SELECT AVG(change_percent) FROM analyses WHERE change_percent IS NOT NULL"
        ).fetchone()
        avg_pct = round(avg_row[0], 2) if avg_row[0] is not None else None

        last_row = con.execute(
            "SELECT analysed_at FROM analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_at = last_row[0] if last_row else None

    return {
        "total_analyses":  total,
        "by_change_type":  by_type,
        "by_mode":         by_mode,
        "avg_change_pct":  avg_pct,
        "last_analysed_at": last_at,
    }
