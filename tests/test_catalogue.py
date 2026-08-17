"""
tests/test_catalogue.py
------------------------
Unit tests for catalogue.py — SQLite-backed scene catalogue.
Uses a temporary database file so the production DB is never touched.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Redirect all catalogue operations to a fresh temp database."""
    db_file = str(tmp_path / "test_catalogue.db")
    monkeypatch.setenv("TERRALENS_DB_PATH", db_file)
    # Re-import so the module picks up the new env var
    import importlib
    import catalogue as cat
    importlib.reload(cat)
    # Patch the module-level _DB_PATH in catalogue
    monkeypatch.setattr("catalogue._DB_PATH", db_file)
    return db_file


import catalogue


# ---------------------------------------------------------------------------
# add_entry / get_entry
# ---------------------------------------------------------------------------

class TestAddGet:
    def test_add_returns_integer_id(self):
        entry_id = catalogue.add_entry({
            "before_name": "before.png",
            "after_name":  "after.png",
            "change_percent": 12.5,
            "num_regions":    3,
        })
        assert isinstance(entry_id, int)
        assert entry_id >= 1

    def test_get_entry_round_trips_fields(self):
        eid = catalogue.add_entry({
            "before_name":    "b.png",
            "after_name":     "a.png",
            "change_percent": 7.3,
            "num_regions":    2,
            "change_type":    "wildfire",
            "confidence":     "High",
            "narrative":      "Test narrative.",
            "mode":           "upload",
        })
        entry = catalogue.get_entry(eid)
        assert entry is not None
        assert entry["change_type"] == "wildfire"
        assert entry["confidence"] == "High"
        assert entry["narrative"] == "Test narrative."

    def test_get_nonexistent_returns_none(self):
        assert catalogue.get_entry(99999) is None

    def test_bbox_json_serialised_and_deserialised(self):
        bbox = [-10.0, 35.0, 10.0, 55.0]
        eid = catalogue.add_entry({"bbox_json": bbox, "change_percent": 5.0})
        entry = catalogue.get_entry(eid)
        assert entry["bbox_json"] == bbox

    def test_analysed_at_auto_set(self):
        eid = catalogue.add_entry({"change_percent": 1.0})
        entry = catalogue.get_entry(eid)
        assert entry["analysed_at"] is not None
        assert "T" in entry["analysed_at"]   # ISO format


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------

class TestListEntries:
    def _populate(self, n=5):
        ids = []
        for i in range(n):
            ids.append(catalogue.add_entry({
                "change_percent": float(i * 10),
                "change_type": "wildfire" if i % 2 == 0 else "flooding",
                "mode": "nasa" if i < 3 else "upload",
            }))
        return ids

    def test_list_returns_newest_first(self):
        self._populate(3)
        entries = catalogue.list_entries()
        ids = [e["id"] for e in entries]
        assert ids == sorted(ids, reverse=True)

    def test_filter_by_change_type(self):
        self._populate(6)
        entries = catalogue.list_entries(change_type="wildfire")
        for e in entries:
            assert e["change_type"] == "wildfire"

    def test_filter_by_min_change_pct(self):
        self._populate(5)
        entries = catalogue.list_entries(min_change_pct=25.0)
        for e in entries:
            assert e["change_percent"] >= 25.0

    def test_filter_by_mode(self):
        self._populate(5)
        entries = catalogue.list_entries(mode="nasa")
        for e in entries:
            assert e["mode"] == "nasa"

    def test_limit_respected(self):
        self._populate(10)
        entries = catalogue.list_entries(limit=3)
        assert len(entries) <= 3


# ---------------------------------------------------------------------------
# delete_entry
# ---------------------------------------------------------------------------

class TestDeleteEntry:
    def test_delete_existing_returns_true(self):
        eid = catalogue.add_entry({"change_percent": 5.0})
        assert catalogue.delete_entry(eid) is True
        assert catalogue.get_entry(eid) is None

    def test_delete_nonexistent_returns_false(self):
        assert catalogue.delete_entry(99999) is False


# ---------------------------------------------------------------------------
# catalogue_stats
# ---------------------------------------------------------------------------

class TestCatalogueStats:
    def test_empty_catalogue(self):
        stats = catalogue.catalogue_stats()
        assert stats["total_analyses"] == 0
        assert stats["avg_change_pct"] is None

    def test_stats_after_inserts(self):
        catalogue.add_entry({"change_percent": 10.0, "change_type": "wildfire", "mode": "nasa"})
        catalogue.add_entry({"change_percent": 30.0, "change_type": "flooding", "mode": "upload"})
        stats = catalogue.catalogue_stats()
        assert stats["total_analyses"] == 2
        assert abs(stats["avg_change_pct"] - 20.0) < 0.01
        assert "wildfire" in stats["by_change_type"]
        assert "flooding" in stats["by_change_type"]
