"""
tests/test_scheduler.py
------------------------
Unit tests for scheduler.py — AOI management and poll infrastructure.

Tests cover only the database / management layer.  The actual fetch + analysis
loop is not exercised (it would require live network + satellite credentials).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Isolate scheduler DB to a temp file and reload the module."""
    db_file = str(tmp_path / "test_scheduler.db")
    monkeypatch.setenv("TERRALENS_DB_PATH", db_file)
    import importlib
    import scheduler as sched
    importlib.reload(sched)
    monkeypatch.setattr("scheduler._DB_PATH", db_file)
    monkeypatch.setattr("catalogue._DB_PATH", db_file)
    return db_file


import scheduler


SAMPLE_BBOX = (-10.0, 35.0, 10.0, 55.0)


# ---------------------------------------------------------------------------
# add_watched_aoi
# ---------------------------------------------------------------------------

class TestAddWatchedAOI:
    def test_returns_integer_id(self):
        aoi_id = scheduler.add_watched_aoi(
            name="Test AOI", bbox=SAMPLE_BBOX, source="modis"
        )
        assert isinstance(aoi_id, int)
        assert aoi_id >= 1

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            scheduler.add_watched_aoi(name="Bad", bbox=SAMPLE_BBOX, source="invalid")

    def test_default_source_is_modis(self):
        aoi_id = scheduler.add_watched_aoi(name="Default", bbox=SAMPLE_BBOX)
        aois = scheduler.list_watched_aois()
        aoi = next(a for a in aois if a["id"] == aoi_id)
        assert aoi["source"] == "modis"

    def test_custom_fields_stored(self):
        aoi_id = scheduler.add_watched_aoi(
            name="S2 Watch",
            bbox=SAMPLE_BBOX,
            source="sentinel2",
            max_cloud_pct=20.0,
            bands=["B04", "B08", "B03"],
            threshold=30,
            notes="Test note",
        )
        aois = scheduler.list_watched_aois()
        aoi = next(a for a in aois if a["id"] == aoi_id)
        assert aoi["source"] == "sentinel2"
        assert abs(aoi["max_cloud_pct"] - 20.0) < 0.01
        assert aoi["threshold"] == 30
        assert aoi["notes"] == "Test note"
        assert "B03" in aoi["bands"]


# ---------------------------------------------------------------------------
# list_watched_aois
# ---------------------------------------------------------------------------

class TestListWatchedAOIs:
    def test_empty_initially(self):
        assert scheduler.list_watched_aois() == []

    def test_returns_all_by_default(self):
        scheduler.add_watched_aoi("A1", SAMPLE_BBOX)
        scheduler.add_watched_aoi("A2", SAMPLE_BBOX)
        assert len(scheduler.list_watched_aois()) == 2

    def test_enabled_only_filter(self):
        id1 = scheduler.add_watched_aoi("Enabled",  SAMPLE_BBOX)
        id2 = scheduler.add_watched_aoi("Disabled", SAMPLE_BBOX)
        scheduler.disable_aoi(id2)
        active = scheduler.list_watched_aois(enabled_only=True)
        ids = [a["id"] for a in active]
        assert id1 in ids
        assert id2 not in ids


# ---------------------------------------------------------------------------
# enable / disable / delete
# ---------------------------------------------------------------------------

class TestAOILifecycle:
    def test_disable_and_enable(self):
        aoi_id = scheduler.add_watched_aoi("Toggle", SAMPLE_BBOX)
        scheduler.disable_aoi(aoi_id)
        aois = scheduler.list_watched_aois()
        aoi = next(a for a in aois if a["id"] == aoi_id)
        assert aoi["enabled"] == 0

        scheduler.enable_aoi(aoi_id)
        aois = scheduler.list_watched_aois()
        aoi = next(a for a in aois if a["id"] == aoi_id)
        assert aoi["enabled"] == 1

    def test_delete_existing(self):
        aoi_id = scheduler.add_watched_aoi("Delete me", SAMPLE_BBOX)
        assert scheduler.delete_aoi(aoi_id) is True
        ids = [a["id"] for a in scheduler.list_watched_aois()]
        assert aoi_id not in ids

    def test_delete_nonexistent(self):
        assert scheduler.delete_aoi(99999) is False

    def test_enabled_flag_default_is_true(self):
        aoi_id = scheduler.add_watched_aoi("New", SAMPLE_BBOX)
        aoi = next(a for a in scheduler.list_watched_aois() if a["id"] == aoi_id)
        assert aoi["enabled"] == 1

    def test_created_at_is_set(self):
        aoi_id = scheduler.add_watched_aoi("Timestamped", SAMPLE_BBOX)
        aoi = next(a for a in scheduler.list_watched_aois() if a["id"] == aoi_id)
        assert aoi["created_at"] is not None
        assert "T" in aoi["created_at"]


# ---------------------------------------------------------------------------
# poll_once with no AOIs
# ---------------------------------------------------------------------------

class TestPollOnce:
    def test_poll_no_aois_returns_empty(self):
        results = scheduler.poll_once(verbose=False)
        assert results == []

    def test_poll_disabled_aois_skipped(self):
        aoi_id = scheduler.add_watched_aoi("Paused", SAMPLE_BBOX)
        scheduler.disable_aoi(aoi_id)
        results = scheduler.poll_once(verbose=False)
        assert results == []
