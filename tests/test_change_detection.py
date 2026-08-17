"""
tests/test_change_detection.py
-------------------------------
Unit tests for change_detection.py — the core CV pipeline.

All tests are fully offline (no network, no real satellite imagery).
Synthetic numpy arrays are used so the suite runs in < 2 seconds.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from change_detection import (
    detect_change,
    overlay,
    regions_to_geojson,
    _ndvi,
    compute_ndvi_diff,
    compute_false_colour,
    FALSE_COLOUR_PRESETS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_pair():
    """Two identical 200×300 BGR images — should produce zero change."""
    img = np.full((200, 300, 3), 80, dtype=np.uint8)
    return img.copy(), img.copy()


@pytest.fixture
def changed_pair():
    """Before/after pair with a clearly changed rectangle in the top-right."""
    before = np.full((200, 300, 3), 50, dtype=np.uint8)
    after  = before.copy()
    after[20:80, 200:280] = 200   # bright patch — large change
    return before, after


# ---------------------------------------------------------------------------
# detect_change
# ---------------------------------------------------------------------------

class TestDetectChange:
    def test_no_change_returns_zero_fraction(self, blank_pair):
        before, after = blank_pair
        mask, stats = detect_change(before, after, threshold=40)
        assert stats["changed_fraction"] == 0.0
        assert stats["num_regions"] == 0
        assert mask.sum() == 0

    def test_changed_pair_detects_region(self, changed_pair):
        before, after = changed_pair
        mask, stats = detect_change(before, after, threshold=40)
        assert stats["change_percent"] > 5.0, "Expected meaningful change"
        assert stats["num_regions"] >= 1
        assert stats["changed_pixels"] > 0

    def test_stats_keys_present(self, changed_pair):
        before, after = changed_pair
        _, stats = detect_change(before, after)
        for key in ("changed_fraction", "num_regions", "regions",
                    "total_pixels", "changed_pixels", "change_percent",
                    "cloud_masked_pct"):
            assert key in stats, f"Missing key: {key}"

    def test_cloud_mask_suppresses_change(self, changed_pair):
        before, after = changed_pair
        # Mask covering the entire changed region
        cloud_mask = np.zeros((200, 300), dtype=np.uint8)
        cloud_mask[20:80, 200:280] = 255
        mask_no_cm, stats_no_cm = detect_change(before, after, threshold=40)
        mask_cm,    stats_cm    = detect_change(before, after, threshold=40,
                                                 cloud_mask=cloud_mask)
        assert stats_cm["changed_fraction"] < stats_no_cm["changed_fraction"]
        assert stats_cm["cloud_masked_pct"] > 0.0

    def test_threshold_sensitivity(self, changed_pair):
        before, after = changed_pair
        _, stats_low  = detect_change(before, after, threshold=10)
        _, stats_high = detect_change(before, after, threshold=90)
        assert stats_low["changed_fraction"] >= stats_high["changed_fraction"]

    def test_mask_is_binary(self, changed_pair):
        before, after = changed_pair
        mask, _ = detect_change(before, after)
        unique = set(mask.flatten().tolist())
        assert unique <= {0, 255}, f"Mask has non-binary values: {unique - {0,255}}"

    def test_total_pixels_equals_image_size(self, changed_pair):
        before, after = changed_pair
        _, stats = detect_change(before, after)
        assert stats["total_pixels"] == before.shape[0] * before.shape[1]


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

class TestOverlay:
    def test_overlay_returns_bgr_array(self, changed_pair, tmp_path):
        before, after = changed_pair
        mask, _ = detect_change(before, after)
        out_path = str(tmp_path / "overlay.png")
        result = overlay(before, mask, out_path)
        assert result.shape == before.shape
        assert result.dtype == np.uint8
        assert os.path.exists(out_path)

    def test_overlay_unchanged_pixels_close_to_before(self, blank_pair, tmp_path):
        before, _ = blank_pair
        mask = np.zeros((200, 300), dtype=np.uint8)
        result = overlay(before, mask, str(tmp_path / "o.png"))
        # overlay = addWeighted(before, 0.6, zeros, 0.4, 0) = before * 0.6
        # so result should be roughly 60% of before (no red added)
        expected = (before.astype(float) * 0.6).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# regions_to_geojson
# ---------------------------------------------------------------------------

class TestRegionsToGeoJSON:
    def test_empty_regions(self):
        gj = regions_to_geojson([], bbox=(-10, 35, 10, 55), image_shape=(200, 300))
        assert gj["type"] == "FeatureCollection"
        assert gj["features"] == []

    def test_single_region_produces_feature(self):
        regions = [{"x": 10, "y": 20, "w": 50, "h": 40, "area_px": 2000}]
        gj = regions_to_geojson(regions, bbox=(-10, 35, 10, 55), image_shape=(200, 300))
        assert len(gj["features"]) == 1
        feat = gj["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        assert feat["properties"]["area_px"] == 2000


# ---------------------------------------------------------------------------
# NDVI helpers
# ---------------------------------------------------------------------------

class TestNDVI:
    def test_ndvi_pure_vegetation(self):
        nir = np.full((10, 10), 0.8, dtype=np.float32)
        red = np.full((10, 10), 0.1, dtype=np.float32)
        result = _ndvi(nir, red)
        # (0.8-0.1)/(0.8+0.1) ≈ 0.778
        assert np.allclose(result, 0.7 / 0.9, atol=0.01)

    def test_ndvi_zero_denominator_is_zero(self):
        nir = np.zeros((5, 5), dtype=np.float32)
        red = np.zeros((5, 5), dtype=np.float32)
        result = _ndvi(nir, red)
        assert (result == 0).all()

    def test_ndvi_clipped_to_minus_one_one(self):
        nir = np.full((5, 5), 10.0, dtype=np.float32)
        red = np.full((5, 5), 0.001, dtype=np.float32)
        result = _ndvi(nir, red)
        assert result.max() <= 1.0
        assert result.min() >= -1.0


# ---------------------------------------------------------------------------
# False-colour composites
# ---------------------------------------------------------------------------

class TestFalseColour:
    def test_presets_dict_nonempty(self):
        assert len(FALSE_COLOUR_PRESETS) >= 5

    def test_compute_false_colour_returns_rgb(self, tmp_path):
        # Write a tiny 3-band GeoTIFF
        import rasterio
        from rasterio.transform import from_bounds
        tif = str(tmp_path / "test.tif")
        data = (np.random.rand(3, 20, 30) * 3000).astype(np.uint16)
        transform = from_bounds(0, 0, 1, 1, 30, 20)
        with rasterio.open(tif, "w", driver="GTiff", height=20, width=30,
                           count=3, dtype="uint16", transform=transform,
                           crs="EPSG:4326") as dst:
            dst.write(data)
        result = compute_false_colour(tif, r_band=1, g_band=2, b_band=3)
        assert result.shape == (20, 30, 3)
        assert result.dtype == np.uint8

    def test_compute_false_colour_missing_band_raises(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        tif = str(tmp_path / "one_band.tif")
        data = (np.random.rand(1, 10, 10) * 1000).astype(np.uint16)
        transform = from_bounds(0, 0, 1, 1, 10, 10)
        with rasterio.open(tif, "w", driver="GTiff", height=10, width=10,
                           count=1, dtype="uint16", transform=transform,
                           crs="EPSG:4326") as dst:
            dst.write(data)
        with pytest.raises(ValueError, match="band"):
            compute_false_colour(tif, r_band=1, g_band=2, b_band=3)

    def test_compute_false_colour_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            compute_false_colour("/nonexistent/path.tif", 1, 2, 3)
