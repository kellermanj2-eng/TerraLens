"""
change_detection.py
-------------------
Computer-vision pipeline for satellite image change detection.

Public API
~~~~~~~~~~
load_and_align(before_path, after_path)
    Load both images and align the "after" image to the "before" image using
    ORB feature matching + RANSAC homography.

detect_change(before, after, threshold=40, min_area=200)
    Grayscale absolute differencing → Gaussian blur → binary threshold →
    morphological opening → contour extraction.
    Returns (mask, stats) where stats contains changed_fraction, num_regions,
    and the top-10 regions by area.

overlay(before, mask, out_path)
    Save a visualisation with changed pixels highlighted in red at 40 % blend.

save_results(mask, overlay_img, output_dir)
    Write mask + overlay to *output_dir* and return their paths.

run_pipeline(before_path, after_path, ...)
    Convenience end-to-end function used by app.py.

compute_ndvi_diff(before_path, after_path, nir_band=8, red_band=4)
    Compute per-pixel NDVI for before and after GeoTIFF images using the
    specified NIR and Red band indices (1-based, Sentinel-2 defaults).
    Returns (ndvi_before, ndvi_after, ndvi_diff, stats) where ndvi_diff is a
    float32 array in [-2, 2] and stats contains mean/std/gain/loss summaries.

compute_false_colour(path, r_band, g_band, b_band)
    Render a false-colour composite from any three bands of a multi-band
    GeoTIFF and return a uint8 RGB array suitable for display.
    Preset mappings are provided for common analysis composites
    (CIR, Urban/SWIR, Agriculture, Geology, Bathymetric/Shallow water).

apply_scl_mask(scl_path, target_shape)
    Read a Sentinel-2 Scene Classification Layer (SCL) GeoTIFF and return a
    uint8 exclusion mask (255 = pixel should be ignored, 0 = usable).
    Cloud, cloud shadow, saturated, and snow pixels are masked.

cloud_mask_fraction(cloud_mask)
    Return the fraction of pixels flagged as unusable in a cloud mask array.

CLI
~~~
    python change_detection.py --before before.tif --after after.tif \\
                               --out results/overlay.png --threshold 40
"""

import argparse
import json
import os

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load an image from *path* and return it as a uint8 BGR NumPy array.

    GeoTIFF files (.tif / .tiff) are read through rasterio so that
    multi-band rasters are handled correctly.  All other formats fall
    back to OpenCV.

    Returns a 3-channel (H, W, 3) uint8 BGR array.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            # Read first three bands; pad with zeros if fewer than three exist.
            bands = min(src.count, 3)
            data = src.read(
                list(range(1, bands + 1)),
                out_dtype="uint8",
                resampling=Resampling.bilinear,
            )  # shape: (bands, H, W)

            if bands == 1:
                # Grayscale → replicate to 3 channels
                rgb = np.stack([data[0]] * 3, axis=-1)
            else:
                # rasterio is band-first; transpose to (H, W, bands)
                rgb = np.transpose(data, (1, 2, 0))
                if bands == 2:
                    rgb = np.dstack([rgb, np.zeros_like(rgb[:, :, 0])])

        # Convert RGB → BGR for OpenCV consistency
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {path}")

    return bgr


# ---------------------------------------------------------------------------
# Alignment  (ORB feature matching + RANSAC homography)
# ---------------------------------------------------------------------------

# Minimum good matches required before attempting a homography.
# Raised to 12 so that marginal match sets (which produce garbage H matrices)
# are rejected early and fall back to the unwarped image.
_MIN_MATCH_COUNT = 12

# Determinant bounds for homography validation.
# det(H[:2,:2]) ≈ 1.0 for a pure translation/rotation.
# Values outside [0.5, 2.0] indicate extreme scale, shear, or a reflected warp
# — all signs of a degenerate RANSAC solution on low-texture imagery.
_H_DET_MIN = 0.5
_H_DET_MAX = 2.0

# Cloud-cover guard: fraction of pixels brighter than this grayscale threshold
# that triggers a warning.  Does NOT block processing.
_CLOUD_BRIGHT_THRESH   = 200   # grayscale value (0-255)
_CLOUD_FRACTION_WARN   = 0.60  # warn if > 60 % of pixels exceed threshold


def load_and_align(
    before_path: str,
    after_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load both images, optionally warp "after" onto "before", and return a
    list of any advisory warnings produced during alignment.

    Alignment strategy
    ------------------
    1. Resize "after" to the spatial dimensions of "before".
    2. Cloud-cover guard: if either grayscale image has > 60 % very-bright
       pixels (> 200), record a warning but continue.
    3. Detect ORB keypoints + descriptors in both grayscale images.
    4. Match with knnMatch (k=2) + Lowe ratio test (0.75) to discard
       ambiguous matches that cause false homographies on low-texture scenes.
    5. Require ≥ 12 good matches before attempting homography; otherwise skip
       warping entirely and return the resized-but-unwarped "after".
    6. Compute RANSAC homography H and validate its determinant:
       only warp if 0.5 < det(H[:2,:2]) < 2.0.  Values outside that range
       indicate extreme scale/shear/flip — a degenerate solution that would
       smear the image into streaks.  Fall back to unwarped on failure.

    Parameters
    ----------
    before_path : path to the earlier satellite image.
    after_path  : path to the later  satellite image.

    Returns
    -------
    (before, after_aligned, warnings)
        before        – uint8 BGR reference array.
        after_aligned – uint8 BGR array, same shape as before.
        warnings      – list of human-readable advisory strings (may be empty).
    """
    warnings: list[str] = []

    before = load_image(before_path)
    after  = load_image(after_path)

    h, w = before.shape[:2]

    # Step 1 — force identical spatial dimensions
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)

    # Step 2 — cloud-cover guard (advisory only, does not block)
    ref_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)
    total_px = ref_gray.size
    before_bright = np.count_nonzero(ref_gray > _CLOUD_BRIGHT_THRESH) / total_px
    after_bright  = np.count_nonzero(tgt_gray > _CLOUD_BRIGHT_THRESH) / total_px
    if before_bright > _CLOUD_FRACTION_WARN:
        msg = (f"Before image appears heavily cloud-covered "
               f"({before_bright:.0%} bright pixels). "
               "Change detection results may be unreliable.")
        print(f"[TerraLens WARNING] {msg}")
        warnings.append(msg)
    if after_bright > _CLOUD_FRACTION_WARN:
        msg = (f"After image appears heavily cloud-covered "
               f"({after_bright:.0%} bright pixels). "
               "Change detection results may be unreliable.")
        print(f"[TerraLens WARNING] {msg}")
        warnings.append(msg)

    # Step 3 — ORB keypoints & descriptors
    orb = cv2.ORB_create(nfeatures=5000)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_tgt, des_tgt = orb.detectAndCompute(tgt_gray, None)

    # Guard: nothing to match if either image has no descriptors
    if des_ref is None or des_tgt is None or len(kp_ref) < 2 or len(kp_tgt) < 2:
        return before, after, warnings

    # Step 4 — knnMatch (k=2) + Lowe ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des_ref, des_tgt, k=2)

    good = []
    for pair in raw_matches:
        # knnMatch may return fewer than k results for edge-case descriptors
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    # Step 5 — require ≥ 12 good matches before attempting homography
    if len(good) < _MIN_MATCH_COUNT:
        return before, after, warnings

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tgt[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Step 6 — validate H before warping
    if H is None:
        return before, after, warnings

    det = float(np.linalg.det(H[:2, :2]))
    if not (_H_DET_MIN < det < _H_DET_MAX):
        print(f"[TerraLens WARNING] Homography rejected (det={det:.4f}); "
              "using unwarped image to avoid warp artefacts.")
        warnings.append(
            f"Alignment skipped: homography determinant ({det:.3f}) is outside "
            f"the valid range [{_H_DET_MIN}, {_H_DET_MAX}]. "
            "The images are compared without warping — ensure they cover the "
            "same geographic area."
        )
        return before, after, warnings

    after_aligned = cv2.warpPerspective(after, H, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
    return before, after_aligned, warnings


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def detect_change(
    before: np.ndarray,
    after: np.ndarray,
    threshold: int = 40,
    min_area: int = 200,
    cloud_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Detect changed pixels between *before* and *after* images.

    Pipeline
    --------
    1. Convert both images to grayscale.
    2. Compute per-pixel absolute difference.
    3. If *cloud_mask* is provided, zero out masked pixels in the diff so
       cloud/shadow regions cannot trigger false-positive detections.
    4. Apply a 5×5 Gaussian blur to suppress high-frequency sensor noise.
    5. Binary-threshold the blurred difference at *threshold*.
    6. Morphological opening (3×3 ellipse, 2 iterations) removes isolated
       salt-and-pepper specks while preserving region shapes.
    7. Extract external contours; discard any whose bounding area < *min_area*.
    8. Render the final binary mask from kept contours.

    Parameters
    ----------
    before, after : uint8 BGR arrays of identical spatial dimensions.
    threshold     : Intensity difference (0-255) above which a pixel is
                    considered changed.  Lower → more sensitive.
    min_area      : Minimum contour area (px²) to retain.  Smaller blobs
                    are discarded as sensor / compression noise.
    cloud_mask    : Optional uint8 exclusion mask (255 = masked / unusable,
                    0 = clear), same spatial shape as *before*.  Typically
                    produced by apply_scl_mask().  Masked pixels are zeroed
                    in the difference image before thresholding, preventing
                    clouds and shadows from being flagged as real changes.

    Returns
    -------
    mask  : uint8 binary mask (255 = changed, 0 = unchanged), shape (H, W).
    stats : dict with the following keys —
        changed_fraction  float  fraction of total pixels that changed (0-1)
        num_regions       int    number of distinct changed contours retained
        regions           list   top-10 regions by area, each a dict:
                                   x, y, w, h  (bounding-box origin + size)
                                   area_px     (contour area in pixels)
        cloud_masked_pct  float  percentage of pixels excluded by cloud mask
                                 (0.0 if no mask was supplied)
        # legacy keys kept for app.py / narrate.py compatibility:
        total_pixels      int
        changed_pixels    int
        change_percent    float
    """
    # --- Grayscale conversion ---
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

    # --- Absolute difference ---
    diff = cv2.absdiff(before_gray, after_gray)

    # --- Cloud mask suppression (zero out unreliable pixels before threshold) ---
    cloud_masked_pct = 0.0
    if cloud_mask is not None:
        cm = cloud_mask
        # Resize mask to match diff if SCL was resampled at a different resolution
        if cm.shape != diff.shape:
            cm = cv2.resize(cm, (diff.shape[1], diff.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        diff[cm == 255] = 0
        cloud_masked_pct = round(
            float(np.count_nonzero(cm == 255)) / cm.size * 100, 2
        )

    # --- Gaussian blur (reduce sensor noise before thresholding) ---
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # --- Binary threshold ---
    _, binary = cv2.threshold(diff_blur, threshold, 255, cv2.THRESH_BINARY)

    # --- Morphological opening (removes speckle, preserves blobs) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Contour extraction ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Filter by minimum area ---
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]

    # --- Render clean mask from kept contours only ---
    mask = np.zeros(before_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, kept, -1, 255, thickness=cv2.FILLED)

    # --- Statistics ---
    total_px   = int(mask.size)
    changed_px = int(np.count_nonzero(mask))

    # Top-10 regions sorted by area (largest first)
    sorted_contours = sorted(kept, key=cv2.contourArea, reverse=True)
    top_regions = []
    for c in sorted_contours[:10]:
        x, y, bw, bh = cv2.boundingRect(c)
        top_regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "area_px": int(cv2.contourArea(c)),
        })

    stats = {
        # primary keys (requested spec)
        "changed_fraction":  round(changed_px / total_px, 6) if total_px else 0.0,
        "num_regions":       len(kept),
        "regions":           top_regions,
        # cloud-masking report
        "cloud_masked_pct":  cloud_masked_pct,
        # legacy keys (used by app.py / narrate.py)
        "total_pixels":    total_px,
        "changed_pixels":  changed_px,
        "change_percent":  round(changed_px / total_px * 100, 2) if total_px else 0.0,
        # real-world area — filled in by the caller if GSD is known, else None
        "changed_km2":     None,
    }

    return mask, stats


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------

def regions_to_geojson(
    regions: list[dict],
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> dict:
    """
    Convert pixel-space region bounding boxes to a GeoJSON FeatureCollection.

    Each region becomes a GeoJSON Polygon feature whose coordinates are the
    WGS-84 lon/lat corners derived by interpolating the pixel position within
    the image's geographic bounding box.

    Parameters
    ----------
    regions     : list of region dicts from detect_change() stats["regions"].
                  Each must have keys x, y, w, h (pixel bounding box).
    bbox        : (min_lon, min_lat, max_lon, max_lat) geographic extent of
                  the image in WGS-84 degrees.
    image_shape : (height, width) in pixels — before.shape[:2].

    Returns
    -------
    GeoJSON FeatureCollection dict (JSON-serialisable).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    img_h, img_w = image_shape

    def px_to_lonlat(px_x: float, px_y: float) -> list[float]:
        lon = min_lon + (px_x / img_w) * (max_lon - min_lon)
        # pixel row 0 = top = max_lat; row img_h = bottom = min_lat
        lat = max_lat - (px_y / img_h) * (max_lat - min_lat)
        return [round(lon, 6), round(lat, 6)]

    features = []
    for i, r in enumerate(regions):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        # Clockwise ring: top-left → top-right → bottom-right → bottom-left → close
        ring = [
            px_to_lonlat(x,     y),
            px_to_lonlat(x + w, y),
            px_to_lonlat(x + w, y + h),
            px_to_lonlat(x,     y + h),
            px_to_lonlat(x,     y),      # closed
        ]
        features.append({
            "type": "Feature",
            "id": i,
            "properties": {
                "rank":    i + 1,
                "area_px": r["area_px"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Sentinel-2 cloud masking  (SCL — Scene Classification Layer)
# ---------------------------------------------------------------------------

# SCL pixel class values that should be excluded from change detection.
# Reference: Sentinel-2 Level-2A Algorithm Theoretical Basis Document (ATBD)
#   0  NO_DATA
#   1  SATURATED_OR_DEFECTIVE
#   3  CLOUD_SHADOWS
#   8  CLOUD_MEDIUM_PROBABILITY
#   9  CLOUD_HIGH_PROBABILITY
#  10  THIN_CIRRUS
#  11  SNOW_ICE  (optional — keep False to preserve snow-change detection)
_SCL_CLOUD_CLASSES = frozenset([0, 1, 3, 8, 9, 10])
_SCL_SNOW_CLASS    = 11   # masked separately so callers can opt out


def apply_scl_mask(
    scl_path: str,
    target_shape: tuple[int, int],
    mask_snow: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Read a Sentinel-2 SCL (Scene Classification Layer) GeoTIFF and produce a
    binary exclusion mask for use with detect_change().

    SCL classes masked unconditionally (cloud_classes):
        0  NO_DATA
        1  SATURATED / DEFECTIVE
        3  CLOUD SHADOWS
        8  CLOUD MEDIUM PROBABILITY
        9  CLOUD HIGH PROBABILITY
       10  THIN CIRRUS

    SCL class masked when *mask_snow* is True:
       11  SNOW / ICE

    Parameters
    ----------
    scl_path     : Path to the SCL GeoTIFF (single-band, uint8 class values).
    target_shape : (height, width) of the image the mask will be applied to.
                   The SCL band (20 m) is resampled to this resolution with
                   nearest-neighbour interpolation if necessary.
    mask_snow    : If True, also mask class 11 (snow/ice).  Default False —
                   snow cover changes are often real events worth detecting.

    Returns
    -------
    cloud_mask : uint8 array, shape *target_shape*.
                 255 = pixel excluded (cloud / shadow / no-data / snow).
                 0   = pixel usable (clear land, water, vegetation).
    scl_stats  : dict with keys —
        cloud_fraction  float  fraction of pixels in cloud_classes (0-1)
        snow_fraction   float  fraction of class-11 pixels (0-1)
        masked_fraction float  total fraction masked (includes snow if mask_snow)
        masked_pct      float  masked_fraction × 100 (convenience)

    Raises
    ------
    FileNotFoundError if *scl_path* does not exist.
    ValueError        if the file cannot be opened as a single-band raster.
    """
    if not os.path.exists(scl_path):
        raise FileNotFoundError(f"SCL file not found: {scl_path}")

    with rasterio.open(scl_path) as src:
        scl = src.read(1)   # uint8 class values, shape (H_scl, W_scl)

    h, w = target_shape

    # Resample to target resolution (nearest-neighbour preserves class integers)
    if scl.shape != (h, w):
        scl = cv2.resize(scl.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)

    total = scl.size

    # Build boolean masks for each group
    cloud_bool = np.isin(scl, list(_SCL_CLOUD_CLASSES))
    snow_bool  = (scl == _SCL_SNOW_CLASS)

    exclude_bool = cloud_bool | (snow_bool if mask_snow else np.zeros_like(cloud_bool))

    cloud_mask = np.where(exclude_bool, np.uint8(255), np.uint8(0))

    scl_stats = {
        "cloud_fraction":  round(float(cloud_bool.sum()) / total, 6),
        "snow_fraction":   round(float(snow_bool.sum())  / total, 6),
        "masked_fraction": round(float(exclude_bool.sum()) / total, 6),
        "masked_pct":      round(float(exclude_bool.sum()) / total * 100, 2),
    }

    return cloud_mask, scl_stats


def cloud_mask_fraction(cloud_mask: np.ndarray) -> float:
    """
    Return the fraction of pixels set to 255 (excluded) in *cloud_mask*.

    Parameters
    ----------
    cloud_mask : uint8 array as returned by apply_scl_mask().

    Returns
    -------
    float in [0, 1].
    """
    return float(np.count_nonzero(cloud_mask == 255)) / cloud_mask.size


# ---------------------------------------------------------------------------
# NDVI differencing  (Sentinel-2 NIR / Red bands)
# ---------------------------------------------------------------------------

def _read_band_float(path: str, band_index: int) -> np.ndarray:
    """
    Read a single band from a GeoTIFF as a float32 array (H, W).

    Parameters
    ----------
    path       : path to the GeoTIFF.
    band_index : 1-based band number (Sentinel-2 convention:
                 band 4 = Red, band 8 = NIR).

    Raises
    ------
    ValueError  if the file has fewer bands than *band_index*.
    """
    with rasterio.open(path) as src:
        if band_index > src.count:
            raise ValueError(
                f"Band {band_index} requested but '{os.path.basename(path)}' "
                f"only has {src.count} band(s).  "
                "NDVI requires a multi-band GeoTIFF with NIR and Red bands "
                "(Sentinel-2 band 8 = NIR, band 4 = Red)."
            )
        data = src.read(band_index).astype(np.float32)
    return data


def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute NDVI = (NIR − Red) / (NIR + Red).

    Pixels where NIR + Red == 0 receive NDVI = 0 to avoid division errors.
    Output is clipped to [-1, 1] and returned as float32.
    """
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom != 0, (nir - red) / denom, 0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def compute_ndvi_diff(
    before_path: str,
    after_path: str,
    nir_band: int = 8,
    red_band: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute NDVI for *before* and *after* GeoTIFF images and return the
    signed per-pixel difference (after − before).

    Default band assignments follow the Sentinel-2 L2A convention:
        band 4 = Red (664 nm)
        band 8 = NIR (833 nm)
    Adjust *nir_band* / *red_band* for other sensors (e.g. Landsat 8:
    nir_band=5, red_band=4).

    Parameters
    ----------
    before_path : path to the earlier GeoTIFF (must have ≥ max(nir_band, red_band) bands).
    after_path  : path to the later  GeoTIFF.
    nir_band    : 1-based index of the Near-Infrared band.
    red_band    : 1-based index of the Red band.

    Returns
    -------
    ndvi_before : float32 array (H, W), NDVI of the earlier image.
    ndvi_after  : float32 array (H, W), NDVI of the later image.
    ndvi_diff   : float32 array (H, W), signed change (after − before), range [-2, 2].
    stats       : dict with summary keys —
        mean_ndvi_before  float  scene-wide mean NDVI before
        mean_ndvi_after   float  scene-wide mean NDVI after
        mean_ndvi_diff    float  mean signed change
        std_ndvi_diff     float  standard deviation of the change
        gain_fraction     float  fraction of pixels with NDVI increase > +0.05
        loss_fraction     float  fraction of pixels with NDVI decrease < -0.05
        gain_area_pct     float  gain_fraction as percentage (convenience)
        loss_area_pct     float  loss_fraction as percentage (convenience)

    Raises
    ------
    ValueError  if either file is not a GeoTIFF or lacks the required bands.
    """
    for path in (before_path, after_path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".tif", ".tiff"):
            raise ValueError(
                f"NDVI differencing requires GeoTIFF input; got '{os.path.basename(path)}'."
            )

    nir_b = _read_band_float(before_path, nir_band)
    red_b = _read_band_float(before_path, red_band)
    nir_a = _read_band_float(after_path,  nir_band)
    red_a = _read_band_float(after_path,  red_band)

    # Resize after-bands to match before if necessary (mirrors load_and_align logic)
    if nir_a.shape != nir_b.shape:
        h, w = nir_b.shape
        nir_a = cv2.resize(nir_a, (w, h), interpolation=cv2.INTER_AREA)
        red_a = cv2.resize(red_a, (w, h), interpolation=cv2.INTER_AREA)

    ndvi_before = _ndvi(nir_b, red_b)
    ndvi_after  = _ndvi(nir_a, red_a)
    ndvi_diff   = (ndvi_after - ndvi_before).astype(np.float32)

    total = ndvi_diff.size
    gain  = int(np.count_nonzero(ndvi_diff >  0.05))
    loss  = int(np.count_nonzero(ndvi_diff < -0.05))

    stats = {
        "mean_ndvi_before": round(float(np.nanmean(ndvi_before)), 4),
        "mean_ndvi_after":  round(float(np.nanmean(ndvi_after)),  4),
        "mean_ndvi_diff":   round(float(np.nanmean(ndvi_diff)),   4),
        "std_ndvi_diff":    round(float(np.nanstd(ndvi_diff)),    4),
        "gain_fraction":    round(gain / total, 6) if total else 0.0,
        "loss_fraction":    round(loss / total, 6) if total else 0.0,
        "gain_area_pct":    round(gain / total * 100, 2) if total else 0.0,
        "loss_area_pct":    round(loss / total * 100, 2) if total else 0.0,
    }

    return ndvi_before, ndvi_after, ndvi_diff, stats


# ---------------------------------------------------------------------------
# False-colour composites
# ---------------------------------------------------------------------------

# Preset composite definitions: name → (r_band, g_band, b_band, description)
# Band indices are 1-based Sentinel-2 L2A defaults; the UI lets users override.
FALSE_COLOUR_PRESETS: dict[str, tuple[int, int, int, str]] = {
    "CIR (Colour Infrared — vegetation)":  (8, 4, 3, "NIR→R, Red→G, Green→B. Healthy vegetation appears vivid red. Ideal for detecting deforestation, crop stress, and fire scars."),
    "Urban / SWIR":                         (12, 11, 4, "SWIR-2→R, SWIR-1→G, Red→B. Built-up areas show up in bright cyan/white; bare soil is brown; vegetation is green."),
    "Agriculture":                          (11, 8, 4, "SWIR-1→R, NIR→G, Red→B. Crops appear bright green; fallow land is brown; water is dark blue."),
    "Geology / Bare soil":                  (12, 11, 2, "SWIR-2→R, SWIR-1→G, Blue→B. Highlights exposed rock, sand, and soil types by mineral absorption differences."),
    "Bathymetric / Shallow water":          (4, 3, 1, "Red→R, Green→G, Coastal aerosol→B. Enhances water depth gradients and coastal sediment in shallow seas."),
}


def compute_false_colour(
    path: str,
    r_band: int,
    g_band: int,
    b_band: int,
) -> np.ndarray:
    """
    Build a false-colour RGB composite from three bands of a multi-band GeoTIFF.

    Each band is read, normalised to the 2nd–98th percentile range of that
    band's pixel distribution (robust stretch), and mapped to [0, 255] uint8.
    This matches the display behaviour of tools like QGIS and EO Browser.

    Parameters
    ----------
    path   : Path to the multi-band GeoTIFF (e.g. a Sentinel-2 L2A stack).
    r_band : 1-based band index assigned to the Red channel.
    g_band : 1-based band index assigned to the Green channel.
    b_band : 1-based band index assigned to the Blue channel.

    Returns
    -------
    rgb : uint8 ndarray of shape (H, W, 3) in RGB order, ready for
          ``st.image()`` or ``cv2.imwrite()``.

    Raises
    ------
    FileNotFoundError  if *path* does not exist.
    ValueError         if the file has fewer bands than the highest requested index,
                       or if the file is not a valid GeoTIFF.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with rasterio.open(path) as src:
        n_bands = src.count
        needed  = max(r_band, g_band, b_band)
        if needed > n_bands:
            raise ValueError(
                f"Composite requires band {needed} but '{os.path.basename(path)}' "
                f"only has {n_bands} band(s).  "
                "Choose a preset that fits your file's band count, or use a "
                "Sentinel-2 L2A stack with all 12 reflectance bands."
            )

        def _read_norm(band_idx: int) -> np.ndarray:
            """Read a band and apply a 2–98 percentile robust linear stretch."""
            data = src.read(band_idx).astype(np.float32)
            lo, hi = np.percentile(data[data > 0], [2, 98]) if np.any(data > 0) else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            normed = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
            return (normed * 255).astype(np.uint8)

        r = _read_norm(r_band)
        g = _read_norm(g_band)
        b = _read_norm(b_band)

    # Resize g and b to r dimensions if they differ (rare — different resolution bands)
    if g.shape != r.shape:
        g = cv2.resize(g, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
    if b.shape != r.shape:
        b = cv2.resize(b, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)

    return np.dstack([r, g, b])  # (H, W, 3) uint8 RGB


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay(
    before: np.ndarray,
    mask: np.ndarray,
    out_path: str,
) -> np.ndarray:
    """
    Create and save a visualisation of changed regions.

    Changed pixels (mask == 255) are coloured solid red, then blended with
    the original "before" image at 40 % opacity so underlying detail is
    still visible.

    Parameters
    ----------
    before   : uint8 BGR array — the reference (earlier) image.
    mask     : uint8 binary mask from detect_change().
    out_path : file path where the PNG will be written.
               Parent directory is created automatically.

    Returns
    -------
    overlay_img : uint8 BGR array of the rendered visualisation.
    """
    # Build a full-red layer, then zero out unchanged pixels
    red_layer = np.zeros_like(before)
    red_layer[mask == 255] = (0, 0, 255)   # BGR red

    # 60 % original + 40 % red highlight
    overlay_img = cv2.addWeighted(before, 0.6, red_layer, 0.4, 0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, overlay_img)

    return overlay_img


# ---------------------------------------------------------------------------
# Save results (mask + overlay together)
# ---------------------------------------------------------------------------

def save_results(
    mask: np.ndarray,
    overlay_img: np.ndarray,
    output_dir: str = "results",
) -> tuple[str, str]:
    """
    Write the binary *mask* and colour *overlay_img* to *output_dir*.

    Returns (mask_path, overlay_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_path    = os.path.join(output_dir, "change_mask.png")
    overlay_path = os.path.join(output_dir, "change_overlay.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay_img)
    return mask_path, overlay_path


# ---------------------------------------------------------------------------
# Convenience end-to-end pipeline (used by app.py)
# ---------------------------------------------------------------------------

def run_pipeline(
    before_path: str,
    after_path: str,
    threshold: int = 40,
    min_area: int = 200,
    output_dir: str = "results",
) -> dict:
    """
    End-to-end change detection pipeline.

    Parameters
    ----------
    before_path : path to the "before" satellite image.
    after_path  : path to the "after"  satellite image.
    threshold   : pixel-level difference threshold passed to detect_change().
    min_area    : minimum contour area passed to detect_change().
    output_dir  : directory where mask + overlay are saved.

    Returns
    -------
    dict with keys:
      before        – loaded BGR array (reference)
      after         – aligned BGR array
      mask          – binary change mask
      overlay       – annotated BGR image
      mask_path     – saved mask file path
      overlay_path  – saved overlay file path
      stats         – change statistics dict
    """
    before, after_aligned, warnings = load_and_align(before_path, after_path)
    mask, stats = detect_change(before, after_aligned, threshold, min_area)

    overlay_path = os.path.join(output_dir, "change_overlay.png")
    overlay_img  = overlay(before, mask, overlay_path)
    mask_path, _ = save_results(mask, overlay_img, output_dir)

    return {
        "before":       before,
        "after":        after_aligned,
        "mask":         mask,
        "overlay":      overlay_img,
        "mask_path":    mask_path,
        "overlay_path": overlay_path,
        "stats":        stats,
        "warnings":     warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="change_detection",
        description="TerraLens — satellite image change detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--before",    required=True,  help="Path to the 'before' image.")
    p.add_argument("--after",     required=True,  help="Path to the 'after' image.")
    p.add_argument("--out",       default="results/change_overlay.png",
                   help="Output path for the change overlay PNG.")
    p.add_argument("--threshold", type=int, default=40,
                   help="Pixel difference threshold (0-255).")
    p.add_argument("--min-area",  type=int, default=200,
                   help="Minimum contour area in px² to retain.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    print(f"[TerraLens] Loading and aligning images…")
    before, after_aligned, warnings = load_and_align(args.before, args.after)
    for w in warnings:
        print(f"[TerraLens WARNING] {w}")

    print(f"[TerraLens] Detecting changes (threshold={args.threshold}, "
          f"min_area={args.min_area})…")
    mask, stats = detect_change(before, after_aligned, args.threshold, args.min_area)

    print(f"[TerraLens] Saving overlay → {args.out}")
    overlay(before, mask, args.out)

    # Print stats to stdout as formatted JSON
    print("\n── Change Statistics ──────────────────────────")
    output = {
        "changed_fraction": stats["changed_fraction"],
        "change_percent":   stats["change_percent"],
        "changed_pixels":   stats["changed_pixels"],
        "total_pixels":     stats["total_pixels"],
        "num_regions":      stats["num_regions"],
        "top_regions":      stats["regions"],
    }
    print(json.dumps(output, indent=2))
