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
) -> tuple[np.ndarray, dict]:
    """
    Detect changed pixels between *before* and *after* images.

    Pipeline
    --------
    1. Convert both images to grayscale.
    2. Compute per-pixel absolute difference.
    3. Apply a 5×5 Gaussian blur to suppress high-frequency sensor noise.
    4. Binary-threshold the blurred difference at *threshold*.
    5. Morphological opening (3×3 ellipse, 2 iterations) removes isolated
       salt-and-pepper specks while preserving region shapes.
    6. Extract external contours; discard any whose bounding area < *min_area*.
    7. Render the final binary mask from kept contours.

    Parameters
    ----------
    before, after : uint8 BGR arrays of identical spatial dimensions.
    threshold     : Intensity difference (0-255) above which a pixel is
                    considered changed.  Lower → more sensitive.
    min_area      : Minimum contour area (px²) to retain.  Smaller blobs
                    are discarded as sensor / compression noise.

    Returns
    -------
    mask  : uint8 binary mask (255 = changed, 0 = unchanged), shape (H, W).
    stats : dict with the following keys —
        changed_fraction  float  fraction of total pixels that changed (0-1)
        num_regions       int    number of distinct changed contours retained
        regions           list   top-10 regions by area, each a dict:
                                   x, y, w, h  (bounding-box origin + size)
                                   area_px     (contour area in pixels)
        # legacy keys kept for app.py compatibility:
        total_pixels      int
        changed_pixels    int
        change_percent    float
    """
    # --- Grayscale conversion ---
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

    # --- Absolute difference ---
    diff = cv2.absdiff(before_gray, after_gray)

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
        "changed_fraction": round(changed_px / total_px, 6) if total_px else 0.0,
        "num_regions":      len(kept),
        "regions":          top_regions,
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
