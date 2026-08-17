"""
sentinel2_fetch.py
------------------
Sentinel-2 L2A imagery acquisition via the Copernicus Data Space Ecosystem
(CDSE) OData API.

Requires a free account at https://dataspace.copernicus.eu/ and the
following environment variables (or a .env file):

    CDSE_USER     – your Copernicus email address
    CDSE_PASSWORD – your Copernicus password

If the credentials are absent the module still imports cleanly; any call
that requires authentication raises ``CopernicusAuthError`` with a
human-readable message linking to the registration page.

Public API
~~~~~~~~~~
search_sentinel2_scenes(bbox, date_from, date_to, max_cloud_pct=30, limit=5)
    Search the CDSE catalogue for Sentinel-2 L2A products whose footprint
    intersects *bbox* and whose sensing date falls within [date_from, date_to].
    Returns a list of scene metadata dicts sorted by sensing date (newest first).

download_sentinel2_scene(scene, out_dir="data", bands=None)
    Download the requested spectral bands from a Sentinel-2 L2A product as
    individual GeoTIFF files and stack them into a single multi-band GeoTIFF
    ready for NDVI analysis.

    Default bands: B04 (Red, 10 m), B08 (NIR, 10 m).
    Additional bands can be requested with *bands* (e.g. ["B02","B03","B04","B08"]).

fetch_sentinel2_pair(bbox, date_from, date_mid, date_to, ...)
    Convenience wrapper: fetches one "before" and one "after" scene and
    returns (before_path, after_path, before_meta, after_meta).

Sentinel-2 band → index mapping (in output GeoTIFF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The output file has one band per requested band in the order supplied to
*bands*.  Default output (bands=["B04","B08"]):
    Band 1 = B04 (Red)
    Band 2 = B08 (NIR)
For NDVI via compute_ndvi_diff() with the default output, pass:
    nir_band=2, red_band=1
"""

import os
import io
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CDSE endpoint constants
# ---------------------------------------------------------------------------

_CDSE_AUTH_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_CDSE_DOWNLOAD  = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

# CDSE OData collection name for Sentinel-2 L2A
_S2_COLLECTION = "SENTINEL-2"
_S2_PRODUCT_TYPE = "S2MSI2A"   # Level-2A (surface reflectance)

# Resolution folder inside the SAFE product for 10 m bands
_S2_RESOLUTION = "R10m"

# Map of commonly-used band names to their 10 m filenames inside the product
# (GSD-10 bands: B02 Blue, B03 Green, B04 Red, B08 NIR)
_S2_BAND_FILES = {
    "B02": "B02_10m.jp2",
    "B03": "B03_10m.jp2",
    "B04": "B04_10m.jp2",
    "B08": "B08_10m.jp2",
}
# 20 m bands (e.g. SWIR, vegetation red-edge) – used only when explicitly requested
_S2_BAND_FILES_20M = {
    "B05": "B05_20m.jp2",
    "B06": "B06_20m.jp2",
    "B07": "B07_20m.jp2",
    "B8A": "B8A_20m.jp2",
    "B11": "B11_20m.jp2",
    "B12": "B12_20m.jp2",
    "SCL": "SCL_20m.jp2",   # Scene Classification Layer (for cloud masking)
}

_DEFAULT_BANDS = ["B04", "B08"]   # Red + NIR → sufficient for NDVI


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CopernicusAuthError(RuntimeError):
    """Raised when CDSE credentials are missing or invalid."""


class CopernicusSearchError(RuntimeError):
    """Raised when the CDSE catalogue search fails."""


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------

def _get_credentials() -> tuple[str, str]:
    """
    Read CDSE_USER / CDSE_PASSWORD from the environment.

    Raises
    ------
    CopernicusAuthError if either variable is missing.
    """
    user = os.getenv("CDSE_USER", "").strip()
    pwd  = os.getenv("CDSE_PASSWORD", "").strip()
    if not user or not pwd:
        raise CopernicusAuthError(
            "Copernicus credentials not found.  "
            "Set CDSE_USER and CDSE_PASSWORD in your .env file.  "
            "Register for free at https://dataspace.copernicus.eu/"
        )
    return user, pwd


def _get_access_token() -> str:
    """
    Obtain a short-lived OAuth2 access token from the CDSE identity service.

    Returns
    -------
    Bearer token string.

    Raises
    ------
    CopernicusAuthError on authentication failure.
    """
    user, pwd = _get_credentials()
    resp = requests.post(
        _CDSE_AUTH_URL,
        data={
            "client_id":  "cdse-public",
            "grant_type": "password",
            "username":   user,
            "password":   pwd,
        },
        timeout=30,
    )
    if resp.status_code == 401:
        raise CopernicusAuthError(
            "Copernicus authentication failed — check CDSE_USER and CDSE_PASSWORD."
        )
    resp.raise_for_status()
    return resp.json()["access_token"]


def credentials_available() -> bool:
    """Return True if CDSE_USER and CDSE_PASSWORD are both set (non-empty)."""
    return bool(os.getenv("CDSE_USER", "").strip()
                and os.getenv("CDSE_PASSWORD", "").strip())


# ---------------------------------------------------------------------------
# OData search
# ---------------------------------------------------------------------------

def search_sentinel2_scenes(
    bbox: tuple[float, float, float, float],
    date_from: str,
    date_to: str,
    max_cloud_pct: float = 30.0,
    limit: int = 5,
) -> list[dict]:
    """
    Search the CDSE catalogue for Sentinel-2 L2A products.

    No authentication is required for the catalogue search.

    Parameters
    ----------
    bbox          : (min_lon, min_lat, max_lon, max_lat) in WGS-84 degrees.
    date_from     : Start of the sensing date window, e.g. "2024-01-01".
    date_to       : End   of the sensing date window, e.g. "2024-06-30".
    max_cloud_pct : Reject scenes with cloud cover > this value (0-100).
    limit         : Maximum number of results (sorted newest-first).

    Returns
    -------
    List of scene dicts (newest sensing date first), each with:
        id            – CDSE product UUID
        name          – product filename (e.g. S2B_MSIL2A_…)
        date          – sensing date "YYYY-MM-DD"
        cloud_pct     – cloud cover percentage (float)
        bbox          – user-requested bbox (for download)
        source        – "Copernicus CDSE"
        product_type  – "S2MSI2A"

    Raises
    ------
    CopernicusSearchError on HTTP or JSON parse error.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # OData $filter expression — intersects + date range + product type + cloud cover
    wkt_polygon = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filter_expr = (
        f"Collection/Name eq '{_S2_COLLECTION}' "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"  and att/OData.CSC.StringAttribute/Value eq '{_S2_PRODUCT_TYPE}') "
        f"and ContentDate/Start gt {date_from}T00:00:00.000Z "
        f"and ContentDate/Start lt {date_to}T23:59:59.000Z "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"  and att/OData.CSC.DoubleAttribute/Value le {max_cloud_pct:.1f}) "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}')"
    )

    params = {
        "$filter":  filter_expr,
        "$orderby": "ContentDate/Start desc",
        "$top":     limit,
        "$expand":  "Attributes",
    }

    try:
        resp = requests.get(_CDSE_ODATA_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise CopernicusSearchError(f"CDSE catalogue search failed: {exc}") from exc

    scenes = []
    for item in data.get("value", []):
        sensing_start = item.get("ContentDate", {}).get("Start", "")
        sensing_date  = sensing_start[:10] if sensing_start else date_from

        # Extract cloud cover from expanded Attributes list
        cloud = None
        for attr in item.get("Attributes", []):
            if attr.get("Name") == "cloudCover":
                cloud = attr.get("Value")
                break

        scenes.append({
            "id":           item.get("Id", ""),
            "name":         item.get("Name", ""),
            "date":         sensing_date,
            "cloud_pct":    cloud,
            "bbox":         bbox,
            "source":       "Copernicus CDSE",
            "product_type": _S2_PRODUCT_TYPE,
        })

    return scenes


# ---------------------------------------------------------------------------
# Download + band stack
# ---------------------------------------------------------------------------

def download_sentinel2_scene(
    scene: dict,
    out_dir: str = "data",
    bands: Optional[list[str]] = None,
    include_scl: bool = False,
) -> tuple[str, Optional[str]]:
    """
    Download selected spectral bands from a Sentinel-2 L2A product and write
    them as a single multi-band GeoTIFF.  Optionally extract the Scene
    Classification Layer (SCL) as a separate sidecar GeoTIFF for cloud masking.

    Requires valid CDSE_USER / CDSE_PASSWORD in the environment.

    Parameters
    ----------
    scene       : Scene dict as returned by search_sentinel2_scenes().
                  Must have key "id" (CDSE product UUID).
    out_dir     : Directory to save the output file(s).
    bands       : List of spectral band identifiers to include in the main
                  GeoTIFF, e.g. ["B04", "B08"].
                  Defaults to ["B04", "B08"] (Red + NIR, 10 m, sufficient
                  for NDVI).  SCL must NOT be listed here — use include_scl.
                  Supported 10 m: B02, B03, B04, B08.
                  Supported 20 m: B05, B06, B07, B8A, B11, B12.
    include_scl : If True, also extract the SCL band (20 m, uint8 class
                  values) into a separate sidecar file named
                  `<safe_name>_SCL.tif`.  This sidecar can be passed
                  directly to apply_scl_mask() for cloud masking.

    Returns
    -------
    (band_path, scl_path)
        band_path : Path to the saved multi-band spectral GeoTIFF.
        scl_path  : Path to the SCL sidecar GeoTIFF, or None if
                    include_scl is False.

    Raises
    ------
    CopernicusAuthError if credentials are missing or authentication fails.
    ValueError          if an unsupported or "SCL" band name is in *bands*.
    RuntimeError        on download or file-write failure.

    Band order in output GeoTIFF
    ----------------------------
    Bands are written in the order specified by *bands*.
    Default (["B04","B08"]):  band 1 = Red (B04), band 2 = NIR (B08).
    For NDVI via compute_ndvi_diff(), pass nir_band=2, red_band=1.
    """
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    import zipfile
    import cv2 as _cv2

    if bands is None:
        bands = _DEFAULT_BANDS

    # SCL must not be in the spectral band list — it has a different dtype
    # and pixel semantics; it belongs in a sidecar file.
    spectral_bands = [b for b in bands if b != "SCL"]
    if len(spectral_bands) < len(bands):
        raise ValueError(
            "'SCL' must not be listed in bands — "
            "use include_scl=True to obtain the SCL sidecar separately."
        )

    all_known = {**_S2_BAND_FILES, **_S2_BAND_FILES_20M}
    for b in spectral_bands:
        if b not in all_known:
            raise ValueError(
                f"Unknown band '{b}'.  "
                f"Supported spectral: {sorted({**_S2_BAND_FILES, **{k:v for k,v in _S2_BAND_FILES_20M.items() if k != 'SCL'}}.keys())}"
            )

    token   = _get_access_token()
    prod_id = scene["id"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # CDSE zipper endpoint: downloads a product zip
    dl_url  = f"{_CDSE_DOWNLOAD}({prod_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(dl_url, headers=headers, timeout=300, stream=True)
    if resp.status_code == 401:
        raise CopernicusAuthError("CDSE download authorisation failed — token may have expired.")
    resp.raise_for_status()

    raw = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        raw.write(chunk)
    raw.seek(0)

    safe_name = scene.get("name", f"S2_{scene['date']}").replace(".SAFE", "")

    # ------------------------------------------------------------------ #
    # Extract spectral bands into the main multi-band GeoTIFF             #
    # ------------------------------------------------------------------ #
    band_arrays = []
    profile = None

    with zipfile.ZipFile(raw) as zf:
        all_names = zf.namelist()

        for band_id in spectral_bands:
            filename = all_known[band_id]
            matches  = [n for n in all_names if n.endswith(filename)]
            if not matches:
                raise RuntimeError(
                    f"Band file '{filename}' not found in product zip for {scene['name']}."
                )
            resolution_hint = _S2_RESOLUTION if band_id in _S2_BAND_FILES else "R20m"
            preferred = [m for m in matches if resolution_hint in m]
            entry     = preferred[0] if preferred else matches[0]

            with zf.open(entry) as band_file:
                band_bytes = band_file.read()

            with MemoryFile(band_bytes) as mem:
                with mem.open() as src:
                    arr = src.read(1)
                    if profile is None:
                        profile = src.profile.copy()
                        profile.update(
                            driver="GTiff",
                            count=len(spectral_bands),
                            dtype="uint16",
                            compress="lzw",
                        )
                    elif arr.shape != band_arrays[0].shape:
                        h, w = band_arrays[0].shape
                        arr = _cv2.resize(
                            arr.astype("float32"), (w, h),
                            interpolation=_cv2.INTER_AREA,
                        ).astype(arr.dtype)
                    band_arrays.append(arr)

        band_path = os.path.join(
            out_dir, f"{safe_name}_{'_'.join(spectral_bands)}.tif"
        )
        with rasterio.open(band_path, "w", **profile) as dst:
            for i, arr in enumerate(band_arrays, start=1):
                dst.write(arr, i)

        # ------------------------------------------------------------------ #
        # Optionally extract SCL sidecar                                       #
        # ------------------------------------------------------------------ #
        scl_path = None
        if include_scl:
            scl_filename = _S2_BAND_FILES_20M["SCL"]  # "SCL_20m.jp2"
            scl_matches  = [n for n in all_names if n.endswith(scl_filename)]
            if scl_matches:
                preferred_scl = [m for m in scl_matches if "R20m" in m]
                scl_entry     = preferred_scl[0] if preferred_scl else scl_matches[0]

                with zf.open(scl_entry) as scl_file:
                    scl_bytes = scl_file.read()

                with MemoryFile(scl_bytes) as mem:
                    with mem.open() as src:
                        scl_arr     = src.read(1)
                        scl_profile = src.profile.copy()
                        scl_profile.update(
                            driver="GTiff",
                            count=1,
                            dtype="uint8",
                            compress="lzw",
                        )

                scl_path = os.path.join(out_dir, f"{safe_name}_SCL.tif")
                with rasterio.open(scl_path, "w", **scl_profile) as dst:
                    dst.write(scl_arr, 1)

    return band_path, scl_path


# ---------------------------------------------------------------------------
# High-level scene-pair fetcher
# ---------------------------------------------------------------------------

def fetch_sentinel2_pair(
    bbox: tuple[float, float, float, float],
    date_from: str,
    date_mid: str,
    date_to: str,
    out_dir: str = "data",
    max_cloud_pct: float = 30.0,
    bands: Optional[list[str]] = None,
    include_scl: bool = False,
) -> tuple[str, str, dict, dict]:
    """
    Search for and download a before/after Sentinel-2 L2A image pair.

    Parameters
    ----------
    bbox          : (min_lon, min_lat, max_lon, max_lat)
    date_from     : Start of "before" window  (YYYY-MM-DD)
    date_mid      : Boundary between before/after windows
    date_to       : End of "after" window
    out_dir       : Output directory for downloaded GeoTIFFs
    max_cloud_pct : Maximum cloud cover accepted (0-100)
    bands         : Band list forwarded to download_sentinel2_scene().
                    Defaults to ["B04","B08"] (Red + NIR).
                    Do NOT include "SCL" here; use include_scl=True instead.
    include_scl   : If True, also download the SCL band as a sidecar
                    GeoTIFF for cloud masking.  The SCL path is stored as
                    ``scl_path`` on the returned meta dicts so callers can
                    pass it directly to apply_scl_mask().  Default False.

    Returns
    -------
    (before_path, after_path, before_meta, after_meta)
        before_meta["scl_path"] / after_meta["scl_path"] are set to the SCL
        GeoTIFF path when include_scl=True, otherwise None.

    Raises
    ------
    CopernicusAuthError    if credentials are missing.
    CopernicusSearchError  if catalogue search fails.
    RuntimeError           if no suitable scenes are found in either window.
    """
    before_scenes = search_sentinel2_scenes(
        bbox, date_from, date_mid, max_cloud_pct=max_cloud_pct, limit=5
    )
    after_scenes = search_sentinel2_scenes(
        bbox, date_mid, date_to, max_cloud_pct=max_cloud_pct, limit=5
    )

    if not before_scenes:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes found for the before window "
            f"({date_from} → {date_mid}) with cloud cover ≤ {max_cloud_pct}%.  "
            "Try widening the date range or increasing the cloud-cover limit."
        )
    if not after_scenes:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes found for the after window "
            f"({date_mid} → {date_to}) with cloud cover ≤ {max_cloud_pct}%.  "
            "Try widening the date range or increasing the cloud-cover limit."
        )

    before_meta = dict(before_scenes[0])
    after_meta  = dict(after_scenes[0])

    before_band_path, before_scl = download_sentinel2_scene(
        before_meta, out_dir=out_dir, bands=bands, include_scl=include_scl
    )
    after_band_path, after_scl = download_sentinel2_scene(
        after_meta, out_dir=out_dir, bands=bands, include_scl=include_scl
    )

    # Attach SCL paths to meta dicts so callers can access them without a
    # separate return value (maintaining the simple 4-tuple interface).
    before_meta["scl_path"] = before_scl
    after_meta["scl_path"]  = after_scl

    return before_band_path, after_band_path, before_meta, after_meta
