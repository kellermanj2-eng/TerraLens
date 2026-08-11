# Sample Data for TerraLens

This folder holds the before/after satellite image pairs used to test the app.
It is intentionally empty in the repository (only a `.gitkeep` is tracked) to
avoid distributing large binary files.

Follow the steps below to download a free wildfire or flood image pair from
**NASA Worldview** — no account or registration required.

---

## Option A — Wildfire (recommended for judges)

The 2021 Dixie Fire in northern California is a high-contrast, well-documented
event that produces a clear change mask.

### Step-by-step

1. Open **NASA Worldview**: <https://worldview.earthdata.nasa.gov/>

2. In the **Layers** panel (left sidebar), confirm that
   **MODIS Terra / Aqua True Color** is active, or add
   **Landsat 8/9 OLI TIRS True Color** for higher resolution.

3. Set the **date** (bottom date bar) to **2021-07-13** (before the main burn).
   - Click the download icon (⬇) in the toolbar → **Download Image**.
   - In the dialog: choose **JPEG** or **PNG**, set resolution to **250 m/px**
     or finer, confirm the bounding box covers the Plumas / Butte County area
     (~39.9°N 121.4°W), then click **Download**.
   - Save the file as `data/before.jpg`.

4. Change the date to **2021-08-20** (after the peak burn spread).
   - Repeat the download step above.
   - Save the file as `data/after.jpg`.

5. Run the app:
   ```bash
   streamlit run app.py
   ```
   Upload `data/before.jpg` as **Before** and `data/after.jpg` as **After**.
   Set the date range to `July 13 – August 20, 2021` in the sidebar.

> **Expected result:** ~15–40% scene change, with large contiguous red regions
> in the overlay corresponding to the burn scar.

---

## Option B — Flood (Bangladesh / Pakistan monsoon events)

1. Open **NASA Worldview**: <https://worldview.earthdata.nasa.gov/>

2. Add the layer **MODIS Terra True Color** (or Landsat for finer detail).

3. Navigate to central Bangladesh (~23.5°N 90.4°E).

4. Download a **before** image dated **2022-06-01** → save as `data/before.jpg`.

5. Download an **after** image dated **2022-06-22** (peak 2022 flood extent)
   → save as `data/after.jpg`.

6. Run the app and set date range to `June 1 – June 22, 2022`.

> **Expected result:** standing water appears as dark uniform patches; the
> overlay highlights inundated agricultural land.

---

## Tips

| Tip | Details |
|-----|---------|
| **Same bounding box** | Use Worldview's **Share** link (🔗 icon) to copy the exact bbox, then paste it when downloading both dates — this ensures the images align well. |
| **Consistent resolution** | Choose the same pixel resolution for both downloads so the images are the same size (reduces alignment work). |
| **GeoTIFF option** | Worldview offers a GeoTIFF download via the **Granule download** tab. TerraLens supports `.tif` natively. |
| **File naming** | The app accepts any filename — name them anything descriptive, e.g. `dixie_before.jpg` / `dixie_after.jpg`. |

---

## Larger / higher-resolution images

For production-quality scenes, register for free at:

- **Copernicus Open Access Hub** (<https://scihub.copernicus.eu/>) — Sentinel-2
  L2A at 10 m resolution, 5-day revisit.
- **USGS Earth Explorer** (<https://earthexplorer.usgs.gov/>) — Landsat
  Collection 2 Level-2 at 30 m resolution.

Both services provide co-registered multi-date imagery that can be exported as
GeoTIFF and loaded directly into TerraLens.
