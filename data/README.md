# Sample Data for TerraLens

## Bundled sample images (no credentials required)

Two synthetic satellite-style images are included in this folder and tracked
in the repository.  They work out-of-the-box — no downloads needed:

| File | Description |
|------|-------------|
| `sample_before.png` | 640×480 synthetic scene — dark-green forest, agriculture, water |
| `sample_after.png` | Same scene with a large bright ash/burn-scar region (simulated wildfire) |

**Expected result at threshold 40:** ~14 % scene change, 1–2 distinct regions,
clear burn-scar overlay in the top-right quadrant.

### How to use

1. `streamlit run app.py`
2. Select **📂 Upload images** in the sidebar
3. Upload `data/sample_before.png` → **Before**
4. Upload `data/sample_after.png` → **After**
5. Click **✨ Generate AI Narration**

---

## Download real satellite imagery (optional)

For real-world analysis, download a free image pair from NASA Worldview.
The 2021 Dixie Fire in northern California is a high-contrast, well-documented
event that produces a clear change mask.

### Step-by-step

1. Open **NASA Worldview**: <https://worldview.earthdata.nasa.gov/>
2. Set date to **2021-07-13** → download → save as `data/dixie_before.jpg`
3. Set date to **2021-08-20** → download → save as `data/dixie_after.jpg`
4. Upload both in the app; set date range `July 13 – August 20, 2021`

> **Tip:** Use the app's **🛰️ Fetch from NASA Worldview** mode with the
> **🔥 2021 Dixie Fire** quick-select preset to download automatically.

---

## Notes

- Downloaded tiles (`.png`, `.jpg`, `.tif`) are excluded from git via
  `.gitignore`.  Only `sample_before.png` and `sample_after.png` are tracked.
- The `results/` directory is where overlay PNGs are saved at runtime.
