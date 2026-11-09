import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import contextily as cx
from pyproj import Transformer
import mplcursors

# ── Constants ──────────────────────────────────────────────────────────────────
TOTAL_OBS      = 1_093_203
TOTAL_REVIEWED = 8_161
EXPECTED_RATE  = TOTAL_REVIEWED / TOTAL_OBS   # 0.747%
MIN_OBS_PER_BIN = 20
CLIP_PP = 0.05   # colorbar clip at ±5 percentage points

# ── Load full dataset (for deviation heatmap) ─────────────────────────────────
print("Loading observation dataset...")
full_df = pd.read_csv(
    "anomaly_scores 2.csv",
    usecols=["LATITUDE", "LONGITUDE", "REVIEWED"]
)
full_df = full_df.dropna(subset=["LATITUDE", "LONGITUDE"])
print(f"  {len(full_df):,} observations | {TOTAL_REVIEWED:,} ground-truth anomalies ({EXPECTED_RATE*100:.3f}%)")
full_df["deviation"] = (full_df["REVIEWED"] == 1).astype(float) - EXPECTED_RATE

# ── Load top-100 model anomalies ──────────────────────────────────────────────
top_100_df = pd.read_csv("top_anomalies.csv").nsmallest(100, "anomaly_rank")

# ── Coordinate Transform ───────────────────────────────────────────────────────
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

full_x, full_y = transformer.transform(
    full_df["LONGITUDE"].values, full_df["LATITUDE"].values
)
top_x, top_y = transformer.transform(
    top_100_df["LONGITUDE"].values, top_100_df["LATITUDE"].values
)
top_100_df = top_100_df.assign(x=top_x, y=top_y)

confirmed   = top_100_df[top_100_df["REVIEWED"] == 1].reset_index(drop=True)
unconfirmed = top_100_df[top_100_df["REVIEWED"] != 1].reset_index(drop=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

# 1. Basemap (bottom layer)
# Draw a placeholder hexbin first to set axis extents, then draw real one after basemap
ax.hexbin(full_x, full_y, gridsize=75, alpha=0, zorder=0)
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, reset_extent=False, zorder=0)

# 2. Diverging deviation heatmap (slightly transparent so basemap geography shows)
hb = ax.hexbin(
    full_x, full_y,
    C=full_df["deviation"].values,
    reduce_C_function=np.mean,
    gridsize=75,
    cmap="RdBu_r",
    mincnt=MIN_OBS_PER_BIN,
    vmin=-CLIP_PP,
    vmax=CLIP_PP,
    alpha=0.70,
    zorder=1
)

# Colorbar
cbar = plt.colorbar(hb, ax=ax, fraction=0.03, pad=0.02, extend="both")
cbar.set_label("Anomaly rate − expected rate (percentage points)", fontsize=11)
cbar.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v*100:+.1f}pp")
)
cbar.ax.axhline(0, color="black", linewidth=1.2, linestyle="--")

# 3. Unconfirmed top-100: small dark-green dots
#    Dark green sits off the red/blue hexbin axis, visible on the white basemap.
sc_unc = ax.scatter(
    unconfirmed["x"], unconfirmed["y"],
    s=22, marker="o",
    color="#2ECC71",          # medium-bright green
    edgecolors="none",
    alpha=0.85,
    zorder=3,
    label=f"Model top-100, not confirmed (n={len(unconfirmed)})"
)

# 4. Confirmed (REVIEWED=1): small vivid-orange dots, same size, different colour
sc_con = ax.scatter(
    confirmed["x"], confirmed["y"],
    s=30, marker="o",
    color="#FFD400",          # golden amber
    edgecolors="none",
    alpha=0.90,
    zorder=4,
    label=f"Model top-100, REVIEWED=1 ground truth (n={len(confirmed)})"
)

# ── Interactive hover tooltips ─────────────────────────────────────────────────
for sc, subset in [(sc_unc, unconfirmed), (sc_con, confirmed)]:
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel, _df=subset):
        row = _df.iloc[sel.index]
        sel.annotation.set_text(
            f"Rank:          #{int(row['anomaly_rank'])}\n"
            f"Anomaly score: {row['anomaly_score']:.4f}\n"
            f"Taxon ID:      {row['TAXON CONCEPT ID']}\n"
            f"Confirmed:     {'Yes ★' if row['REVIEWED'] == 1 else 'No'}"
        )
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.88)

# ── Legend & labels ───────────────────────────────────────────────────────────
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.set_axis_off()
ax.set_title(
    f"Ground-Truth Anomaly Rate Deviation vs. Model Top-100 Predictions\n"
    f"Expected baseline: {EXPECTED_RATE*100:.3f}% | "
    f"Red = anomaly hot spot | Blue = below expected | "
    f"● green = model top-100 (not confirmed) | ● orange = confirmed (REVIEWED=1)  |  Hover for details",
    fontsize=11, pad=14
)

# Save first at 150 dpi (basemap resampling at 300 dpi over 14"x12" is very slow)
plt.savefig("Anomaly_Rate_Deviation_Map.png", dpi=150, bbox_inches="tight")
print("Saved: Anomaly_Rate_Deviation_Map.png")
plt.show()
