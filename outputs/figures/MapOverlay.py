import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import numpy as np

# Load top anomalies from CSV (already pre-ranked; take top 100 by anomaly_rank)
top_100_df = pd.read_csv("top_anomalies.csv").nsmallest(100, "anomaly_rank")

# 1. Round coordinates to ~1 km grid to detect overlapping points
#    (~0.01 degrees ≈ 1 km at mid-latitudes; adjust finer/coarser as needed)
GRID_DEG = 0.1
top_100_df["lat_bin"] = (top_100_df["LATITUDE"] / GRID_DEG).round() * GRID_DEG
top_100_df["lon_bin"] = (top_100_df["LONGITUDE"] / GRID_DEG).round() * GRID_DEG

# 2. Count how many points fall in each bin
binned = (
    top_100_df.groupby(["lat_bin", "lon_bin"])
    .agg(count=("anomaly_rank", "size"), best_rank=("anomaly_rank", "min"))
    .reset_index()
)

# 3. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    binned,
    geometry=gpd.points_from_xy(binned["lon_bin"], binned["lat_bin"]),
    crs="EPSG:4326",
)
gdf = gdf.to_crs(epsg=3857)

# 4. Scale marker size by count (base size + extra per duplicate)
BASE_SIZE = 60
SIZE_SCALE = 80   # extra area per additional point in the bin
gdf["markersize"] = BASE_SIZE + (gdf["count"] - 1) * SIZE_SCALE

# 5. Color by count using a sequential colormap
fig, ax = plt.subplots(figsize=(12, 10))

scatter = ax.scatter(
    gdf.geometry.x,
    gdf.geometry.y,
    s=gdf["markersize"],
    c=gdf["count"],
    cmap="YlOrRd",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85,
    zorder=3,
)

# 6. Add colorbar showing point count per location
cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("# of Top-100 Anomalies at Location", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# 7. Add a size legend so readers understand what circle area means
unique_counts = sorted(gdf["count"].unique())
legend_counts = [c for c in [1, 2, 3, 5] if c <= max(unique_counts)]
if max(unique_counts) not in legend_counts:
    legend_counts.append(max(unique_counts))

legend_handles = [
    plt.scatter([], [], s=BASE_SIZE + (c - 1) * SIZE_SCALE,
                color="tomato", edgecolor="black", linewidth=0.6, alpha=0.85,
                label=f"{c} point{'s' if c > 1 else ''}")
    for c in legend_counts
]
ax.legend(handles=legend_handles, title="Count per location",
          loc="lower left", fontsize=9, title_fontsize=9,
          framealpha=0.85)

# 8. Add basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_axis_off()
ax.set_title("Top 100 Anomalies — Dot Size & Color = Count per Location",
             fontsize=13, pad=12)

plt.savefig("Top_100_Anomalies_300dpi.png", dpi=300, bbox_inches="tight")
plt.show()