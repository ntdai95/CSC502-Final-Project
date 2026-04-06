import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
import mplcursors

# ── Build TAXON CONCEPT ID → species name lookup from raw eBird file ──────────
# Read only the 3 needed columns in chunks to avoid loading the 11 GB file fully.
RAW_FILE = "ebd_CA-BC_201601_202601_smp_relFeb-2026.txt"
CHUNK_SIZE = 200_000

print("Building species name lookup from raw eBird file (chunked)...")
taxon_lookup = {}   # { "avibase-XXXX": ("Common Name", "Scientific Name") }
for chunk in pd.read_csv(
    RAW_FILE,
    sep="\t",
    usecols=["TAXON CONCEPT ID", "COMMON NAME", "SCIENTIFIC NAME"],
    dtype=str,
    chunksize=CHUNK_SIZE,
    low_memory=False,
):
    # Only keep rows whose taxon ID we haven't seen yet
    new_rows = chunk[~chunk["TAXON CONCEPT ID"].isin(taxon_lookup)]
    for _, row in new_rows.drop_duplicates("TAXON CONCEPT ID").iterrows():
        taxon_lookup[row["TAXON CONCEPT ID"]] = (
            row["COMMON NAME"], row["SCIENTIFIC NAME"]
        )
print(f"  {len(taxon_lookup):,} unique species found.")

# ── Data Loading ──────────────────────────────────────────────────────────────
print("Loading full observation dataset for density background...")
full_df = pd.read_csv("anomaly_scores 2.csv", usecols=["LATITUDE", "LONGITUDE"])
full_df = full_df.dropna(subset=["LATITUDE", "LONGITUDE"])
print(f"  {len(full_df):,} total observations loaded.")

top_100_df = pd.read_csv("top_anomalies.csv").nsmallest(100, "anomaly_rank")

# Attach species names
top_100_df["common_name"]     = top_100_df["TAXON CONCEPT ID"].map(lambda t: taxon_lookup.get(t, ("Unknown", ""))[0])
top_100_df["scientific_name"] = top_100_df["TAXON CONCEPT ID"].map(lambda t: taxon_lookup.get(t, ("", "Unknown"))[1])

# ── Spatial binning (~10 km grid) ─────────────────────────────────────────────
GRID_DEG = 0.1
top_100_df["lat_bin"] = (top_100_df["LATITUDE"] / GRID_DEG).round() * GRID_DEG
top_100_df["lon_bin"] = (top_100_df["LONGITUDE"] / GRID_DEG).round() * GRID_DEG

def summarise_bin(g):
    species_list = ", ".join(sorted(g["common_name"].unique()))
    return pd.Series({
        "num_anomalies":  len(g),
        "any_confirmed":  int(g["REVIEWED"].max()),
        "species_names":  species_list,
        "best_rank":      int(g["anomaly_rank"].min()),
    })

binned = (
    top_100_df.groupby(["lat_bin", "lon_bin"])
    .apply(summarise_bin, include_groups=False)
    .reset_index()
)

# ── Coordinate Transform ───────────────────────────────────────────────────────
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

full_x, full_y = transformer.transform(
    full_df["LONGITUDE"].values, full_df["LATITUDE"].values
)
bin_x, bin_y = transformer.transform(
    binned["lon_bin"].values, binned["lat_bin"].values
)
binned = binned.assign(x=bin_x, y=bin_y)

BASE_SIZE  = 100
SIZE_SCALE = 120
binned["markersize"] = BASE_SIZE + (binned["num_anomalies"] - 1) * SIZE_SCALE

confirmed   = binned[binned["any_confirmed"] == 1]
unconfirmed = binned[binned["any_confirmed"] != 1]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

# 1. Background density heatmap
hb = ax.hexbin(
    full_x, full_y,
    gridsize=100, cmap="Blues", bins="log",
    alpha=0.65, mincnt=1, zorder=1
)
cbar = plt.colorbar(hb, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Total Observations per Cell (log₁₀ scale)", fontsize=11)

# 2. Basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, reset_extent=False, zorder=0)

# 3. Red circles (unconfirmed bins)
sc_unc = ax.scatter(
    unconfirmed["x"], unconfirmed["y"],
    s=unconfirmed["markersize"], marker="o",
    color="crimson", edgecolor="white",
    linewidth=0.8, alpha=0.9, zorder=3,
    label=f"Top-100 anomaly, not confirmed (n={int(unconfirmed['num_anomalies'].sum())})"
)

# 4. Gold stars (confirmed bins)
sc_con = ax.scatter(
    confirmed["x"], confirmed["y"],
    s=confirmed["markersize"] * 2.0, marker="*",
    color="gold", edgecolor="black",
    linewidth=0.7, alpha=0.95, zorder=4,
    label=f"Top-100 anomaly, REVIEWED=1 (n={int(confirmed['num_anomalies'].sum())})"
)

# ── Interactive hover tooltips ─────────────────────────────────────────────────
def make_tooltip(row):
    lines = [
        f"Species: {row['species_names']}",
        f"Anomalies in cell: {int(row['num_anomalies'])}",
        f"Best rank: #{int(row['best_rank'])}",
        f"Confirmed (REVIEWED=1): {'Yes' if row['any_confirmed'] else 'No'}",
    ]
    return "\n".join(lines)

# Attach hover to both scatter artists
for sc, subset in [(sc_unc, unconfirmed), (sc_con, confirmed)]:
    rows = subset.reset_index(drop=True)
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel, _rows=rows):
        sel.annotation.set_text(make_tooltip(_rows.iloc[sel.index]))
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.85)

# ── Size legend ───────────────────────────────────────────────────────────────
max_count = int(binned["num_anomalies"].max())
legend_counts = sorted(set([1, 2] + ([max_count] if max_count > 2 else [])))
size_handles = [
    plt.scatter([], [], s=BASE_SIZE + (c - 1) * SIZE_SCALE,
                color="crimson", edgecolor="white", linewidth=0.8, alpha=0.9,
                label=f"{c} anomal{'y' if c == 1 else 'ies'} in cell")
    for c in legend_counts
]
size_legend = ax.legend(
    handles=size_handles,
    title="Dot size = # anomalies\nin 0.1° grid cell",
    loc="lower right", fontsize=9, title_fontsize=9, framealpha=0.88
)
ax.add_artist(size_legend)

ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
ax.set_axis_off()
ax.set_title(
    "Top-100 Isolation Forest Anomalies vs. Background Birder Activity\n"
    "(Hex colour = observation density; dot size = anomalies per 0.1° cell; ★ = REVIEWED=1)  |  Hover dots for species",
    fontsize=12, pad=14
)

plt.savefig("Top_100_Anomalies_300dpi.png", dpi=300, bbox_inches="tight")
print("Saved: Top_100_Anomalies_300dpi.png")
plt.show()
