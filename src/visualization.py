import os
import matplotlib.pyplot as plt


def save_score_histogram(scored_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(scored_df["anomaly_score"], bins=40)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Isolation Forest Anomaly Scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_top_anomalies_map(scored_df, out_path, top_k=100):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    top_df = scored_df.head(top_k)
    plt.figure(figsize=(8, 6))
    plt.scatter(top_df["LONGITUDE"], top_df["LATITUDE"], s=12)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Top {top_k} Anomalous Observations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_experiment_mean_score_plot(summary_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grouped = summary_df.groupby("n_trees", as_index=False)["score_mean"].mean().sort_values("n_trees")
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["n_trees"], grouped["score_mean"], marker="o")
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Anomaly Score")
    plt.title("Mean Anomaly Score vs Number of Trees")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_experiment_std_plot(summary_df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grouped = summary_df.groupby("sample_size", as_index=False)["score_std"].mean().sort_values("sample_size")
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["sample_size"], grouped["score_std"], marker="o")
    plt.xlabel("Sample Size")
    plt.ylabel("Standard Deviation of Scores")
    plt.title("Score Standard Deviation vs Sample Size")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()