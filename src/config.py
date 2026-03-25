from dataclasses import dataclass


@dataclass
class IForestConfig:
    input_csv: str = "data/ebird_data_processed.csv"
    output_scores_csv: str = "outputs/anomaly_scores.csv"
    output_top_csv: str = "outputs/top_anomalies.csv"
    output_experiment_csv: str = "outputs/experiment_summary.csv"
    figures_dir: str = "outputs/figures"
    n_trees: int = 100
    sample_size: int = 256
    random_state: int = 42
    top_k: int = 100
    anomaly_threshold: float = 0.60
    tree_grid: tuple = (25, 50, 100, 200)
    sample_grid: tuple = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)