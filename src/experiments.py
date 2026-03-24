import itertools
import pandas as pd
from src.data_utils import ensure_parent_dir, load_processed_data, get_feature_matrix
from src.iforest import IsolationForestCustom
from src.visualization import save_experiment_mean_score_plot, save_experiment_std_plot


def run_single_experiment(input_csv, n_trees, sample_size, random_state):
    df = load_processed_data(input_csv)
    X = get_feature_matrix(df)
    model = IsolationForestCustom(n_trees, sample_size, random_state)
    model.fit(X)
    scores = model.score_samples(X)
    return pd.DataFrame({"n_trees": [n_trees], "sample_size": [sample_size], "score_mean": [float(scores.mean())],
                         "score_std": [float(scores.std())], "score_min": [float(scores.min())],
                         "score_median": [float(pd.Series(scores).median())], "score_max": [float(scores.max())]})
    

def run_grid_experiments(config):
    ensure_parent_dir(config.output_experiment_csv)
    tree_grid = [25, 50, 100, 200]
    sample_grid = [64, 128, 256, 512]
    rows = []
    for n_trees, sample_size in itertools.product(tree_grid, sample_grid):
        result = run_single_experiment(config.input_csv, n_trees, sample_size, config.random_state)
        rows.append(result)

    summary_df = pd.concat(rows, ignore_index=True)
    summary_df.to_csv(config.output_experiment_csv, index=False)
    save_experiment_mean_score_plot(summary_df, f"{config.figures_dir}/experiment_mean_score_vs_trees.png")
    save_experiment_std_plot(summary_df, f"{config.figures_dir}/experiment_std_score_vs_sample_size.png")
    return summary_df