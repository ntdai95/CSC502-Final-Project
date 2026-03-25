import itertools
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data_utils import ensure_parent_dir, load_processed_data, get_feature_matrix, get_labels
from src.iforest import IsolationForestCustom
from src.visualization import save_auc_vs_sample_size, save_time_vs_sample_size, save_auc_vs_n_trees, save_time_vs_n_trees


def run_single_experiment(input_csv, n_trees, sample_size, random_state):
    df = load_processed_data(input_csv)
    X = get_feature_matrix(df)
    y = get_labels(df)
    sample_size = min(sample_size, len(df))
    model = IsolationForestCustom(n_trees, sample_size, random_state)
    start_train = time.perf_counter()
    model.fit(X)
    train_time_sec = time.perf_counter() - start_train
    start_eval = time.perf_counter()
    scores = model.score_samples(X)
    eval_time_sec = time.perf_counter() - start_eval
    total_time_sec = train_time_sec + eval_time_sec
    auc = roc_auc_score(y, scores)
    return pd.DataFrame({"n_trees": [n_trees], "sample_size": [sample_size], "auc": [float(auc)],
                         "train_time_sec": [float(train_time_sec)], "eval_time_sec": [float(eval_time_sec)],
                         "total_time_sec": [float(total_time_sec)], "score_mean": [float(scores.mean())],
                         "score_std": [float(scores.std())], "score_min": [float(scores.min())],
                         "score_median": [float(pd.Series(scores).median())], "score_max": [float(scores.max())]})
    

def run_grid_experiments(config):
    ensure_parent_dir(config.output_experiment_csv)
    rows = []
    for n_trees, sample_size in itertools.product(config.tree_grid, config.sample_grid):
        result = run_single_experiment(config.input_csv, n_trees, sample_size, config.random_state)
        rows.append(result)

    summary_df = pd.concat(rows, ignore_index=True)
    summary_df.to_csv(config.output_experiment_csv, index=False)
    save_auc_vs_sample_size(summary_df, f"{config.figures_dir}/auc_vs_sample_size.png")
    save_time_vs_sample_size(summary_df, f"{config.figures_dir}/time_vs_sample_size.png")
    save_auc_vs_n_trees(summary_df, f"{config.figures_dir}/auc_vs_n_trees.png")
    save_time_vs_n_trees(summary_df, f"{config.figures_dir}/time_vs_n_trees.png")
    return summary_df