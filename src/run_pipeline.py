from src.data_utils import ensure_parent_dir, load_processed_data, get_feature_matrix
from src.iforest import IsolationForestCustom
from src.metrics import add_ranking_columns
from src.visualization import save_score_histogram, save_top_anomalies_map


def run_pipeline(config):
    ensure_parent_dir(config.output_scores_csv)
    ensure_parent_dir(config.output_top_csv)
    df = load_processed_data(config.input_csv)
    X = get_feature_matrix(df)
    model = IsolationForestCustom(config.n_trees, config.sample_size, config.random_state)
    model.fit(X)
    scores = model.score_samples(X)
    preds = model.predict(X, threshold=config.anomaly_threshold)
    scored_df = df.copy()
    scored_df["anomaly_score"] = scores
    scored_df["is_anomaly"] = preds
    scored_df = add_ranking_columns(scored_df, score_col="anomaly_score")
    scored_df.to_csv(config.output_scores_csv, index=False)
    scored_df.head(config.top_k).to_csv(config.output_top_csv, index=False)
    save_score_histogram(scored_df, f"{config.figures_dir}/score_histogram.png")
    save_top_anomalies_map(scored_df, f"{config.figures_dir}/top_anomalies_map.png", config.top_k)
    print(f"Saved scores to: {config.output_scores_csv}")
    print(f"Saved top anomalies to: {config.output_top_csv}")
    print(f"Saved figures to: {config.figures_dir}")
    print(f"\nTop 10 anomalies:\n{scored_df.head(10)[['TAXON CONCEPT ID', 'anomaly_score', 'is_anomaly', 'LATITUDE', 'LONGITUDE']]}")