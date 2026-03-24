import pandas as pd
import numpy as np


def summarize_scores(scores):
    return pd.DataFrame({"score_mean": [float(np.mean(scores))], "score_std": [float(np.std(scores))], 
                         "score_min": [float(np.min(scores))], "score_median": [float(np.median(scores))],
                         "score_max": [float(np.max(scores))]})


def add_ranking_columns(df, score_col="anomaly_score"):
    out = df.copy()
    out["anomaly_rank"] = out[score_col].rank(method="dense", ascending=False).astype(int)
    return out.sort_values(score_col, ascending=False)