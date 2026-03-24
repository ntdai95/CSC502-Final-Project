import os
import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["species_frequency", "OBSERVATION COUNT", "LATITUDE", "LONGITUDE", "day_sin", "day_cos", "REVIEWED"]
META_COLUMNS = ["TAXON CONCEPT ID"]


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_processed_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}\nRun 'python scripts/process_data.py' first.")

    df = pd.read_csv(csv_path)
    required_cols = META_COLUMNS + FEATURE_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_feature_matrix(df):
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values.")
    
    return X