from src.config import IForestConfig
from src.experiments import run_grid_experiments


if __name__ == "__main__":
    config = IForestConfig()
    summary_df = run_grid_experiments(config)
    print(summary_df)