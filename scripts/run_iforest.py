from src.config import IForestConfig
from src.run_pipeline import run_pipeline


if __name__ == "__main__":
    config = IForestConfig()
    run_pipeline(config)