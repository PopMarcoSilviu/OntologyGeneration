from pathlib import Path

BASE = Path(__file__).parent.parent.parent
ROOT = BASE / "src"
DATA_PATH = BASE / "data"
MLFLOW_URI = f"sqlite:///{BASE / 'mlflow.db'}"
