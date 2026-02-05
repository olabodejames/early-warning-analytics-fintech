# src/io_utils.py
import pandas as pd
from pathlib import Path

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def enforce_columns(df: pd.DataFrame, required: list[str], name: str = "DataFrame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def safe_to_numeric(series: pd.Series) -> pd.Series:
    # Handles commas and stray spaces; preserves NaN
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")
