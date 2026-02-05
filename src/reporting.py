# src/reporting.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, top_n: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if top_n is not None:
        d = d.head(top_n)
    plt.figure()
    plt.bar(d[x].astype(str), d[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
