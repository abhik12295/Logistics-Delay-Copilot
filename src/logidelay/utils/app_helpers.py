from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_sample_data(root_dir: str) -> pd.DataFrame:
    sample_path = Path(root_dir) / "data" / "sample" / "sample_logistics_events.csv"

    if not sample_path.exists():
        raise FileNotFoundError(
            "Sample data not found. Run: uv run python scripts/prepare_sample_data.py"
        )

    return pd.read_csv(sample_path)


def get_project_root(current_file: str) -> Path:
    return Path(current_file).resolve().parents[2]


def inject_src_path(root_dir: Path) -> None:
    import sys

    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))