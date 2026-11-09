"""Data loading utilities."""

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import yaml


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: Union[str, Path]) -> pd.DataFrame:
    """Load the raw NFHS Excel file."""
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_excel(path)


def load_csv_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file with path validation."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found at: {path}")
    return pd.read_csv(path)
