"""Integration tests verifying all Streamlit dashboard pages load without exceptions."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_home_page_loads(project_root: Path) -> None:
    """Verify app/Home.py executes cleanly without exceptions."""
    home_file = project_root / "app/Home.py"
    at = AppTest.from_file(str(home_file))
    at.run(timeout=30)
    assert not at.exception, f"Home.py raised exception: {at.exception}"


def test_map_page_loads(project_root: Path) -> None:
    """Verify app/pages/1_Map.py executes cleanly without exceptions."""
    map_file = project_root / "app/pages/1_Map.py"
    at = AppTest.from_file(str(map_file))
    at.run(timeout=30)
    assert not at.exception, f"1_Map.py raised exception: {at.exception}"


def test_district_lookup_page_loads(project_root: Path) -> None:
    """Verify app/pages/2_District_Lookup.py executes cleanly without exceptions."""
    lookup_file = project_root / "app/pages/2_District_Lookup.py"
    at = AppTest.from_file(str(lookup_file))
    at.run(timeout=30)
    assert not at.exception, f"2_District_Lookup.py raised exception: {at.exception}"


def test_what_if_page_loads(project_root: Path) -> None:
    """Verify app/pages/3_What_If.py executes cleanly without exceptions."""
    whatif_file = project_root / "app/pages/3_What_If.py"
    at = AppTest.from_file(str(whatif_file))
    at.run(timeout=30)
    assert not at.exception, f"3_What_If.py raised exception: {at.exception}"


def test_priority_and_anomalies_page_loads(project_root: Path) -> None:
    """Verify app/pages/4_Priority_and_Anomalies.py executes cleanly without exceptions."""
    priority_file = project_root / "app/pages/4_Priority_and_Anomalies.py"
    at = AppTest.from_file(str(priority_file))
    at.run(timeout=30)
    assert not at.exception, f"4_Priority_and_Anomalies.py raised exception: {at.exception}"
