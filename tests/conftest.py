"""Gemeinsame Fixtures: Importpfad der Demo, Reihen mit festem Zufalls-Seed, die exponentielle Mini-Reihe, an der Skalierung und Reihenfolge geprüft werden."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def reihe_spleiss10():
    """Eine Reihe des Spleiß-Falls (Anteil 10 %, n = 2000, Zufalls-Seed 1): Teststandard für die Eigenschaften."""
    import sws_cases as K
    return K.draw("splice10", 2000, 1)


@pytest.fixture(scope="session")
def reihe_exponential():
    import sws_cases as K
    return K.draw("exponential", 2000, 2)
