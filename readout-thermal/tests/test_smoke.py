"""M0 smoke test: every module imports."""

import importlib

import pytest

MODULES = [
    "readout_thermal",
    "readout_thermal.constants",
    "readout_thermal.sensor",
    "readout_thermal.thermal",
    "readout_thermal.selfconsistent",
    "readout_thermal.noise",
    "readout_thermal.sweep",
    "readout_thermal.calibrate",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    importlib.import_module(name)


def test_constants_si() -> None:
    from readout_thermal import constants

    assert constants.KB == 1.380649e-23
    assert constants.E == 1.602176634e-19
