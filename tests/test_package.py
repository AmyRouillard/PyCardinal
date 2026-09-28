import pycardinal


def test_package_has_version() -> None:
    assert pycardinal.__version__ == "0.1.0"


def test_simulation_api_is_exported() -> None:
    assert callable(pycardinal.simulate_spectra)
    assert callable(pycardinal.simulate_image)
