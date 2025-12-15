from collections import UserDict
from pathlib import Path
from .Logger import W4Logger
import importlib.resources as resources

class Parameters(UserDict):
    """
    A configurable parameter dictionary with default values for use in the W4Benchmark library.

    Default fields:
        - geominfo_url (PathLike): Path to mol_geoms.json
        - resources_url (PathLike): Path to the resources directory
        - api_url (str): REST API base URL
        - basis (str): Basis set name
        - cli_function (str): Function to invoke from CLI ('process' or 'analyze')
        - debug (int): Debug level for logging output
    """

    DEFAULTS: dict = {
        "geominfo_url": Path,
        "resources_url": Path,
        "api_url": str,
        "basis": str,
        "debug": int
    }

    def __init__(self, **kwargs):
        super().__init__({**Parameters.DEFAULTS, **kwargs})

    @staticmethod
    def _gen_defaults() -> None:
        try:
            with resources.path('w4benchmark.resources', 'mol_geoms.json') as f_geom:
                Parameters.DEFAULTS["geominfo_url"] = f_geom
            with resources.path('w4benchmark', 'resources') as res_dir:
                Parameters.DEFAULTS["resources_url"] = res_dir
        except FileNotFoundError:
            W4Logger.critical("Could not find default resources directory")

        Parameters.DEFAULTS["api_url"] = "https://w4dataset.foci.rpi.edu"
        Parameters.DEFAULTS["basis"] = 'sto6g'
        Parameters.DEFAULTS["debug"] = W4Logger.getEffectiveLevel()

    def __getattr__(self, key):
        if key == "data" or "data" not in self.__dict__:
            return super().__getattribute__(key)
        return self.data[key]

    def __setattr__(self, key, value):
        if key == "data" or "data" not in self.__dict__:
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __repr__(self): return f"Parameters({super().__repr__()})"
