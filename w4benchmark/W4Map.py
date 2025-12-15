from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar, Generic, Iterator
from .Molecule import Molecule
from .Params import Parameters
from .Logger import W4Logger
import json, requests, shutil, argparse

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

K = TypeVar('K')
V = TypeVar('V')

class ImmutableDict(Mapping[K, V], Generic[K, V]):
    def __init__(self, *args, **kwargs):
        self._store = dict(*args, **kwargs)

    def __getitem__(self, key: K) -> V: return self._store[key]
    def __iter__(self) -> Iterator[K]: return iter(self._store)
    def __len__(self) -> int: return len(self._store)
    def __repr__(self) -> str: return f"I{self._store!r}"

    def copy(self) -> "ImmutableDict[K, V]":
        return ImmutableDict(self._deep_copy(self._store))

    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj

    def __setitem__(self, key, value): raise self.ImmutableMutationError()
    def __delitem__(self, key): raise self.ImmutableMutationError()

    class ImmutableMutationError(Exception):
        def __init__(self):
            super().__init__("This data is immutable and must be explicitly dereferenced.")


class W4Map(metaclass=SingletonMeta):
    """
    Singleton wrapper for the W4 dataset manager.

    This class is responsible for coordinating access to W4 benchmark data,
    including downloading tensor data from a remote API, reading associated
    geometry and tensor files, parsing them into `Molecule` objects, and storing
    them in an immutable dictionary for safe usage across the application.

    Attributes:
        parameters (Parameters): The current configuration, including API URLs and options.
        data (ImmutableDict[str, Molecule]): Mapping from molecule names to Molecule instances.

    Functions:
        init():
            Initializes the W4 dataset. If the data file doesn't exist locally,
        it fetches it from the API. It then loads the dataset and triggers the
        appropriate decorated CLI function (process or analyze).
    """
    def __init__(self):
        self.parameters: Parameters = Parameters()
        self.data: ImmutableDict[str, Molecule] = ImmutableDict()

    def _api_call(self, basis_name: str) -> Path:
        try:
            url = self.parameters.api_url + "/entries/" + basis_name
            response = requests.get(url, headers={"Accept": "application/json"}, stream=True)
            response.raw.decode_content = True

            if response.status_code == 200:
                outfile = Path(self.parameters.resources_url) / f'{basis_name}.json'
                W4Logger.debug(f'Downloading {basis_name} entries...')
                with open(outfile, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
                return outfile.resolve()

            elif response.status_code == 404: raise FileNotFoundError(f"Basis not found (404): {basis_name}")
            elif response.status_code == 500: raise RuntimeError("Server error (500).")
            else: response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    @staticmethod
    def _lazy_load_tensor(tensor_url: Path):
        with open(tensor_url, 'r') as tensor_file:
            while True:
                line = tensor_file.readline()
                if line: yield json.loads(line)
                else:    break

    def _set_dataset(self, tensor_url: Path):
        """Loads JSON geometry and tensor data, and maps molecule names to Molecule objects."""
        with open(self.parameters.geominfo_url, 'r') as geom_file:
            geom_data = json.load(geom_file)

        molecule_dict = {}
        for element in self._lazy_load_tensor(tensor_url):
            if "$basis" in element:
                if element['$basis'] != W4.parameters.basis:
                    raise TypeError(f"Basis '{element['$basis']}' of dataset '{tensor_url}' does not match basis '{W4.parameters.basis}'")
                else: continue

            species = element['name']
            if species not in geom_data:
                W4Logger.warn(f'Molecule "{species}" not found in molecule geometries')
                continue
            geom = geom_data[species]
            molecule_dict[species] = Molecule.parse_from_dict(species, geom, element)

        self.data = ImmutableDict(molecule_dict)

    def __getitem__(self, key) -> Molecule: return self.data[key]

    def __repr__(self): return f"W4 Data({self.data})"

    def __iter__(self) -> Iterator[tuple[str, Molecule]]:
        for key, value in self.data.items():
            yield key, value

    def init(self):
        if self.data: return

        if W4.parameters["debug"]:
            W4Logger.setLevel(W4.parameters["debug"])

        W4Logger.debug("Initializing dataset")
        if self.parameters.basis is None:
            W4Logger.critical("No basis specified.")
            return

        local_file = self.parameters.resources_url / f"{self.parameters.basis}.json"
        if local_file.exists():
            tensor_file = local_file
        else:
            W4Logger.info(f"Querying API data for basis '{self.parameters.basis}' ...")
            tensor_file = self._api_call(self.parameters.basis)
        W4Logger.debug(f"Loading data from {local_file}")

        self._set_dataset(tensor_file)

# Initialize W4Map Singleton
Parameters._gen_defaults()
W4 = W4Map()