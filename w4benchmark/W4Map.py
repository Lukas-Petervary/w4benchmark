from collections.abc import Mapping
from typing import TypeVar, Generic, Iterator
from .Molecule import Molecule
from .Params import Parameters
import json

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
    def __init__(self, params=Parameters.DEFAULTS):
        self.parameters: Parameters = Parameters(params)
        self.data: ImmutableDict[str, Molecule] = ImmutableDict()

    def set_dataset(self, dataset_url: str):
        """Loads a JSON dataset and maps molecule names to Molecule objects."""
        try:
            with open(dataset_url, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    molecule_dict = {
                        k: Molecule.parse_from_dict(k, v) for k, v in data.items()
                    }
                    self.data = ImmutableDict(molecule_dict)  # Store as immutable dictionary
                else: raise ValueError("JSON file must contain an object at the root.")
        except FileNotFoundError: print(f"Error: File '{dataset_url}' not found.")
        except json.JSONDecodeError: print(f"Error: Failed to decode JSON from '{dataset_url}'.")

    def __getitem__(self, key) -> Molecule: return self.data[key]

    def __repr__(self): return f"W4 Data({self.data})"

    def __iter__(self) -> Iterator[tuple[str, Molecule]]:
        for key, value in self.data.items():
            yield key, value

    def init(self):
        """Initializes the dataset and runs the corresponding CLI function."""
        self.set_dataset(self.parameters.dataset_url)

        from .Decorators import W4Decorators
        if self.parameters.cli_function == "process":
            W4Decorators.main_process()
        elif self.parameters.cli_function == "analyze":
            W4Decorators.main_analyze()


# Initialize W4Map Singleton
Parameters._init_defaults()
W4 = W4Map(Parameters.DEFAULTS)