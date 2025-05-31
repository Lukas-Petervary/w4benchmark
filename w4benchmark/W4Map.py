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

    def set_dataset(self, geom_url: str, tensor_url: str):
        """Loads JSON geometry and tensor data, and maps molecule names to Molecule objects."""
        try:
            with open(geom_url, 'r') as geom_file:
                geom_data = json.load(geom_file)
            with open(tensor_url, 'r') as tensor_file:
                tensor_data = json.load(tensor_file)

            if not isinstance(geom_data, dict) or not isinstance(tensor_data, dict):
                raise ValueError("Both JSON files must contain an object at the root.")

            molecule_dict = {}
            for name, geom in geom_data.items():
                if name not in tensor_data:
                    print(f"Warning: No tensor info for molecule '{name}', skipping.")
                    continue
                basis = tensor_data[name]
                molecule_dict[name] = Molecule.parse_from_dict(name, geom, basis)

            self.data = ImmutableDict(molecule_dict)

        except FileNotFoundError as e:
            print(f"Error: File not found - {e.filename}")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON - {e}")

    def __getitem__(self, key) -> Molecule: return self.data[key]

    def __repr__(self): return f"W4 Data({self.data})"

    def __iter__(self) -> Iterator[tuple[str, Molecule]]:
        for key, value in self.data.items():
            yield key, value

    def init(self):
        """Initializes the dataset and runs the corresponding CLI function."""
        self.set_dataset(self.parameters.geominfo_url, self.parameters.tensorinfo_url)

        from .Decorators import W4Decorators
        if self.parameters.cli_function == "process":
            W4Decorators.main_process()
        elif self.parameters.cli_function == "analyze":
            W4Decorators.main_analyze()


# Initialize W4Map Singleton
Parameters._init_defaults()
W4 = W4Map(Parameters.DEFAULTS)