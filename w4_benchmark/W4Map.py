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


class ImmutableDict(dict):
    def __setitem__(self, key, value):
        raise self.ImmutableMutationError()

    def __delitem__(self, key):
        raise self.ImmutableMutationError()

    def __repr__(self):
        return f"I{super().__repr__()}"

    def copy(self):
        return ImmutableDict(self._deep_copy(self))

    def _deep_copy(self, obj):
        if isinstance(obj, dict):
            return ImmutableDict({k: self._deep_copy(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj

    class ImmutableMutationError(Exception):
        def __init__(self): super().__init__("This data is immutable and must be explicitly dereferenced.")


class W4Map(metaclass=SingletonMeta):
    def __init__(self, params=Parameters.DEFAULTS):
        self.parameters: Parameters = Parameters(params)
        self.data = ImmutableDict()  # Store dataset as an immutable dictionary

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

    def save_json(self, json_file):
        """Saves the current dataset to a JSON file."""
        with open(json_file, 'w') as file:
            json.dump(self.data, file, indent=4)

    def __getitem__(self, key): return self.data[key]
    def __iter__(self): return iter(self.data.keys())
    def __repr__(self): return f"W4 Data({self.data})"

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