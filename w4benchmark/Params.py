import argparse

class Parameters(dict):
    DEFAULTS = {                # library required parameters
        "dataset_url": str,    # dynamically set in package __init__.py
        "cli_function": str,   # dynamically set in W4Map.py
    }

    @classmethod
    def _init_defaults(cls):
        import importlib.resources as resources
        with resources.path('w4benchmark', 'out.json') as path:
            Parameters.DEFAULTS["dataset_url"] = path

        parser = argparse.ArgumentParser(description="Run functions based on the argument passed.")
        parser.add_argument('--process', action='store_true', help="Run the process function")
        parser.add_argument('--analyze', action='store_true', help="Run the analyze function")
        args = parser.parse_args()

        if args.process:
            Parameters.DEFAULTS["cli_function"] = "process"
        elif args.analyze:
            Parameters.DEFAULTS["cli_function"] = "analyze"

    def __init__(self, copy: dict, **kwargs):
        super().__init__(self.DEFAULTS)
        self.update(copy)
        self.update(kwargs)

    def __getattr__(self, item): return self[item]
    def __setattr__(self, key, value): self[key] = value
    def __repr__(self): return f"Parameters({super().__repr__()})"