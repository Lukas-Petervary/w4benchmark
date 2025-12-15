from .Params import Parameters
from .Logger import W4Logger
from .W4Map import W4
import sys

class W4Decorators:
    class DecoratorParameterError(Exception):
        def __init__(self): super().__init__("The decorated function must have exactly 2 parameters: 'def func(name: str, mol: Molecule)'")

    @classmethod
    def process(cls, **kwargs):
        def decorator(func):
            if func.__code__.co_argcount != 2: raise cls.DecoratorParameterError()

            if "--process" in sys.argv:
                W4.parameters = Parameters(**kwargs)
                W4.init()
                cls.iterate_function(func)

            return func
        return decorator

    @classmethod
    def analyze(cls, **kwargs):
        def decorator(func):
            if func.__code__.co_argcount != 2: raise cls.DecoratorParameterError()

            if "--analyze" in sys.argv:
                W4.parameters = Parameters(**kwargs)
                W4.init()
                cls.iterate_function(func)

            return func
        return decorator

    @staticmethod
    def _pnum(func): return f"{func.__name__} at {func.__code__.co_filename}: line {func.__code__.co_firstlineno}"

    @classmethod
    def iterate_function(cls, func):
        for key, value in W4.data.items():
            try: func(key, value)
            except Exception as e:
                W4Logger.error(f"Error while processing {key}: {e}")
