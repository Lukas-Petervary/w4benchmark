import traceback
from . import Parameters
from .W4Map import W4

class W4Decorators:
    _PROCESS_FUNC = None
    _ANALYZE_FUNC = None

    class DecoratorParameterError(Exception):
        def __init__(self): super().__init__("The decorated function must have exactly 2 parameters, <name, molecule>")

    class MultipleDecoratedError(Exception):
        def __init__(self): super().__init__("Each decorator can only be used once.")

    @classmethod
    def process(cls, **kwargs):
        if cls._PROCESS_FUNC is not None:
            raise cls.MultipleDecoratedError()

        def decorator(func):
            if func.__code__.co_argcount != 2:
                raise cls.DecoratorParameterError()
            cls._PROCESS_FUNC = func
            if Parameters.DEFAULTS["cli_function"] == "process":
                print("@PROCESS\n")
                W4.parameters = Parameters({}, **kwargs)
                W4.init()
            return func
        return decorator

    @classmethod
    def analyze(cls, **kwargs):
        if cls._ANALYZE_FUNC is not None:
            raise cls.MultipleDecoratedError()

        def decorator(func):
            if func.__code__.co_argcount != 2:
                raise cls.DecoratorParameterError()
            cls._ANALYZE_FUNC = func
            if Parameters.DEFAULTS["cli_function"] == "analyze":
                print("@ANALYZE\n")
                W4.parameters = Parameters({}, **kwargs)
                W4.init()
            return func
        return decorator

    @classmethod
    def main_process(cls):
        if cls._PROCESS_FUNC is not None: cls.execute_registered_functions(cls._PROCESS_FUNC)
        else: print("No process function defined.")
        cls._PROCESS_FUNC = None

    @classmethod
    def main_analyze(cls):
        if cls._ANALYZE_FUNC is not None: cls.execute_registered_functions(cls._ANALYZE_FUNC)
        else: print("No analyze function defined.")
        cls._ANALYZE_FUNC = None


    @classmethod
    def execute_registered_functions(cls, func):
        for key, value in W4.data.items():
            try: func(key, value)
            except Exception as e:
                print(f"\033[91mError while processing {key}: {e}\033[0m")
                traceback.print_exc()
