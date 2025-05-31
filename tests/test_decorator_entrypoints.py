from w4benchmark import *

@W4Decorators.process(print_values = True)
def func1(key: str, value: Molecule):
    print(key)
    if W4.parameters.print_values:
        print(value.basis.ecore)

@W4Decorators.analyze(print_keys = False)
def func2(key: str, value: Molecule):
    print(key if W4.parameters.print_keys else value)

if __name__ == '__main__':
    print("MAIN ENTRYPOINT")