from w4benchmark import W4, W4Decorators, Molecule
import logging

@W4Decorators.process(basis="sto6g", print_values = False, debug=logging.DEBUG)
def func1(key: str, value: Molecule):
    print(key)
    if W4.parameters.print_values:
        print(value.basis.ecore)

@W4Decorators.analyze(basis="sto6g", print_keys = True, debug=logging.DEBUG)
def func2(key: str, value: Molecule):
    print(key if W4.parameters.print_keys else value)

if __name__ == '__main__':
    print("MAIN ENTRYPOINT")