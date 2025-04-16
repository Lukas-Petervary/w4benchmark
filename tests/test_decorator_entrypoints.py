from w4benchmark import W4, W4Decorators

@W4Decorators.process(print_values = True)
def func1(key, value):
    print(key)
    if W4.parameters.print_values:
        print(value)

@W4Decorators.analyze(print_keys = True)
def func2(key, value):
    print(key if W4.parameters.print_keys else value)

if __name__ == '__main__':
    print("MAIN ENTRYPOINT")