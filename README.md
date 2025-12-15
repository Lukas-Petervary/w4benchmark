# W4 Benchmark

An algorithm benchmarking tool built around the W4-11 dataset of high-accuracy thermochemical data.  
Designed for evaluating the performance of computational chemistry methods.

## Features

- Decorator-based workflow with CLI execution
- Programmatic entrypoint for manual workflows
- Built-in support for iterating over all molecules in the W4-11 dataset

## Installation

Install the package via pip:

    pip install w4benchmark

Requires: `Python 3.6+, numpy >= 2.2.4, requests >= 2.32.3`

## Usage

This package provides two main decorators:

- `@W4Decorators.process(...)`: For processing each molecule (e.g., computing energies)
- `@W4Decorators.analyze(...)`: For analyzing the results (e.g., comparing predictions to reference data)

Each decorated function **must accept exactly two parameters**: the molecule name (`str`) and a `Molecule` object.

Each decorator accepts a list of runtime parameters, with a predefined set of variables set by default at decoration time:
- `geominfo_url`: specifies the directory of species geometries data file
- `resources_url`: specifies the root of the resources directory for finding cached basis sets, geometries and more
- `api_url`: the URL to query for basis set info if the specified basis is not found in resources directory
- `basis`: the basis information to parse from resource directory (defaults to "sto6g", and if set to a value not found in resources will query from `api_url`)
- `debug`: the debug level for logging output (defaults to `logging.WARNING`)

All parameters can be queried and modified during runtime via the `W4.parameters` object. Arbitrary parameters can be added and queried at runtime to keep track of other runtime parameters for specific endpoints (ex. `@W4Decorators.process( printValues = true )` will add a field `printValues` to `W4.parameters` with a value of `true`).

### Example: Script with Decorators

Create a script like `compute.py`:

    from w4benchmark import W4Decorators, Molecule, W4

    @W4Decorators.process(basis="sto6g", debug=logging.DEBUG)
    def compute_energy(name: str, mol: Molecule):
        # Replace with real computation

    @W4Decorators.analyze(basis="sto6g")
    def analyze_results(name: str, mol: Molecule):
        # Replace with real analytics

Then run from the command line with either:

    python compute.py --process

or:

    python compute.py --analyze

Each command will iterate over every molecule in the dataset and apply the corresponding decorated function.

### Manual Iteration

If you want full control, you can manually run the W4 benchmark from within a `__main__` block:

    from w4benchmark import W4

    if __name__ == '__main__':
        W4.parameters.basis = "sto6g"  # Set runtime parameters
        W4.init() # manual execution requires the .init() function to be called

        # Example usage
        for name, mol in W4:
            print(f"{name}: spin = {mol.spin}, charge = {mol.charge}")

This allows you to iterate through the dataset as a normal iterable. You can also dereference `W4` with a specific species if you want to select a singular molecule object (ex. `W4["acetaldehyde"]`).
## SQD Minimal Working Example

The MWE illustrates the essential components of the SQD algorithm:

- **Basic random quantum circuit**: A small quantum circuit is generated with random single-qubit rotations and entangling gates.
- **Measurement simulation**: The wavefunction is sampled ideally (without noise) to generate measurement bitstrings.
- **Electron configuration conversion**: Measurements are filtered into electron configurations.
- **SQD execution**: A toy 4-qubit Hamiltonian is used with the SQD method to approximate the ground-state energy.
- **Simplified output**: Key results like sampled configurations and computed energies are printed.

This provides a lightweight, self-contained example of how SQD can be implemented, separate from the full W4-11 benchmark infrastructure.

## Features

- **No dataset dependencies**: The MWE uses a manually defined Hamiltonian rather than reading from benchmark datasets.
- **Simple circuit creation**: Random gates are applied to generate a demonstrative quantum circuit.
- **Ideal measurement sampling**: Bitstrings are sampled from the exact wavefunction, not from noisy hardware simulations.
- **Subspace filtering**: Bitstrings are filtered based on electron count constraints to form a subspace.
- **Projected Hamiltonian diagonalization**: An approximate ground-state energy is computed from the subspace.
- **Minimal dependencies**: Only `numpy`, `qiskit`, and standard Python libraries are required.


## Dataset Attribution

This tool uses the **W4-11 dataset** provided by:

    Goerigk, L., & Grimme, S. (2011).
    A thorough benchmark of density functional methods for general main group thermochemistry, kinetics, and noncovalent interactions.
    Phys. Chem. Chem. Phys., 13, 6670–6688.
    https://doi.org/10.1039/C0CP02984J

Please **cite this publication** in any work that uses this package or the underlying dataset.

The dataset is intended for **academic use only**. Redistribution of the dataset itself may be restricted by the publisher.

## License

This software is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
You may use and adapt the software for non-commercial purposes with proper attribution.

See the full license in the `LICENSE` file.
