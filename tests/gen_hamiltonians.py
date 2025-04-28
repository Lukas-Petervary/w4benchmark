import base64, json, numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc
from pyscf.data.elements import chemcore
from w4benchmark import W4Decorators, Molecule, W4

hamiltonian_dict = {}

def serialize_tensor(tensor: np.ndarray):
    if isinstance(tensor, tuple): return [serialize_tensor(i) for i in tensor]
    return {
        "dtype": str(tensor.dtype),
        "shape": tensor.shape,
        "data": base64.b64encode(tensor.tobytes()).decode("utf-8")
    }

def gen_hamiltonians(molecule: Molecule):
    """funciton based on mwe of SQD"""
    # Create mol object
    scfMol = gto.Mole()
    scfMol.atom = molecule.geom
    scfMol.basis = W4.parameters.basis
    scfMol.symmetry = False
    scfMol.charge = molecule.charge
    scfMol.spin =   int(molecule.spin - 1)
    scfMol.verbose = 5
    scfMol.build()

    # Run Hartree Fock on all-electron space
    if 0 == scfMol.spin: mf = scf.RHF(scfMol)
    else: mf = scf.ROHF(scfMol)

    mf.kernel()
    # assert mf.converged

    # Active space computations

    # Freeze core
    nao = scfMol.nao_nr()
    ncore = chemcore(scfMol)
    nelec_as = tuple(nelec - ncore for nelec in scfMol.nelec)  # KEY2
    ncas = nao - ncore  # KEY1
    active_orbs = [p for p in range(ncore, nao)]
    frozen_orbs = [i for i in range(nao) if i not in active_orbs]

    open_shell = not (nelec_as[0] == nelec_as[1])

    np_orbs = np.array(active_orbs)
    # Post-Hartree calculations
    ## Hartree-Fock SCF solutions: basis functions (eigenvectors of Fock operator)
    ##   defined as linear combinations of the basis set functions
    ## Reference state: Hartree-Fock ground state
    # Get molecular coefficients of basis functions from Hartree Fock results (Slater determinants)
    mo = mf.mo_coeff
    # Form subset of Slater determinants in active space by picking out the corresponding coefficients
    # Note: the Hartree-Fock solutions are automatically ordered/sorted according to eigenvalues by PySCF
    h1e_cas = ecore = h2e_cas = mf_cas = None
    if isinstance(mf, scf.hf.RHF) or isinstance(mf, scf.rohf.ROHF):
        mf_cas = mcscf.CASSCF(mf, ncas, nelec_as)
        h1e_cas, ecore = mf_cas.get_h1eff()
        eri = mf_cas.mol.intor('int2e')  # This is not working for mo_cas
        h2e = ao2mo.incore.full(eri, mo)
        h2e_cas = h2e[np.ix_(np_orbs, np_orbs, np_orbs, np_orbs)]
    elif isinstance(mf, scf.uhf.UHF):
        mf_cas = mcscf.UCASSCF(mf, ncas, nelec_as)
        h1e_cas, ecore = mf_cas.get_h1eff()
        h2e_cas = mf_cas.get_h2eff()

    cc_as = cc.CCSD(mf).run() if len(frozen_orbs) == 0 else cc.CCSD(mf, frozen=frozen_orbs).run()
    # print(cc_as)
    hamiltonian_dict[molecule.species] = {
        "basis": {
            W4.parameters.basis: {
                "h1e": serialize_tensor(h1e_cas),
                "ecore": str(ecore),
                "h2e": serialize_tensor(h2e_cas),
                "ncas": mf_cas.ncas,
                "nelecas": mf_cas.nelecas,
                "cct2": serialize_tensor(cc_as.t2)
            }
        }
    }

@W4Decorators.process(basis="sto6g")
def iter_gen(name: str, mol: Molecule):
    print(f"\n\Generating {name}:")
    gen_hamiltonians(mol)

if __name__ == '__main__':
    # uncomment to run without CLI arg "--process"
    # W4.parameters.basis = "sto6g"
    # W4.init()
    # W4Decorators.main_process()
    with open("hamiltonian_dataset.json", "w") as f:
        json.dump(hamiltonian_dict, f, indent=4)
    print("Finished calculating hamiltonians.")