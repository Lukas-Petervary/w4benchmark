import time
import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
import ffsim

from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import (
    bitstring_matrix_to_ci_strs,
    solve_fermion,
)
from qiskit_addon_sqd.subsampling import postselect_and_subsample
from w4_benchmark import *
from w4_benchmark.Molecule import Basis


# Create quantum circuits using t2 amplitudes
def get_LUCJ_qiskit_circuit(ncas, nelecas, isCASSCF, cct2, backend, layout_strategy, sampling_circuit=True):
    nelec = nelecas
    nela, nelb = nelec
    nvra, nvrb = ncas - nela, ncas - nelb
    spin = nela - nelb
    # spin = mf_as.mol().spin

    # n_reps = 2

    if "ibm_rensselaer" == backend and layout_strategy == 'diagonal':
        pairs_aa = [(i, i + 1) for i in range(ncas - 1)]
        if (ncas < 6):
            pairs_ab = [(0, 0)]
        elif (ncas == 6):
            pairs_ab = [(0, 0), (5, 5)]
        elif (ncas <= 25):
            pairs_ab = [(i, i) for i in range(ncas - 1) if not i % 4]
        elif (ncas <= 28):
            pairs_ab = [(i, i) for i in range(21 - 1) if not i % 4]
        else:
            pairs_ab = [(i, i) for i in range(17 - 1) if not i % 4]
        interaction_pairs = (pairs_aa, pairs_ab)
    elif "ibm_rensselaer" == backend and layout_strategy == 'grid':
        pairs_aa = [(i, i + 1) for i in range(ncas - 1)]
        pairs_ab = [(i, i) for i in range(ncas - 1) if not i % 4]
        interaction_pairs = (pairs_aa, pairs_ab)
    elif "ibm_rensselaer" == backend and layout_strategy == 'grid_dense':
        pairs_aa = [(i, i + 1) for i in range(ncas - 1)] + [pair for pair in
                                                            [(2, 28), (6, 24), (10, 20), (18, 44), (22, 40), (26, 36)]
                                                            if (pair[0] <= (ncas - 1) and pair[1] <= (ncas - 1))]
        pairs_ab = [(i, i) for i in range(ncas - 1) if not i % 4]
        interaction_pairs = (pairs_aa, pairs_ab)
    else:
        print("Interaction pairs not defined for this backend")

    interaction_pairs_bal = interaction_pairs
    interaction_pairs_unbal = (interaction_pairs[0], interaction_pairs[1], interaction_pairs[0])

    qubits = QuantumRegister(2 * ncas, name="q")
    lucj_circuit = QuantumCircuit(qubits)

    lucj_circuit.append(ffsim.qiskit.PrepareHartreeFockJW(ncas, nelec), qubits)
    lucj_circuit.barrier()
    if ncas > 1:  # Hydrogen throws an error because there are no orbitals to excite to in the minimal basis
        if isCASSCF:
            if 0 == spin:
                ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(cct2, n_reps=2,
                                                                   interaction_pairs=interaction_pairs)
                truncated_ucj_op = ffsim.UCJOpSpinBalanced(
                    diag_coulomb_mats=ucj_op.diag_coulomb_mats[:-1],
                    orbital_rotations=ucj_op.orbital_rotations[:-1],
                    final_orbital_rotation=ucj_op.orbital_rotations[-1]
                )
                lucj_op_params = truncated_ucj_op.to_parameters(interaction_pairs=interaction_pairs_bal)
                lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
                    lucj_op_params,
                    norb=ncas, n_reps=1,
                    interaction_pairs=interaction_pairs_bal,
                    with_final_orbital_rotation=True
                )
                lucj_circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
            else:
                t2 = cct2
                t2_tilde = np.zeros((nelb, nelb, nvrb, nvrb))
                # # excitations from the doubly-occupied to the virtuals from the opposite-spin t2's
                # t2_tilde[:nelb, :nelb, -nvra:, -nvra:] = t2[1][:nelb, :nelb, -nvra:, -nvra:]
                # # excitations from the doubly-occupied to the singly occupied from the minority (beta) same-spin t2's
                # t2_tilde[:nelb, :nelb, :spin, :spin] = t2[2][:nelb, :nelb, :spin, :spin]
                if nvra > 0:
                    # excitations from the doubly-occupied to the virtuals from the opposite-spin t2's
                    t2_tilde[:nelb, :nelb, -nvra:, -nvra:] = t2[1][:nelb, :nelb, -nvra:, -nvra:]
                else:
                    # excitations from the doubly-occupied to the virtuals from the minority (beta) t2's
                    t2_tilde[:nelb, :nelb, -nvra:, -nvra:] = t2[2][:nelb, :nelb, -nvra:, -nvra:]
                # excitations from the doubly-occupied to the singly occupied from the minority (beta) same-spin t2's
                t2_tilde[:nelb, :nelb, :spin, :spin] = t2[2][:nelb, :nelb, :spin, :spin]
                # in the absence of spin-down particles, take excitations from spin-up t2's (e.g., in the case of hygrogen atom)
                if nelb == 0:
                    t2_tilde = t2[0]

                ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                    t2_tilde, n_reps=2, interaction_pairs=interaction_pairs
                )
                truncated_ucj_op = ffsim.UCJOpSpinBalanced(
                    diag_coulomb_mats=ucj_op.diag_coulomb_mats[:-1],
                    orbital_rotations=ucj_op.orbital_rotations[:-1],
                    final_orbital_rotation=ucj_op.orbital_rotations[-1]
                )
                lucj_op_params = truncated_ucj_op.to_parameters(interaction_pairs=interaction_pairs_bal)
                lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
                    lucj_op_params,
                    norb=ncas,
                    n_reps=1,
                    interaction_pairs=interaction_pairs_bal,
                    with_final_orbital_rotation=True
                )

                lucj_circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)

    if sampling_circuit:
        lucj_circuit.measure_all()

    return lucj_circuit


@W4Decorators.process(basis="sto6g") # IF RAN WITH --process SQD WILL ITERATE OVER ALL MOLECULES
def sqd_example(species: str, molecule: Molecule):
    mol_basis: Basis = molecule.basis[W4.parameters.basis]
    ncas = mol_basis.ncas
    nelec_as = mol_basis.nelecas
    spin_sq = molecule.spin / 2 * (molecule.spin / 2 + 1)

    rand_seed = 1234

    # :: TO FIX ::
    # mf_as used in the circuit sampling, but if it's not an instance
    # of CASSCF, then the system terminates so just pass true (needs a better fix)
    lucj_circuit_sampling = get_LUCJ_qiskit_circuit(
        ncas,
        nelec_as,
        True,
        mol_basis.cct2,
        'ibm_rensselaer',
        'diagonal',
        sampling_circuit=True
    )

    # Sample the quantum circuit and get counts
    sampler = StatevectorSampler()
    sampler.run
    pub = (lucj_circuit_sampling)
    job = sampler.run([pub], shots=100_000)
    result = job.result()[0]
    counts = result.data.meas.get_counts()

    ######## The code below is SQD ###########

    num_elec_a = nelec_as[0]
    num_elec_b = nelec_as[1]
    open_shell = num_elec_a != num_elec_b

    hcore = mol_basis.h1e
    eri = mol_basis.h2e
    nuclear_repulsion_energy = mol_basis.ecore

    bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)

    # SQD options
    iterations = 15
    num_batches = [20]
    samples_sizes = [1]

    def compute_num_configs(norb, nelec_ab):
        nvirt_a = norb - nelec_ab[0]
        nvirt_b = norb - nelec_ab[1]
        nocc_a = nelec_ab[0]
        nocc_b = nelec_ab[1]
        fci_configs = (math.comb(norb, nelec_ab[0]), math.comb(norb, nelec_ab[1]))
        singles = (nocc_a * nvirt_a, nocc_b * nvirt_b)
        doubles = (math.comb(nocc_a, 2) * math.comb(nvirt_a, 2), math.comb(nocc_b, 2) * math.comb(nvirt_b, 2))
        return (fci_configs, singles, doubles)

    def get_configs_limit(norb, nelec_ab, multiplier):
        fci_configs, singles, doubles = compute_num_configs(norb, nelec_ab)

        open_shell = nelec_ab[0] == nelec_ab[1]

        # Compute total number of configurations for singles and doubles, plus HF determinant
        a_lim = singles[0] + doubles[0]
        b_lim = singles[1] + doubles[1]
        lim = a_lim + b_lim + 1

        print(f"CISD # configs: {lim}")

        # Take sqrt because the SCI size will be upper bounded by lim*lim
        lim = np.sqrt(lim)

        print(f"SQD # samples limit for CISD: {lim}")

        if open_shell:
            lim = lim * multiplier
        else:
            lim = lim * multiplier / 2  # Because for closed shell the left and right halves are merged, doubling number of alpha and beta configurations

        lim = math.ceil(min(max(fci_configs[0], fci_configs[1]), lim))  # Make sure we are not exceeding fci size

        print(f"Samples limit: {lim}     Open Shell:{open_shell}")

        lim = min([lim, fci_configs[0], fci_configs[1]])

        print(f"SQD samples limit: {lim}")

        # print(f'% of FCI size: { min( total**2 / (fci_configs[0]*fci_configs[1]) )}')
        print(f"SQD dim limit: {lim ** 2} \nFCI dim: {fci_configs[0] * fci_configs[1]}  {fci_configs}")

        return lim


    for _seed, n_batches in enumerate(num_batches):
        for samples_per_batch in samples_sizes:
            max_davidson_cycles = 80

            ci_limit = get_configs_limit(ncas, nelec_as, samples_per_batch)

            e_hist = np.zeros((iterations, n_batches))  # energy history
            s_hist = np.zeros((iterations, n_batches))  # spin history
            o_hist = np.zeros((iterations, n_batches, 2 * ncas)) - 1
            c_hist = np.zeros((iterations, n_batches, 5))  # alpha_beta_FCI_size history
            t_hist = np.zeros((iterations))  # alpha_beta_FCI_size history
            occupancies_bitwise = None  # orbital i corresponds to column i in bitstring matrix
            restart_iter = -1

            for iteration in range(restart_iter + 1, iterations):

                # Start timer
                tstart = time.time()

                print(
                    f"Starting configuration recovery iteration {iteration} for n_batches {n_batches} and batch_size {samples_per_batch}")
                # On the first iteration, we have no orbital occupancy information from the
                # solver, so we just post-select from the full bitstring set based on hamming weight.
                if occupancies_bitwise is None:
                    bs_mat_tmp = bitstring_matrix_full
                    probs_arr_tmp = probs_arr_full

                # If we have average orbital occupancy information, we use it to refine the full set of noisy configurations
                else:
                    bs_mat_tmp, probs_arr_tmp = recover_configurations(
                        bitstring_matrix_full,
                        probs_arr_full,
                        (occupancies_bitwise[:ncas], occupancies_bitwise[ncas:]),
                        num_elec_a,
                        num_elec_b,
                        rand_seed=rand_seed + iteration + _seed,
                    )

                # Throw out configurations with incorrect particle number in either the spin-up or spin-down systems
                batches = postselect_and_subsample(
                    bs_mat_tmp,
                    probs_arr_tmp,
                    hamming_right=num_elec_a,
                    hamming_left=num_elec_b,
                    samples_per_batch=ci_limit,
                    num_batches=n_batches,
                    rand_seed=rand_seed + iteration + _seed,
                )

                # Run eigenstate solvers in a loop. This loop should be parallelized for larger problems.
                e_tmp = np.zeros(n_batches)
                s_tmp = np.zeros(n_batches)
                occs_tmp = np.zeros((n_batches, 2 * ncas))
                c_tmp = np.zeros((n_batches, 5))
                coeffs = []

                for j in range(n_batches):
                    ci_strs = bitstring_matrix_to_ci_strs(batches[j], open_shell=open_shell)
                    print(f"Batch {j}: {(len(ci_strs[0]), len(ci_strs[1]))} configurations")
                    e_sci, sci_state, avg_occupancy, spin_squared = solve_fermion(
                        ci_strs,
                        hcore, eri, open_shell=open_shell,
                        spin_sq=spin_sq,
                        max_davidson=max_davidson_cycles,
                        verbose=0
                    )
                    e_sci += float(nuclear_repulsion_energy)
                    e_tmp[j] = e_sci
                    s_tmp[j] = spin_squared
                    c_tmp[j, :] = np.array(
                        [len(sci_state.ci_strs_a), len(sci_state.ci_strs_b), num_elec_a, num_elec_b, ncas])

                    occs_tmp[j, :ncas] = avg_occupancy[0]
                    occs_tmp[j, ncas:] = avg_occupancy[1]

                    coeffs.append(sci_state)
                print(e_tmp)

                # Combine batch results
                # avg_occupancies = np.mean(occs_tmp, axis=0)
                # # The occupancies from the solver should be flipped to match the bits in the bitstring matrix.
                # occupancies_bitwise = flip_orbital_occupancies(avg_occupancies)
                occupancies_bitwise = tuple(np.mean(occs_tmp, axis=0))

                # Dump coefficients
                # Dump energies
                # Dump spins
                # Dump occs
                # Write data handlers for these!
                # Extend the batched loop to be able to restart by loading coefficients

                tend = time.time()

                e_hist[iteration, :] = e_tmp
                s_hist[iteration, :] = s_tmp
                o_hist[iteration, :, :] = occs_tmp
                c_hist[iteration, :, :] = c_tmp
                t_hist[iteration] = tstart - tend

            print(
                ' ' * 8 + f'Completed {iterations} iterations for n_batches {n_batches} and batch_size {samples_per_batch}')


if __name__ == "__main__":
    W4.parameters.basis = "sto6g"
    W4.init()
    species = "alcl"
    sqd_example(species, W4[species])