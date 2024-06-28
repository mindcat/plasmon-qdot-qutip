from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from icecream import ic
import qutip as qt
import numpy as np
import time
import copy

import utils


if __name__ == '__main__':
    #####################################################################
    hbar = 0.6582119565476324   # eV/fs
    N = 5  # Number of plasmon levels
    sim_time = np.linspace(0, 2000, 1000)   # fs
    # sim_time = np.linspace(0, 250, 1000)
    # sim_time = np.linspace(0, 50, 1000)
    plasmon_energy_quantum = qdot_energy_quantum = efield_energy_quantum = 2.042   # eV
    interaction_energies = [0.0108, 0.0108]   # eV
    qdot_damping_energy = 268e-9   # eV
    qdot_dephasing_energy = 0.00127   # eV
    plasmon_damping_energy = 0.150   # eV
    qdot_dipole_magnitude = 13.9   # D
    plasmon_dipole_magnitude = 2990   # D
    laser_intensity = 1.38e-7 * 100   # a.u. (intensity 0.001 MW/cm^2)
    #####################################################################
    efield_frequency = efield_energy_quantum / hbar
    qdot_damping_rate = qdot_damping_energy / hbar
    qdot_dephasing_rate = qdot_dephasing_energy / hbar
    plasmon_damping_rate = plasmon_damping_energy / hbar
    #####################################################################

    # make a basic quantum dot system
    quantum_dot = utils.System(
        qt.ket2dm(qt.basis(2, 0)),
        lower=qt.destroy(2),
        number=qt.num(2),
        emission=np.sqrt(qdot_damping_rate) * qt.destroy(2),
        dephase=np.sqrt(2 * qdot_dephasing_rate) * qt.num(2)
    )
    # make a basic plasmon system
    plasmon = utils.System(
        qt.ket2dm(qt.basis(N, 0)),
        lower=qt.destroy(N),
        number=qt.num(N),
        damping=np.sqrt(plasmon_damping_rate) * qt.destroy(N)
    )

    # combine however many quantum dot systems and the plasmon system into a multipart system
    # system = quantum_dot * copy.deepcopy(quantum_dot) * plasmon
    system = quantum_dot * plasmon
    # extract the subsystem operators from the multipart system
    qdots = system.as_list[:-1]
    plasmon = system.as_list[-1]

    # make a list of collapse operators
    collapse_operators = [qdot.emission for qdot in qdots] + [qdot.dephase for qdot in qdots] + [plasmon.damping]

    # construct static part of hamiltonian
    H_qdot = qdot_energy_quantum * sum([qdot.number for qdot in qdots])
    H_plasmon = plasmon_energy_quantum * plasmon.lower.dag() @ plasmon.lower
    H_interaction = 1
    for interaction_energy, qdot in zip(interaction_energies, qdots):
        H_interaction += interaction_energy * qdot.lower @ plasmon.lower.dag()
        H_interaction += interaction_energy * qdot.lower.dag() @ plasmon.lower
    H_static = H_qdot + H_plasmon + H_interaction

    # construct dipole operator
    dipole_qdot = qdot_dipole_magnitude * sum([qdot.lower.dag() + qdot.lower for qdot in qdots])
    dipole_plasmon = plasmon_dipole_magnitude * (plasmon.lower.dag() + plasmon.lower)
    dipole_operator = dipole_qdot + dipole_plasmon

    # divide hamiltonian by hbar before calculations
    H_static /= hbar
    dipole_operator /= hbar

    def E(t):
        """ Return electric field strength at a point in time `t`. """
        return laser_intensity * np.cos(efield_frequency * t)

    def make_hamiltonian(static_part, dipole_part):
        """ Return a hamiltonian closure. """
        def hamiltonian(t):
            return static_part - dipole_part * E(t)
        return hamiltonian

    # make a hamiltonian function using QuTiP's representation of operators
    H = make_hamiltonian(H_static, dipole_operator)
    # evolve the system in time using QuTiP's master equation solver
    start = time.perf_counter()
    result2 = qt.mesolve(
        H,
        system.state,
        sim_time,
        c_ops=collapse_operators,
        e_ops=[qdots[0].number, plasmon.number]
    )
    print(f'integration time (QuTiP version): {time.perf_counter() - start}')
    qdot_expected_values, plasmon_expected_values = result2.expect

    # # make a hamiltonian function using the matrix representation of operators
    # H_matrix = make_hamiltonian(H_static.full(), dipole_operator.full())
    # # evolve the system in time using the matrix representation
    # start = time.perf_counter()
    # result = solve_ivp(
    #     utils.make_lindblad_equation(H_matrix, [np.asarray(op.full()) for op in collapse_operators]),
    #     (sim_time[0], sim_time[-1]),
    #     system.state.full().ravel(),
    #     t_eval=sim_time,
    #     # method='BDF',
    #     method='RK45',
    #     rtol=1e-6,
    #     atol=1e-9
    # )
    # print(f'integration time (matrix version): {time.perf_counter() - start}')
    # density_matrices = [vec.reshape(system.state.shape) for vec in result.y.T]
    # qdot_expected_values_mat = utils.expectation(qdots[0].number.full(), density_matrices)
    # plasmon_expected_values_mat = utils.expectation(plasmon.number.full(), density_matrices)
    # purity = [np.trace(mat @ mat) for mat in density_matrices]
    # # compare QuTiP version to explicit version
    # ic(np.max(np.abs(qdot_expected_values - qdot_expected_values_mat)))
    # ic(np.max(qdot_expected_values - qdot_expected_values_mat))
    # ic(np.max(np.abs(plasmon_expected_values - plasmon_expected_values_mat)))


    # line plot
    plt.figure(dpi=300)
    plt.plot(sim_time, qdot_expected_values, label='Quantum Dot', color='tab:blue')
    plt.plot(sim_time, plasmon_expected_values, label='Plasmon',  color='tab:orange')
    # plt.plot(sim_time, purity, label='purity (all)',  color='red')
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Time (fs)')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

    # try to replicate plot from paper
    plt.figure(figsize=(7/1.5, 5.11/1.5), dpi=600)
    plt.plot(sim_time[::25], qdot_expected_values[::25], label='Quantum Dot', color='tab:blue', marker='x',
             linestyle='None', markersize=4, markeredgewidth=1.5)
    plt.plot(sim_time[::25], plasmon_expected_values[::25], label='Plasmon',  color='tab:orange', marker='x',
             linestyle='None', markersize=4, markeredgewidth=1.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Time (fs)')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
