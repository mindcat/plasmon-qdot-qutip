from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from icecream import ic
import qutip as qt
import numpy as np
import time
import copy
from functools import reduce
from operator import mul
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils

# TODO: interesting params:
#       qdot_dipole_magnitude = 0 (doesn't change much)
#       qdot_dipole_magnitude = plasmon_dipole_magnitude (ringdown, qdot and plasmon "vibrate" faster than E field)
#       interaction_energies = [0.0108, 0.0108*2]
#       interaction_energies = [0.0108, 0.0108*2] & efield_energy_quantum = 2.032
#       laser_intensity = 1.38e-7 * 1000
if __name__ == '__main__':
    ###### Conversion factors to atomic units
    # eV to atomic units 
    eV_to_au = 1 / 27.211396 

    # Debye to atomic units
    debye_to_au = 1 / 2.541746473

    # Seconds to atomic units
    seconds_to_au = 1 / 2.4188843265864e-17

    # fs to atomic units
    femtoseconds_to_au = 1e-15 / 2.4188843265864e-17

    # Electric field in V/m to atomic units
    volts_over_meters_to_au = 1 / 5.14220652e11


    #####################################################################
    # hbar = 0.6582119565476324   # eV/fs
    hbar = 1   # JJF Note: this is the value in atomic units eV/fs
    N = 5  # Number of plasmon levels
    # sim_time = np.linspace(0, 2000, 1000)   # fs
    # sim_time = np.linspace(0, 2000, 5000)   # fs
    sim_time = np.linspace(0, 2000, 100000)   # fs
    # sim_time = np.linspace(0, 2000, 1000000)   # fs
    # plasmon_energy_quantum = qdot_energy_quantum = efield_energy_quantum = 2.042   # eV
    plasmon_energy_quantum = 2.042   # eV
    qdot_energy_quantum = 2.042  # eV
    efield_energy_quantum = 2.042   # eV

    interaction_energies = [0.0108, 0.0108]   # eV
    qdot_damping_energy = 268e-9   # eV
    qdot_dephasing_energy = 0.00127   # eV
    plasmon_damping_energy = 0.150   # eV
    qdot_dipole_magnitude = 13.9   # D
    plasmon_dipole_magnitude = 2990   # D
    laser_intensity = 1.38e-7  # * 100   # a.u. (intensity 0.001 MW/cm^2)

    ####################################
    # Convert all values to atomic units
    ####################################
    sim_time_au = sim_time * femtoseconds_to_au
    plasmon_energy_quantum_au = plasmon_energy_quantum * eV_to_au
    qdot_energy_quantum_au = qdot_energy_quantum * eV_to_au
    efield_energy_quantum_au = efield_energy_quantum * eV_to_au

    interaction_energies_au = [0.0108 * eV_to_au, 0.0108 * eV_to_au]
    qdot_damping_energy_au = qdot_damping_energy * eV_to_au
    qdot_dephasing_energy_au = qdot_dephasing_energy * eV_to_au
    plasmon_damping_energy_au = plasmon_damping_energy * eV_to_au

    qdot_dipole_magnitude_au = qdot_dipole_magnitude * debye_to_au
    plasmon_dipole_magnitude_au = plasmon_dipole_magnitude * debye_to_au
    laser_intensity_au = laser_intensity #* volts_over_meters_to_au



    #####################################################################
    ### JJF Comment: Sus of these as rates, hbar is in atomic units but these 
    ### energies are in eV.  The frequencies / rates are therefore in mixed unit systems
    efield_frequency = efield_energy_quantum / hbar
    qdot_damping_rate = qdot_damping_energy / hbar
    qdot_dephasing_rate = qdot_dephasing_energy / hbar
    plasmon_damping_rate = plasmon_damping_energy / hbar
    #####################################################################

    ########## 
    ### JJF Commment: now with the energies in atomic units, we can identify rates or frequencies in atomic units
    ### and they will have the same magnitudes as their respective energies
    efield_frequency_au = efield_energy_quantum_au / hbar
    qdot_damping_rate_au = qdot_damping_energy_au / hbar
    qdot_dephasing_rate_au = qdot_dephasing_energy_au / hbar
    plasmon_damping_rate_au = plasmon_damping_energy_au / hbar

    print(F'QD energy {qdot_energy_quantum_au}')
    print(F'QD damping energy {qdot_damping_energy_au}')
    print(F'QD dephasing energy {qdot_dephasing_energy_au}')

    print(F'Plasmon energy {plasmon_energy_quantum_au}')
    print(F'Plasmon damping energy {plasmon_damping_energy_au}')

    print(F'Efield energy {efield_energy_quantum_au}')
    
    print(F'Sim time in fs t0 and t1 {sim_time[0]} {sim_time[1]}')
    print(F'Sim time in au t0 and t1 {sim_time_au[0]} {sim_time_au[1]}')
    print(F'Laser magnitude {laser_intensity_au}')

    print(F'QDot dipole {qdot_dipole_magnitude_au}')
    print(F'Plasmon dipole {plasmon_dipole_magnitude_au}')
    print(F'Interaction energies {interaction_energies_au[0]}')

    # make a basic quantum dot system
    ### JJF Comment: I am creating this instance with the parameters in atomic units now
    quantum_dot = utils.System(
        qt.ket2dm(qt.basis(2, 0)),
        lower=qt.destroy(2),
        number=qt.num(2),
        emission=np.sqrt(qdot_damping_rate_au) * qt.destroy(2),
        dephase=np.sqrt(2 * qdot_dephasing_rate_au) * qt.num(2)
    )

    ### JJF Comment: creating plasmon instance using parameters in atomic units
    plasmon = utils.System(
        qt.ket2dm(qt.basis(N, 0)),
        lower=qt.destroy(N),
        number=qt.num(N),
        damping=np.sqrt(plasmon_damping_rate_au) * qt.destroy(N)
    )

    # combine however many quantum dot systems and the plasmon system into a multipart system
    # JJF Comment: quantum_dot and plasmon instances are using atomic units, also using interaction_energies_au
    # which is in atomic units
    system = reduce(mul, [copy.deepcopy(quantum_dot) for _ in interaction_energies_au] + [plasmon])
    # extract the subsystem operators from the multipart system
    qdots = system.as_list[:-1]
    plasmon = system.as_list[-1]

    # make a list of collapse operators
    collapse_operators = [qdot.emission for qdot in qdots] + [qdot.dephase for qdot in qdots] + [plasmon.damping]

    # construct static part of hamiltonian
    ### JJF Comment: updating energies to be in atomic units
    H_qdot = qdot_energy_quantum_au * sum([qdot.number for qdot in qdots])
    H_plasmon = plasmon_energy_quantum_au * plasmon.lower.dag() @ plasmon.lower
    H_interaction = 1
    for interaction_energy, qdot in zip(interaction_energies_au, qdots):
        H_interaction += interaction_energy * qdot.lower @ plasmon.lower.dag()
        H_interaction += interaction_energy * qdot.lower.dag() @ plasmon.lower
    H_static = (H_qdot + H_plasmon + H_interaction) / hbar

    # construct dipole operator
    ### JJF Comment: updating dipole operators to be in atomic units
    dipole_qdot = qdot_dipole_magnitude_au * sum([qdot.lower.dag() + qdot.lower for qdot in qdots])
    dipole_plasmon = plasmon_dipole_magnitude_au * (plasmon.lower.dag() + plasmon.lower)
    dipole_operator = (dipole_qdot + dipole_plasmon) / hbar


    def E(t):
        """ Return electric field strength at a point in time `t`. """
        return laser_intensity * np.cos(efield_frequency_au * t)

    def make_hamiltonian(static_part, dipole_part):
        """ Return a hamiltonian closure. """
        def hamiltonian(t):
            return static_part - dipole_part * E(t)
        return hamiltonian

    # make a hamiltonian function using QuTiP's representation of operators
    H = make_hamiltonian(H_static, dipole_operator)
    # evolve the system in time using QuTiP's master equation solver
    start = time.perf_counter()
    result = qt.mesolve(
        H,
        system.state,
        sim_time_au,
        c_ops=collapse_operators,
        e_ops=system.number + [lambda t, state: np.real((state @ state).tr())]
    )
    print(f'Integration time (QuTiP): {time.perf_counter() - start}')

    # line plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
    for i, expectation_vals in enumerate(result.expect[:len(qdots)]):
        fig.add_trace(go.Scatter(x=sim_time_au, y=expectation_vals, name=f'Quantum Dot {i}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim_time_au, y=result.expect[len(qdots)], name='Plasmon'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=sim_time, y=E(sim_time)*1000, name='E Field (x1000)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim_time_au, y=result.expect[-1], name="System Purity"), row=2, col=1)
    fig.update_yaxes(title_text=r'$\langle N \rangle$', row=1, col=1)
    fig.update_yaxes(title_text='Purity', row=2, col=1)
    fig.update_xaxes(title_text='Time (au)', row=2, col=1)
    fig.update_layout(margin=dict(l=20, r=20, t=5, b=20))
    fig.show()


    b = ''
