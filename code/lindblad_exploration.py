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
    # Flag to use parameters in atomic units
    _use_au = True

    N = 5  # Number of plasmon levels

    # time step data in fs
    sim_start_time_fs = 0
    sim_end_time_fs = 2000
    N_steps = 100000

    ###### Conversion factors 
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

    # Debye to SI units
    debye_to_SI = 3.33564e-30 

    # Joule to eV
    SI_to_eV = 6.24150e18

    # scale simulation time parameters for au option
    sim_start_time_au = 0
    sim_end_time_au = sim_end_time_fs * femtoseconds_to_au


    #####################################################################
    # conditional to see which unit system we will use

    if _use_au:
        hbar = 1   # au
        sim_time = np.linspace(sim_end_time_au, sim_end_time_au, N_steps)   # au
        plasmon_energy_quantum = 2.042 * eV_to_au # au
        qdot_energy_quantum = 2.042 * eV_to_au # au
        efield_energy_quantum = 2.042 * eV_to_au # au

        interaction_energies = [0.0108 * eV_to_au, 0.0108 * eV_to_au]   # au
        qdot_damping_energy = 268e-9  * eV_to_au # au
        qdot_dephasing_energy = 0.00127  * eV_to_au # au
        plasmon_damping_energy = 0.150  * eV_to_au # au
        qdot_dipole_magnitude = 13.9 * debye_to_au # au
        plasmon_dipole_magnitude = 2990  * debye_to_au # au
        laser_intensity = 1.4e-6  # au, (intensity 0.1 MW/cm^2), see pg 4, column 2, line 6
        print(F'Performing Simulation in Atomic Units.  All parameters are in atomic units ')
        print(F'Propagation timestep is {sim_time[1]} a.u. of time')
        print(F'Total simulation time will be {sim_end_time_au} a.u. of time')
        print(F'Which is the same as {sim_end_time_au / femtoseconds_to_au} fs')

    else:

        hbar = 0.6582119565476324   # eV/fs
        sim_time = np.linspace(sim_start_time_fs, sim_end_time_fs, N_steps)   # fs 
        plasmon_energy_quantum = 2.042   # eV
        qdot_energy_quantum = 2.042  # eV
        efield_energy_quantum = 2.042   # eV

        interaction_energies = [0.0108, 0.0108]   # eV
        qdot_damping_energy = 268e-9   # eV
        qdot_dephasing_energy = 0.00127   # eV
        plasmon_damping_energy = 0.150   # eV
        qdot_dipole_magnitude = 13.9   # D
        plasmon_dipole_magnitude = 2990   # D
        laser_intensity = 1.4e-6  # a.u. (intensity 0.1 MW/cm^2), see pg 4, column 2, line 6
        print(F'Performing Simulation with energy in eV and time in fs.  All energy parameters are in eV')
        print(F'The dipole moments are currently in Debye and the electric field is in atomic units')
        print(F'We will convert the E * mu to eV in make_hamiltonian(static_part, dipole_part)')
        print(F'Propagation timestep is {sim_time[1]} fs of time')
        print(F'Total simulation time will be {sim_end_time_fs} fs of time')
        print(F'Which is the same as {sim_end_time_fs * femtoseconds_to_au} au')
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
        emission=np.sqrt(qdot_damping_rate) * qt.destroy(2),
        dephase=np.sqrt(2 * qdot_dephasing_rate) * qt.num(2)
    )

    ### JJF Comment: creating plasmon instance using parameters in atomic units
    plasmon = utils.System(
        qt.ket2dm(qt.basis(N, 0)),
        lower=qt.destroy(N),
        number=qt.num(N),
        damping=np.sqrt(plasmon_damping_rate) * qt.destroy(N)
    )

    ### MEW Comment: Print out the plasmon and quantum dot instances for comparison between a.u. and eV
    print(f'Plasmon Op Names: {plasmon.operator_names}')
    print(f'Quantum Dot Op Names: {quantum_dot.operator_names}')

    # combine however many quantum dot systems and the plasmon system into a multipart system
    # JJF Comment: quantum_dot and plasmon instances are using atomic units, also using interaction_energies_au
    # which is in atomic units
    system = reduce(mul, [copy.deepcopy(quantum_dot) for _ in interaction_energies] + [plasmon])
    # extract the subsystem operators from the multipart system
    qdots = system.as_list[:-1]
    plasmon = system.as_list[-1]

    # make a list of collapse operators
    collapse_operators = [qdot.emission for qdot in qdots] + [qdot.dephase for qdot in qdots] + [plasmon.damping]

    # construct static part of hamiltonian
    ### JJF Comment: updating energies to be in atomic units
    H_qdot = qdot_energy_quantum * sum([qdot.number for qdot in qdots])
    H_plasmon = plasmon_energy_quantum * plasmon.lower.dag() @ plasmon.lower
    H_interaction = 1
    for interaction_energy, qdot in zip(interaction_energies, qdots):
        H_interaction += interaction_energy * qdot.lower @ plasmon.lower.dag()
        H_interaction += interaction_energy * qdot.lower.dag() @ plasmon.lower
    H_static = (H_qdot + H_plasmon + H_interaction) / hbar

    # construct dipole operator
    ### JJF Comment: updating dipole operators to be in atomic units
    dipole_qdot = qdot_dipole_magnitude * sum([qdot.lower.dag() + qdot.lower for qdot in qdots])
    dipole_plasmon = plasmon_dipole_magnitude * (plasmon.lower.dag() + plasmon.lower)
    dipole_operator = (dipole_qdot + dipole_plasmon) / hbar


    def E(t):
        """ Return electric field strength at a point in time `t`. """
        return laser_intensity * np.cos(efield_frequency_au * t)

    def make_hamiltonian(static_part, dipole_part):
        """ Return a hamiltonian closure. """
        def hamiltonian(t):
            ### going to define a conversion factor for
            ### dipole_part * E(t) so that the resulting
            ### energy is in the same units as the rest of the
            ### simulation... if we are using a.u. it is already 
            ### taken care of, if we are using eV, then we need to do some work
            if _use_au:
                fac = 1
            else:
                fac = debye_to_SI * SI_to_eV / volts_over_meters_to_au

            return static_part - dipole_part * E(t) * fac
        return hamiltonian

    # make a hamiltonian function using QuTiP's representation of operators
    H = make_hamiltonian(H_static, dipole_operator)
    # evolve the system in time using QuTiP's master equation solver
    start = time.perf_counter()
    result = qt.mesolve(
        H,
        system.state,
        sim_time,
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
