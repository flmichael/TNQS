from src.nqs_wrapper import *

import jVMC.operator.branch_free as jVMC_ops

def tfi_pbc_hamiltonian(chain_l, g):
    """returns the transverse-field Ising hamiltonian with periodic boundary conditions for a chain lattice with a given length and g"""
    hamiltonian = jVMC.operator.BranchFreeOperator()
    for l in range(chain_l):
        hamiltonian.add(jVMC.operator.scal_opstr(-1.0, (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % chain_l))))
        hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))

    return hamiltonian

def energy_single_p_mode(h_t, P):
    return np.sqrt(1 + h_t ** 2 + 2 * h_t * np.cos(P))

def ground_state_energy(N, h_t):
    """returns the exact ground state energy for a transverse-field Ising chain with given length and h"""
    Ps = 0.5 * np.arange(- (N - 1), N - 1 + 2, 2)
    Ps = Ps * 2 * np.pi / N
    energies_p_modes = np.array([energy_single_p_mode(h_t, P) for P in Ps])
    return - np.sum(energies_p_modes)

class Model:
    """super class for physical models like e.g. transverse-field Ising chain"""
    hamiltonian: jVMC_ops.BranchFreeOperator
    exact_energy: float

class TransverseFieldIsingChainPBC(Model):
    def __init__(self, chain_l, g):
        self.hamiltonian = tfi_pbc_hamiltonian(chain_l, g)
        self.exact_energy = ground_state_energy(chain_l, g)

    def energy_single_p_mode(self, h_t, P):
        return energy_single_p_mode(h_t, P)

    def ground_state_energy(N, h_t):
        return ground_state_energy(N, h_t)