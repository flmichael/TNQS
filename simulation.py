from pathlib import Path

from flax.core.frozen_dict import freeze

import jax
import jax.numpy as jnp
import jax.random as random

import jVMC
import jVMC.nets.activation_functions as jVMC_act

import os
import pickle

from src.nqs_wrapper import *
from src.networks import *
from src.timing import *
from src.iohandling import *
from src.model import *
from src.parameters import *
from src.plotting import *

class Simulation:
    """this class is a state class and does not follow the functional programming paradigm"""
    epoch_num: int = 0
    struct_num: int = 0
    param_updates: int = 0
    recent_update: int = 0

    def __init__(self, init_params, hyper_params=Hyperparameters(), init_obs=None, init_nqs=None):
        """"""
        self.init_params = init_params
        self.hyper_params = hyper_params

        if init_obs is not None:
            self.obs = init_obs
            if self.obs.epoch_nums != []:
                self.epoch_num = self.obs.epoch_nums[-1]
            if self.obs.param_update_counts != []:
                self.param_updates = self.obs.param_update_counts[-1]
            if self.obs.step_epochs != []:
                self.struct_num = len(self.obs.step_epochs)
        else:
            self.obs = Observables(self.hyper_params.var_tol_lower, self.hyper_params.var_batch_size)
        
        self.model = TransverseFieldIsingChainPBC(self.init_params.chain_l, self.init_params.g)
        
        if init_nqs is not None:
            tensor_dims = list(map(lambda x: x.shape, init_nqs.values()))
            _, oup, bond = tn_iob(tensor_dims)
        else:
            oup = self.init_params.init_oup()
            bond = self.init_params.init_bond()

        net = TNANN(
            vis_n=self.init_params.chain_l, 
            inp=self.init_params.inp, 
            oup=oup, 
            bond=bond, 
            sigma=self.hyper_params.sigma, 
            act_fun=jVMC_act.log_cosh, 
            transl_inv=self.init_params.transl_inv)
            
        self.nqs = jVMC.vqs.NQS(net, batchSize=self.hyper_params.batch_size, seed=self.init_params.seed)

        if init_nqs is not None:
            self.nqs(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, self.nqs.net.vis_n)))
            self.nqs.params = init_nqs

        self.init_samp_tdvp()

        self.stepper = jVMC.util.stepper.Euler(timeStep=self.hyper_params.learningrate)

        self.timer = Timer(max_time=self.init_params.max_time)
        self.timer.start()

    def init_samp_tdvp(self):
        """initializes sampler and tdvp equation, needs to be called after setting params of NQS"""
        nqs_sample = psi_sampler(self.nqs)
        self.sampler = jVMC.sampler.MCSampler(
            nqs_sample, 
            (self.init_params.chain_l,), 
            random.PRNGKey(4321), 
            updateProposer=jVMC.sampler.propose_spin_flip_Z2, 
            numChains=self.hyper_params.num_chains, 
            sweepSteps=self.init_params.chain_l, 
            numSamples=self.hyper_params.n_samples, 
            thermalizationSweeps=self.hyper_params.therm_sweeps)
        self.tdvp_eq = jVMC.util.tdvp.TDVP(
            self.sampler, 
            rhsPrefactor=1.0, 
            svdTol=self.hyper_params.svdTol, 
            diagonalShift=self.hyper_params.diagonalShift, 
            makeReal='real',
            )

    @classmethod
    def load(cls, sim_path):
        print("loading: ", sim_path)
        checkpoint_file = open(Path(sim_path.parent.parent, 'params', sim_path.stem + '.pkl'),'rb')
        checkpoint = pickle.load(checkpoint_file)
        checkpoint_file.close()
        nqs_npz = jnp.load(sim_path, allow_pickle=True)
        init_nqs = freeze({k: jnp.array(nqs_npz[k]) for k in nqs_npz.keys()})

        return Simulation(*checkpoint, init_nqs)
    
    @classmethod
    def load_pkl(cls, sim_path):
        """function for loading the former data format, not compatible between versions"""
        print("loading: ", sim_path)
        checkpoint_file = open(sim_path,'rb')
        checkpoint = pickle.load(checkpoint_file)
        checkpoint_file.close()

        return Simulation(checkpoint[0], checkpoint[1], checkpoint[3], checkpoint[2])

    def file_path(self, sim_name):
        return Path(
            'data',
            sim_name, 
            str(int(-10.0 * self.init_params.g)), 
            str(self.init_params.seed))

    def save_checkpoint(self, sim_name, cp_name=None):
        if cp_name is None:
            cp_name = str(self.struct_num)

        sim_checkpoint_path = Path(self.file_path(sim_name), 'sim', cp_name + '.npz')
        os.makedirs(sim_checkpoint_path.parent, exist_ok = True)

        jnp.savez(sim_checkpoint_path, **self.nqs.params)

        params_checkpoint_path = Path(self.file_path(sim_name), 'params', cp_name + '.pkl')
        os.makedirs(params_checkpoint_path.parent, exist_ok = True)

        pkl_save_data = [self.init_params, self.hyper_params, self.obs]

        with open(params_checkpoint_path, 'wb') as checkpoint:
            pickle.dump(pkl_save_data, checkpoint, pickle.HIGHEST_PROTOCOL)

    def update(self):
        delta_params, _ = self.stepper.step(
            0, 
            self.tdvp_eq, 
            self.nqs.get_parameters(), 
            hamiltonian=self.model.hamiltonian, 
            psi=self.nqs, 
            numSamples=None)
        self.nqs.set_parameters(delta_params)
        self.sampler.net.params = tn_to_mat_params(self.nqs)

        self.epoch_num += 1
        self.param_updates += self.nqs.net.total_parameters()
        self.recent_update += 1

        self.obs.update(
            self.epoch_num,
            self.param_updates,
            self.tdvp_eq,
            self.init_params.chain_l,
            self.model.exact_energy)

        self.timer.update()

    def end_sim(self):
        max_epochs_reached = self.recent_update >= self.init_params.max_epochs
        out_of_time = self.timer.out_of_time()

        return max_epochs_reached or out_of_time

    def continue_updating(self):
        cooldown_over = self.recent_update >= self.hyper_params.update_cooldown
        should_update = cooldown_over and self.obs.req_update()

        max_epochs_reached = self.recent_update >= self.init_params.max_epochs
        out_of_time = self.timer.out_of_time()

        if self.init_params.use_variance is False:
            return not (max_epochs_reached or out_of_time)
        else:
            if not (max_epochs_reached or out_of_time):
                return not should_update
            else:
                return False

    def simulate_step(self, sim_name=None):
        """should be able to handle arbitrary update orders"""
        self.update()
        self.obs.print_recent()
        
        self.timer.create_checkpoint()

        while(self.continue_updating()):
            self.update()
            self.obs.print_recent()

        self.timer.create_checkpoint()

        if sim_name is not None:
            self.save_checkpoint(sim_name)

    def increase_dims(self, new_bond=None, new_oup=None):
        """TODO: implement assertion for size increase"""
        self.struct_num += 1
        self.obs.update_step(self.epoch_num, self.param_updates)

        if new_bond is None:
            new_bond = self.init_params.bond_step(self.struct_num)
        if new_oup is None:
            new_oup = self.init_params.oup_step(self.struct_num)

        self.nqs = padded_tn_psi(self.nqs, new_oup, new_bond, self.hyper_params.reduced_sigma)
        self.init_samp_tdvp()
        self.recent_update = 0

    def simulate(self, sim_name=None):
        self.simulate_step(sim_name)

        while not self.timer.out_of_time() and self.struct_num < self.init_params.total_steps() - 1:
            self.increase_dims()
            self.simulate_step(sim_name)

        self.timer.print()

def get_dir_list_in(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(dir + name)]

def load_sim_list(name, g_str, sim_version=None):
    g_dir = 'data/' + name + '/' + g_str + '/'
    #g_dir = PurePath('data', name, g_str)
    seed_list = get_dir_list_in(g_dir)
    #sim_list = [Simulation.load(g_dir + seed + '/sim/' + str(len(os.listdir(g_dir + seed + '/')) - 1) + '.pkl') for seed in seed_list]
    sim_list = []

    for seed in seed_list:
        if sim_version is None:
            sim_name = str(len(os.listdir(g_dir + seed + '/sim/')) - 1)
        else:
            sim_name = sim_version
        
        sim_path = Path(g_dir, seed, 'sim', sim_name + '.npz')
        sim = Simulation.load(sim_path)
        sim_list.append(sim)

    return sim_list

def rel_energy(energy_diff, exact_energy, sites_n):
    reng = 1 - (energy_diff + exact_energy / sites_n) / (exact_energy / sites_n)

    return reng

def get_best_sim(sim_list):
    best_sim = sim_list[0]

    exact_energy = best_sim.model.exact_energy
    sites_n = best_sim.init_params.chain_l
    best_reng = rel_energy(best_sim.obs.eng_diff_avrg(200), exact_energy, sites_n)

    for sim in sim_list:
        exact_energy = sim.model.exact_energy
        sites_n = sim.init_params.chain_l
        reng = rel_energy(sim.obs.eng_diff_avrg(200), exact_energy, sites_n)
        print(reng)

        if reng < best_reng:
            best_sim = sim
            best_reng = reng

    return best_sim