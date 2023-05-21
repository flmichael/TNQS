from typing import Sequence, Union
import os

import numpy as np
import jax.numpy as jnp

from src.networks import *

class Parameters:
    """super class for classes that need to be saved, e.g. input parameters, hyper parameters, observables,
    subclasses should only hold data types or sequences of data types"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        params_str = 'parameters given:'

        for key, value in self.__dict__.items():
            params_str += '\n' + key + ' = ' + str(value)

        return params_str

    #change implementation to iohandler
    def save(self, file_path):
        os.makedirs(file_path.parent, exist_ok=True)
        jnp.savez(file_path, **self.__dict__)

class Hyperparameters(Parameters):
    """component class, constant parameters shared by all simulations"""
    def __init__(self, **kwargs):
        self.sigma = 0.5 # for initialization
        self.sigma_fac = 1e-10 # scaling factor for sigma for padding

        self.learningrate = 1e-2

        self.svdTol = 1e-6 # cutoff for Moore-Penrose pseudo inverse
        self.diagonalShift = 10.0

        self.n_samples = 5000 # number of samples in each step of MC chain
        self.batch_size = 10
        self.num_chains = 100 # number of elements in chain
        self.therm_sweeps = 25 # number of sweeps for thermalization

        self.var_tol_lower = -1e-4 # lower boundary for the variance slope
        self.var_batch_size = 500 # number of previous results used to calculate the variance slope

        self.update_cooldown = 500 # minimum number of epochs between updates

        self.reduced_sigma = self.sigma * self.sigma_fac

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load(self, file_path):
        params_dict = jnp.load(file_path, allow_pickle=True)

        return Hyperparameters(**params_dict)

class InitializationParameters(Parameters):
    """component class, constant paramters shared by simulations within a package"""
    inp: Sequence[int]
    oup: Union[int, Sequence]
    bond: Union[int, Sequence]

    chain_l: int
    g: float

    transl_inv: bool
    seed: int

    use_variance: bool
    max_epochs: int

    max_time: int

    def static_oup(self):
        return isinstance(self.oup[0], int)
    
    def isrbm(self):
        return len(self.bond) == 0

    def static_bond(self):
        if len(self.bond) != 0:
            return isinstance(self.bond[0], int)
        else:
            return True

    def oup_step(self, i):
        if self.static_oup():
            return self.oup
        else:
            return self.oup[i]

    def bond_step(self, i):
        if self.isrbm():
            return self.bond
        else:
            if self.static_bond():
                return self.bond
            else:
                return self.bond[i]

    def init_oup(self):
        return self.oup_step(0)

    def init_bond(self):
        return self.bond_step(0)

    def total_steps(self):
        if not self.static_bond():
            return len(self.bond)
        elif not self.static_oup():
            return len(self.oup)
        else:
            return 1
        
    def compression_list(self):
        comp_list = []

        if self.static_bond():
            bond_list = [self.bond,]
        else:
            bond_list = self.bond

        for b in bond_list:
            comp_list.append(compression_rate(self.inp, self.oup, b))

        return comp_list
        
    @classmethod
    def load(self, file_path):
        params_dict = jnp.load(file_path, allow_pickle=True)

        return InitializationParameters(**params_dict)

class Observables(Parameters):
    """component class for Simulation that stores sequences of values observed after each step"""
    def __init__(self, var_tol_lower, var_batch_size, **kwargs):
        self.epoch_nums: Sequence[int] = []
        self.eng_diffs: Sequence[float] = []
        self.vars: Sequence[float] = []
        self.param_update_counts: Sequence[int] = []
        self.var_fits: Sequence[float] = []
        self.step_epochs: Sequence[int] = []
        self.step_param_updates: Sequence[int] = []

        self.var_tol_lower = var_tol_lower
        self.var_batch_size = var_batch_size

        for key, value in kwargs.items():
            setattr(self, key, value)

    def eng_diff_per_site(self, tdvp_eq, sites_n, exact_energy):
        return float(abs(jnp.real(tdvp_eq.ElocMean0) / sites_n - exact_energy / sites_n))

    def eng_diff_avrg(self, batch):
        return sum(self.eng_diffs[-batch:]) / batch
    
    def eng_diff_avrg_list(self, batch):
        eng_diff_list = []

        epoch_steps = self.step_epochs + [self.epoch_nums[-1],]

        for s in epoch_steps:
            eng_diff = sum(self.eng_diffs[(s - batch):s]) / batch
            eng_diff_list.append(eng_diff)

        return eng_diff_list

    def var_per_site(self, tdvp_eq, sites_n):
        return float(tdvp_eq.ElocVar0 / sites_n)

    def var_slope(self):
        if len(self.vars) < 2:
            return -1
        else:
            var_slice = jnp.array(self.vars[-min(self.var_batch_size, len(self.vars)):])
            epoch_slice = list(range(len(var_slice)))
            m, _ = np.polyfit(epoch_slice, jnp.log(var_slice), deg = 1)

            return m

    def req_update(self):
        slope = self.var_slope()
        req = slope > self.var_tol_lower and slope < 0.0

        return req

    def update(self, epoch_num, param_updates, tdvp_eq, sites_n, exact_energy):
        self.epoch_nums.append(epoch_num)
        self.eng_diffs.append(self.eng_diff_per_site(tdvp_eq, sites_n, exact_energy))
        self.vars.append(self.var_per_site(tdvp_eq, sites_n))
        self.param_update_counts.append(param_updates)
        self.var_fits.append(self.var_slope())

    def update_step(self, step_epoch, step_param_updates):
        self.step_epochs.append(step_epoch)
        self.step_param_updates.append(step_param_updates)

    def print(self, epoch=None, last=None):
        """prints the observables for a given epoch, will default to printing the entire history"""
        if epoch is None and last is None:
            epoch_range = self.epoch_nums
        elif epoch is not None and last is None:
            epoch_range = [epoch,]
        elif epoch is None and last is not None:
            epoch_range = self.epoch_nums[-last:]
        
        for i in epoch_range:
            print(
                self.epoch_nums[i - 1], 
                self.eng_diffs[i - 1], 
                self.vars[i - 1], 
                self.param_update_counts[i - 1], 
                self.var_fits[i - 1],
                sep='\t')

    def print_recent(self):
        self.print(last=1)

    def plot(self, dir):
        pass

    @classmethod
    def load(self, file_path):
        params_dict = jnp.load(file_path, allow_pickle=True)

        return Observables(**params_dict)
