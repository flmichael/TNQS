from math import prod
from typing import Callable, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn

import tensornetwork as tn
tn.set_default_backend("jax")

def init_tensor_var(rng, var, shape, dtype):
    """conforms to the Flax module initialization functions"""
    rng1, rng2 = random.split(rng)
    return (random.normal(rng1, shape, dtype = np.float64) + 1.0j * random.normal(rng2, shape, dtype = np.float64)) * var

def tn_dims(inp, oup, bond):
    """returns the dimensions of the tensors in a tensor network with given input, output and bond dimensions"""
    l = len(inp)
    b = len(bond)
    idxrange = [i for i in range(l)]
    tensor_dims = list(map(lambda x: [inp[x], oup[x]], idxrange))

    for i in range(l - 1):
        tensor_dims[i].append(bond[i])

    for i in range(1, l):
        tensor_dims[i].append(bond[i - 1])

    if l == b:
        tensor_dims[0].append(bond[-1])
        tensor_dims[-1].append(bond[-1])

    return tensor_dims

def tn_idcs(inp, bond):
    """assumes that the length of inp is equal to length of oup, which should always be the case anyway"""
    inp_l = len(inp)
    bond_l = len(bond)
    idxrange = [i for i in range(inp_l)]
    indices = list(map(lambda x: [-2 * (x + 1) + 1, -2 * (x + 1)], idxrange))

    for i in range(inp_l - 1):
        indices[i].append(inp_l - (i + 1))

    for i in range(1, inp_l):
        indices[i].append(inp_l - i)

    if inp_l == bond_l:
        indices[0].append(inp_l)
        indices[-1].append(inp_l)

    return indices

def tn_iob(tensor_dims):
    """TODO: check for correct behaviour for PBC"""
    t = len(tensor_dims)
    inp = tuple(map(lambda x: x[0], tensor_dims))
    oup = tuple(map(lambda x: x[1], tensor_dims))
    bond = []

    for tn in range(t - 1):
        bond.append(tensor_dims[tn][2])
    
    if len(tensor_dims[0]) == 4:
        bond.append(tensor_dims[0][2])

    return inp, oup, tuple(bond)

def tn_to_mat(tensors, indices, inp, oup):
    """contracts a tensor network and returs the resulting weight matrix"""
    mat = tn.ncon(tensors, indices)
    inp_n = prod(inp)
    oup_n = prod(oup)
    mat = jnp.reshape(mat, (oup_n, inp_n))

    return mat

def total_parameters(inp, oup, bond):
    """returns the total number of trainables parameters in a tensor network"""
    return sum(map(prod, tn_dims(inp, oup, bond)))

def compression_rate(inp, oup, bond):
    """returns the compression rate of the network in comparism to a full NN with the same alpha"""
    inp_n = prod(inp)
    oup_n = prod(oup)
    base_n = inp_n * oup_n
    param_n = total_parameters(inp, oup, bond)

    return param_n/base_n

def max_bond(inp, oup, periodic=False):
    bond = []

    for i in range(1, len(inp)):
        prod_left = prod(inp[:i]) * prod(oup[:i])
        prod_right = prod(inp[i:]) * prod(oup[i:])
        bond.append(min(prod_left, prod_right))

    return tuple(bond)

def new_dims(inp, oup, bond, add_to_oup = None, add_to_bond = None):
    """returns the new oup and bond dimensions either increased by specified amount or decided by max bond dimension (for boundary tensor)"""
    if add_to_oup != None:
        new_oup = tuple(map(lambda x, y: x + y, oup, add_to_oup))
    if add_to_bond != None:
        new_bond = tuple(map(lambda x, y: x + y, bond, add_to_bond))
        
    if add_to_oup == None and add_to_bond == None:
        if bond[0] < inp[0] * oup[0]:
            new_oup = oup
            new_bond = tuple(map(lambda x: x + 1, bond))
        else:
            new_oup = tuple(map(lambda x: x + 1, oup))
            new_bond = bond

    return new_oup, new_bond

class CpxRBM(nn.Module):
    """Restricted Boltzmann Machine with complex parameters based on FLAX module class"""
    vis_n: int
    inp_n: int
    oup_n: int
    sigma: float
    act_fun: Callable
    transl_inv: bool

    def setup(self):
        self.weights_dims = (self.oup_n, self.inp_n)
        self.weights = self.param('weights', init_tensor_var, self.sigma, self.weights_dims, np.complex128)

    def __call__(self, vis_states):
        vis_states = 2 * vis_states - 1

        if not self.transl_inv:
            return self.feature_detection(vis_states)
        elif self.transl_inv:
            sym_vis_states = jnp.vstack(list(map(lambda x: jnp.concatenate((vis_states[x:], (vis_states[:x])))[:self.inp_n], range(self.vis_n - 1))))
            feature_states = jax.vmap(self.feature_detection)(sym_vis_states)

        return sum(feature_states)

    def feature_detection(self, vis_state):
        return jnp.sum(self.act_fun(jnp.matmul(self.weights, vis_state)))

class TNANN(nn.Module):
    """Tensor-Network Artificial Neural Network based on FLAX module class"""
    vis_n: int
    inp: Sequence[int]
    oup: Sequence[int]
    bond: Sequence[int]
    sigma: float
    act_fun: Callable
    transl_inv: bool

    def setup(self):
        self.tensor_n = len(self.inp)
        self.inp_n = prod(self.inp)
        self.oup_n = prod(self.oup)
        self.bond_n = len(self.bond)
        self.tensor_dims = self.tn_dims()
        self.indices = self.tn_idcs()
        self.tensors = list(map(lambda x: self.param('tensor' + str(x), init_tensor_var, self.var_calc(x), self.tensor_dims[x], np.complex128), range(self.tensor_n)))

    def __call__(self, vis_states):
        vis_states = 2 * vis_states - 1

        if not self.transl_inv:
            return self.feature_detection(vis_states)
        elif self.transl_inv:
            sym_vis_states = jnp.vstack(list(map(lambda x: jnp.concatenate((vis_states[x:], (vis_states[:x])))[:self.inp_n], range(self.vis_n - 1))))
            feature_state = jax.vmap(self.feature_detection)(sym_vis_states)

        return jnp.sum(feature_state)

    def feature_detection(self, vis_state):
        weight_mat = self.tn_to_mat()
        feature_state = jnp.sum(self.act_fun(jnp.matmul(weight_mat, vis_state)))

        return feature_state

    def tn_dims(self):
        return tn_dims(self.inp, self.oup, self.bond)

    def tn_idcs(self):
        return tn_idcs(self.inp, self.bond)

    def tn_to_mat(self):
        """returns the weight matrix resulting from tensor contraction of the tensor network over the bond dimensions"""
        return tn_to_mat(self.tensors, self.indices, self.inp, self.oup)

    def var_calc(self, i):
        """calculates the initial variance for each tensor in the network"""
        if len(self.bond) == 0 or self.sigma == 0: return self.sigma

        n = len(self.inp)
        D = sum(list(self.bond)) / len(self.bond)
        numerator = jnp.product(jnp.array(self.inp)) * self.sigma / (self.inp[i] ** (2 * n - 1))

        if i == 0 or i == n - 1:
            denomenator = D
        else:
            denomenator = D ** (2 * n)

        return 1 / self.oup[i] * (numerator / denomenator) ** (1 / (2 * n - 1))

    def total_parameters(self):
        return total_parameters(self.inp, self.oup, self.bond)

    def compression_rate(self):
        return compression_rate(self.inp, self.oup, self.bond)

    def new_dims(self, add_to_oup = None, add_to_bond = None):
        return new_dims(self.inp, self.oup, self.bond, add_to_oup, add_to_bond)
