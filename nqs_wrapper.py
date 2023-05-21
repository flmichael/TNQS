from src.networks import *

from flax.core.frozen_dict import freeze

import jVMC

def replace_weights(psi, weights_params):
    """returns a CpxRBM psi with updated weights parameters"""
    new_net = CpxRBM(
        psi.net.vis_n, 
        psi.net.inp_n, 
        psi.net.oup_n, 
        psi.net.sigma, 
        psi.net.act_fun, 
        psi.net.transl_inv)
    new_psi = jVMC.vqs.NQS(new_net, batchSize=psi.batchSize, seed=psi.seed)
    new_psi(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, psi.net.vis_n)))
    new_psi.params = weights_params

    return new_psi

def replace_tensors(psi, tensors):
    """returns a TNANN psi with updated tensors"""
    new_net = TNANN(
        psi.net.vis_n, 
        psi.net.inp, 
        psi.net.oup, 
        psi.net.bond, 
        psi.net.sigma, 
        psi.net.act_fun, 
        psi.net.transl_inv)
    new_psi = jVMC.vqs.NQS(new_net, batchSize=psi.batchSize, seed=psi.seed)
    new_psi(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, psi.net.vis_n)))
    new_psi.params = freeze(dict(zip(psi.params.keys(), tensors)))

    return new_psi

def padded_tn_psi(psi, new_oup, new_bond, reduced_sigma):
    """returns a TNANN psi with padded tensors"""
    new_net = TNANN(
        psi.net.vis_n, 
        psi.net.inp, 
        new_oup, 
        new_bond, 
        reduced_sigma, 
        psi.net.act_fun, 
        psi.net.transl_inv)
    new_psi = jVMC.vqs.NQS(new_net, batchSize=psi.batchSize, seed=psi.seed)
    new_psi(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, psi.net.vis_n)))
    new_values = map(lambda old_tensor, new_tensor: jax.lax.dynamic_update_slice(new_tensor, old_tensor, tuple(jnp.zeros(old_tensor.ndim, int))), list(psi.params.values()), list(new_psi.params.values()))
    new_psi.params = freeze(dict(zip(psi.params.keys(), new_values)))

    return new_psi

def tn_to_mat_params(psi):
    param_weights = psi.net.apply({'params': psi.params}, method = psi.net.tn_to_mat)
    return freeze({'weights': param_weights})

def psi_sampler(psi):
    """uses a TNANN to create a CpxRBM for sampling"""
    psi(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, psi.net.vis_n)))
    weights_params = tn_to_mat_params(psi)
    new_net = CpxRBM(
        psi.net.vis_n, 
        prod(psi.net.inp), 
        prod(psi.net.oup), 
        psi.net.sigma, 
        psi.net.act_fun, 
        psi.net.transl_inv)
    new_psi = jVMC.vqs.NQS(new_net, batchSize=psi.batchSize, seed=psi.seed)
    new_psi(jnp.zeros((jVMC.global_defs.myDeviceCount, 1, new_psi.net.vis_n)))
    new_psi.params = weights_params

    return new_psi