from src.simulation import *

from pathlib import PurePath
import jax.numpy as jnp

import os
import pickle

def save_dict(dict, npz_path):
    """saves a dictionary as an npz file"""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(npz_path, **dict)

def load_dict(npz_path):
    """loads a dictionary from an npz file"""
    return jnp.load(npz_path, allow_pickle=True)

def get_dir_list(dir):
    return [child for child in dir.iterdir() if child.is_dir()]

def save_sim(sim, npz_path):
    save_data = sim.save_data()
    save_dict(save_data, npz_path)

def load_sim(npz_path):
    load_data = load_dict(npz_path)

class IOHandler():
    def __init__(self, name):
        self.name = name
        
    def npz_path(self, file_path, name):
        return file_path / (name + '.npz')

def load_g(dir, g):
    g_dir = dir + '/' + str(int(-10.0 * g)) + '/'
    seed_list = get_dir_list_in(g_dir)
    #return [Simulation.load(str(len(get_dir_list_in(g_dir + s))), g_dir + s) for s in seed_list]
    return [Simulation.load(str(3), g_dir + s) for s in seed_list]

def load_g_config(dir, g, num):
    g_dir = dir + '/' + str(int(-10.0 * g)) + '/'
    seed_list = get_dir_list_in(g_dir)
    sim_list = []

    for s in seed_list:
        try: 
            sim = Simulation.load(str(num), g_dir + s)
            sim_list.append(sim)
        except:
            pass

    return sim_list

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_snapshot(dir, g, seed, n):
    sim_dir = dir + '/' + str(int(-10.0 * g)) + '/' + str(seed) + '/'
    return Simulation.load(str(n), sim_dir)

