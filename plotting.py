from src.iohandling import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Plotter:
    """holds the parameters necessary to plot data, can be given to e.g. observable classes to plot them"""
    pass

def single_g_point(sim, batch):
    return sum(np.array(sim.res)[-batch:, 1]) / batch

def g_point(sim_list, batch):
    """returns the average difference between exact and jVMC energy, over a given batch size"""
    return sum(map(lambda sim: single_g_point(sim, batch), sim_list)) / len(sim_list)

def min_g_point(sim_list, batch):
    """returns the minimal average difference between exact and jVMC energy, over a given batch size"""
    g_points = list(map(lambda sim: single_g_point(sim, batch), sim_list))
    return np.min(g_points)

def plot_simulations(sim_list, dir, show_updates = False, plot_type = 'epochs'):
    """plots and saves a list of simulations"""
    if plot_type == 'epochs':
        res_col = 0
    elif plot_type == 'updates':
        res_col = 3

    fig, ax = plt.subplots(2, 1, sharex = True, figsize = [4.8, 4.8])

    for sim in sim_list:
        res = np.array(sim.res)
        ax[0].semilogy(res[:, res_col], res[:, 1], '-')
        ax[1].semilogy(res[:, res_col], res[:, 2], '-')

        if show_updates:
            for p in range(2):
                for e, u in sim.dim_updates:
                    if plot_type == 'epochs':
                        ax[p].axvline(x = float(e), color = 'red', linewidth = 0.5, linestyle = ':')
                    elif plot_type == 'updates':
                        ax[p].axvline(x = float(u), color = 'red', linewidth = 0.5, linestyle = ':')

    ax[0].set_ylabel(r'$(E-E_0)/L$')
    ax[1].set_ylabel(r'Var$(E)/L$')

    plt.xlabel(plot_type)
    plt.tight_layout()

    plot_path = dir + '/plot_' + plot_type + '.pdf'
    plt.savefig(plot_path, transparent = True)

def plot_sim_dir(dir, g, show_updates = True):
    """plots all most recent simulations in the directory"""
    sim_list = load_g(dir, g)
    
    for sim in sim_list:
        for r in sim.res:
            print(*r, sep = '\t')

    plot_simulations(sim_list, dir + '/' + str(int(-10.0 * g)), show_updates)

def plot_sim_dirs(dir, g_list):
    for g in g_list:
        plot_sim_dir(dir, g, False)

def plot_phasediagram(dir, g_list):
    """plots the phase diagram for all existing values of g in the directory"""
    g_data_points = []
    
    for g in g_list:
        sim_list = load_g(dir, g)
        g_data_point = g_point(sim_list, 10)
        print(g_data_point)
        g_data_points.append(g_data_point)

        for sim in sim_list:
            print(sim.seed)
            print(*sim.res[-1], sep = '\t')

    fig, ax = plt.subplots(1, 1, sharex = True, figsize = [4.8, 4.8])
    ax.semilogy(g_list, g_data_points, '-')
    
    ax.set_ylabel(r'$(E-E_0)/L$')
    plt.xlabel(r'g')
    plt.tight_layout()

    plot_path = dir + '/phasediag' + '.pdf'
    plt.savefig(plot_path, transparent = True)

def plot_full_phasediagrams(dir, g_list, num = 10):
    """plots the phase diagram for all existing values of g in the directory"""
    for n in range(num):
        g_data_points = []

        for g in g_list:
            sim_list = load_g_config(dir, g, n)
            g_data_point = g_point(sim_list, 10)
            print(g_data_point)
            g_data_points.append(g_data_point)

            for sim in sim_list:
                print(sim.seed)
                print(*sim.res[-1], sep = '\t')

        fig, ax = plt.subplots(1, 1, sharex = True, figsize = [9.6, 9.6])
        ax.semilogy(g_list, g_data_points, '-')

        ax.set_ylabel(r'$(E-E_0)/L$')
        plt.xlabel(r'g')
        plt.tight_layout()

        plot_path = dir + '/phasediag'
        plt.savefig(plot_path + '_' + str(n) + '.pdf', transparent = True)

def plot_weights(dir, g, seed, n):
    sim = load_snapshot(dir, g, seed, n)
    weights_real = sim.psi_sampler.params['weights'].real
    weights_imag = sim.psi_sampler.params['weights'].imag

    plt_dir = dir + '/' + str(int(-10.0 * g)) + '/' + str(seed) + '/'
    plt.imshow(weights_real, cmap='coolwarm', norm=colors.LogNorm(), aspect='equal', interpolation='nearest')
    plt.savefig(plt_dir + str(n) + '_weights_real.pdf', transparent=True)
    plt.imshow(weights_imag, cmap='coolwarm', norm=colors.LogNorm(), aspect='equal', interpolation='nearest')
    plt.savefig(plt_dir + str(n) + '_weights_imag.pdf', transparent=True)

def plot_energy(sim_list, dir, show_updates=False, plot_type='epochs'):
    """plots and saves a list of simulations"""
    fig, ax = plt.subplots(2, 1, sharex = True, figsize = [4.8, 4.8])

    for sim in sim_list:
        if plot_type == 'epochs':
            x_vals = sim.obs.epoch_nums
        elif plot_type == 'updates':
            x_vals = sim.obs.param_update_counts

        #sim.obs.print()

        y0_vals = sim.obs.eng_diffs
        #print(y0_vals[-1])
        #print(*y0_vals, sep='\n')
        y1_vals = sim.obs.vars

        ax[0].semilogy(x_vals, y0_vals, '-')
        ax[1].semilogy(x_vals, y1_vals, '-')

        if show_updates:
            for p in range(2):
                if plot_type == 'epochs':
                    for u in sim.obs.step_epochs:
                        ax[p].axvline(x = float(u), color = 'red', linewidth = 0.5, linestyle = ':')
                elif plot_type == 'updates':
                    for u in sim.obs.step_param_updates:
                        ax[p].axvline(x = float(u), color = 'red', linewidth = 0.5, linestyle = ':')

    ax[0].set_ylabel(r'$(E-E_0)/L$')
    ax[1].set_ylabel(r'Var$(E)/L$')

    plt.xlabel(plot_type)
    plt.tight_layout()

    plot_path = dir + 'plot_' + plot_type + '.pdf'
    plt.savefig(plot_path, transparent = True)

def plot_energy_compression(sim_list, dir):
    """plots and saves a list of simulations"""
    fig, ax = plt.subplots(1, 1, sharex = True, figsize = [4.8, 4.8])

    x_vals = sim_list[0].init_params.compression_list()

    for sim in sim_list:
        y_vals = sim.obs.eng_diff_avrg_list(200)

        ax.semilogy(x_vals, y_vals, 'o')

    ax.set_ylabel(r'$(E-E_0)/L$')

    plt.xlabel('compression')
    plt.tight_layout()

    plot_path = dir + 'plot_compression.pdf'
    plt.savefig(plot_path, transparent = True)