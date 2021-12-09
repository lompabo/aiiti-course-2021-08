import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

figsize=(9,3)

# ==============================================================================
# Plotting functions
# ==============================================================================


def plot_target_histograms(data, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    nbins = 20
    # --- Susceptibles
    plt.subplot(211)
    plt.title('#survivors')
    plt.hist(data['survivors'], bins=nbins, density=True)
    # --- Infected
    plt.subplot(212)
    plt.title('#zombies')
    plt.hist(data['i_num'], bins=nbins, density=True)
    plt.tight_layout()
    plt.show()


def plot_variability_histograms(data, target, figsize=figsize, autoclose=True):
    sim_in = ['edge_ratio', 'inf_prob', 'act_rate', 'rec_rate', 'ds_rate', 'di_rate']
    gby = data[sim_in + [target]].groupby(sim_in)
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    nbins = 20
    # --- Susceptibles
    plt.subplot(211)
    plt.title(f'Distribution of maxima')
    plt.hist(gby.max(), bins=nbins, density=True)
    # --- Infected
    plt.subplot(212)
    plt.title(f'Distribution of minima')
    plt.hist(gby.min(), bins=nbins, density=True)
    plt.tight_layout()
    plt.show()


def plot_impacts(data, target, cmap='Blues', figsize=figsize, autoclose=True):
    in_bins = {'edge_ratio':5,
                'inf_prob':3,
                'act_rate':3,
                'rec_rate':3,
                'ds_rate':3,
                'di_rate':3}
    if autoclose:
        plt.close('all')
    plt.figure(figsize=(figsize[0], 1.2*figsize[1]))
    for i, in_name in enumerate(in_bins):
        plt.subplot(2, 3, i+1)
        plt.hist2d(data[in_name], data[target],
                bins=(in_bins[in_name], 20), cmap=cmap, density=True)
                # vmin=0, vmax=1)
        plt.xlabel(in_name)
        if i % 3 == 0:
            plt.ylabel(target)
    plt.tight_layout()


def plot_training_history(history, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.tight_layout()


def plot_pred_scatter(y_pred, y_true, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    alpha = max(0.1, min(.3, 1000 / len(y_true)))
    plt.scatter(y_pred, y_true, marker='.', alpha=alpha)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()


def print_model_metrics(nn, tr, ts, sim_in, sim_out):
    tr_pred = nn.predict(tr[sim_in])
    tr_mse = np.mean(np.square(tr_pred - tr[sim_out].values))
    tr_var = np.var(tr[sim_out].values)
    tr_r2 = 1 - tr_mse / tr_var

    ts_pred = nn.predict(ts[sim_in])
    ts_mse = np.mean(np.square(ts_pred - ts[sim_out].values))
    ts_var = np.var(ts[sim_out].values)
    ts_r2 = 1 - ts_mse / ts_var
    print(f'MSE: {tr_mse:.2f} (training) {ts_mse:.2f} (test)')
    print(f'R2: {tr_r2:.2f} (training) {ts_r2:.2f} (test)')
    return tr_pred, ts_pred



def plot_capped_standardization(figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot([-2, -1, 1, 2], [-1, -1, 1, 1])
    plt.xlabel('Natural scale variable')
    plt.ylabel('Standardized/normalized variable')
    plt.xticks([-2, -1, 1, 2], ['relaxed min', 'std/norm min', 'std/norm max', 'relaxed max'])
    plt.yticks([-1, 1], ['std/norm min', 'std/norm max'])
    plt.tight_layout()





# ==============================================================================
# Data Manipulation & Models
# ==============================================================================

def load_zombie_data(data_folder):
    data_fname = os.path.join(data_folder, 'za_data.csv')
    data = pd.read_csv(data_fname)
    return data


def prepare_for_training(data, tr_ts_ratio, pop_size, sim_in, sim_out):
    # Row indices
    snum = len(data)
    sidx = np.arange(snum, dtype=int)

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(sidx)

    # Define the separators
    sep = int(snum * tr_ts_ratio)
    sidx_tr = sidx[:sep]
    sidx_ts = sidx[sep:]

    # Split the dataset
    data_tr = data.iloc[sidx_tr]
    data_ts = data.iloc[sidx_ts]

    # Standardize inputs
    means_in = data_tr[sim_in].mean(axis=0)
    stds_in = data_tr[sim_in].std(axis=0)
    tr = data_tr.copy()
    ts = data_ts.copy()
    tr[sim_in] = (data_tr[sim_in] - means_in) / stds_in
    ts[sim_in] = (data_ts[sim_in] - means_in) / stds_in

    # Standardize output
    tr[sim_out] /= pop_size
    ts[sim_out] /= pop_size

    # Cast to float32
    tr = tr.astype(np.float32)
    ts = ts.astype(np.float32)

    return tr, ts, means_in, stds_in


def build_mlpregressor(in_shape, out_shape=1, hidden=[]):
    mdl = keras.Sequential()
    if len(hidden) == 0:
        mdl.add(layers.Dense(out_shape, activation='linear'))
    else:
        mdl.add(layers.Dense(hidden[0], activation='relu'))
        for h in hidden[1:]:
            mdl.add(layers.Dense(h, activation='relu'))
        mdl.add(layers.Dense(out_shape, activation='linear'))
    return mdl


def print_solution(mdl, sol, cost_var, mvars, cvars, xvars_n, yvars_n, effects, combinations):
    if sol is None:
        print('No solution found')
    else:
        print('=== SOLUTION DATA')
        print('Solution time: {:.2f} (sec)'.format(mdl.solve_details.time))
        print('Solver status: {}'.format(sol.solve_details.status))
        print('Survivors: {}'.format(sol[yvars_n[1]]))
        print('Zombies: {}'.format(sol[yvars_n[0]]))
        print('Cost: {}'.format(sol[cost_var]))
        print('Chosen measures:')
        for x, effect in zip(mvars, effects):
            if sol[x] > 0:
                print('* {}'.format(effect['name']))
        print('Applicable bonuses:')
        for x, combo in zip(cvars, combinations):
            if sol[x] > 0:
                print('* {}'.format(' + '.join(combo['deps'])))
        cstring = 'c({})'.format(', '.join('{:.3f}'.format(max(0, sol[x])) for x in xvars_n))
        print('Evaluation string: {}'.format(cstring))

# ==============================================================================
# Combinatorial model building
# ==============================================================================

def build_inout_vars(mdl, sim_in, sim_out, xmin, xmax):
    xvars, yvars = [], []
    for in_name, lb, ub in zip(sim_in, xmin, xmax): # input variables
        xvars.append(mdl.continuous_var(lb=lb, ub=ub, name=in_name))
    for out_name in sim_out: # output variables
        yvars.append(mdl.continuous_var(lb=-1.2, ub=1.2, name=out_name))
    return xvars, yvars


def build_measure_vars(mdl, effects, combinations):
    mvars, cvars, mmap = [], [], {}
    # Build one binary variable per measure
    for i, effect in enumerate(effects):
        mvar = mdl.binary_var(name=effect['name'])
        mvars.append(mvar)
        mmap[effect['name']] = i
    # Build one binary variable per combination
    for i, combo in enumerate(combinations):
        cvar = mdl.binary_var(name='-'.join(combo['deps']))
        cvars.append(cvar)
    return mvars, cvars, mmap


def build_dependencies(bkd, mdl, combinations, mvars, mmap, cvars):
    for i, combo in enumerate(combinations):
        ndeps = len(combo['deps'])
        tmp = [mvars[mmap[name]] for name in combo['deps']]
        mdl.add_constraint(ndeps * cvars[i] <= sum(tmp))


def build_nat_in(mdl, xvars, sim_in, xmin, xmax, means_in, stds_in, bkd, encode_pwl):
    xvars_n = []
    for i, (in_name, lb, ub) in enumerate(zip(sim_in, xmin, xmax)):
        # Build the natural scale variable
        mean, std = means_in[in_name], stds_in[in_name]
        lb_nat, ub_nat = lb * std + mean, ub * std + mean
        span = ub_nat - lb_nat
        xnat = mdl.continuous_var(lb=lb_nat-span, ub=ub_nat+span, name=in_name+'_nat')
        xvars_n.append(xnat)

        # Add the capping & standardization constraints
        nat_nodes = [lb_nat-span, lb_nat, ub_nat, ub_nat+span]
        std_nodes = [lb, lb, ub, ub]
        encode_pwl(bkd, mdl, xvars=[xvars_n[i], xvars[i]],
                        nodes=[nat_nodes, std_nodes], name=in_name)
    return xvars_n


def build_nat_out(mdl, yvars, sim_out, pop_size, bkd, encode_pwl):
    yvars_n = []
    for i, out_name in enumerate(sim_out):
        # Build the natural scale variable
        ynat = mdl.continuous_var(lb=0, ub=pop_size, name=out_name+'_nat')
        yvars_n.append(ynat)

        # Add the capping & standardization constraints
        norm_nodes = [yvars[i].lb, 0, 1, yvars[i].ub]
        nat_nodes = [0, 0, pop_size, pop_size]
        encode_pwl(bkd, mdl, xvars=[yvars[i], yvars_n[i]],
                        nodes=[norm_nodes, nat_nodes], name=out_name)
    return yvars_n


def build_measure_effect_csts(mdl, mvars, cvars, xvars_n,
        in_defaults, sim_in, effects, combinations):
    for i, in_name in enumerate(sim_in):
        # Effects to input
        coefs_m = [e[in_name] for j, e in enumerate(effects) if in_name in e]
        evars_m = [mvars[j] for j, e in enumerate(effects) if in_name in e]
        # Combinations to input
        coefs_c = [c[in_name] for j, c in enumerate(combinations) if in_name in c]
        evars_c = [cvars[j] for j, c in enumerate(combinations) if in_name in c]
        # Build the connection constraint
        mdl.add_constraint(xvars_n[i] == in_defaults[i] +
                mdl.scal_prod(evars_m + evars_c, coefs_m + coefs_c))


def build_measure_cost(mdl, mvars, effects):
    cost_var = mdl.continuous_var(name='cost')
    # Measures to cost
    coefs = [e['cost'] for i, e in enumerate(effects)]
    evars = [mvars[i] for i, e in enumerate(effects)]
    mdl.add_constraint(cost_var == mdl.scal_prod(evars, coefs))
    return cost_var
