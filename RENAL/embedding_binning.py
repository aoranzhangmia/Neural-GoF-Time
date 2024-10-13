import numpy as np
import torch
import torch.nn as nn
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import random
import itertools
from IPython.display import clear_output
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy.ndimage import laplace
from scipy.spatial import distance
from astropy.stats import bayesian_blocks, knuth_bin_width


def calculate_smoothness(matrix):
    # Normalize f to get transition probabilities P
    if matrix.shape[0] < 3:
        return 0
    smoothness_measure = np.sum(laplace(matrix)**2)
    return -smoothness_measure

def objective_function(f, g):
    discrepancy = np.linalg.norm(f - g, 'fro')
    zeros_f = np.mean(f == 0)
    zeros_g = np.mean(g == 0)
    smoothness_f = calculate_smoothness(f)
    smoothness_g =  calculate_smoothness(g)
    return discrepancy, smoothness_f, smoothness_g

def select_opt_bin(H_1_test, H_1_train, bin_range, num_folds = 1,  lower = 0, upper = 100, 
                   threshold = 1, lam_reg=0.5, tolerance = 0.01, verbose = False,
                   zero = 0.25, min_state = 3, max_state = 1000):
    fro_norms = {}

    def split_into_folds(data, num_folds):
        fold_size = data.shape[0] // num_folds
        return [data[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    for k in itertools.product(bin_range, repeat=H_1_test.shape[1]):
        
        if (np.prod(k) < min_state or np.prod(k) > max_state):
            continue
        f, p1, g, p2 = discretize(H_1_test, H_1_train, k, False, lower, upper, threshold)
        if verbose: print(np.sum(f != 0),np.prod(k), (np.prod(k)**2) * zero, np.sum(g != 0), (np.prod(k)**2) * zero)
        if (np.sum(f >= 1) < (np.prod(k)**2) * zero) or (np.sum(g >= 1) < (np.prod(k)**2) * zero):
            continue
        
        discrepancy, smoothness_f, smoothness_g = objective_function(p1, p2)

        fro = discrepancy + lam_reg * (smoothness_f + smoothness_g)
        fro_norms[k] = fro
        
        if verbose: print(f'{k} bins with discrepancy {discrepancy},\nsmoothness {smoothness_f, smoothness_g}, and sum {fro}')
    max_norm = [key for key in fro_norms if all(fro_norms[temp] <= fro_norms[key] for temp in fro_norms)][0]
    if tolerance == 0:
        return (max_norm, max_norm)

    close_to_max_keys = [k for k, v in fro_norms.items() if (fro_norms[max_norm] - v) <= abs(tolerance * fro_norms[max_norm])]

    if len(close_to_max_keys) > 1:
        if verbose: print(close_to_max_keys)
        min_norm_key = max(close_to_max_keys, key=lambda x: np.prod(x))
    else:
        min_norm_key = close_to_max_keys[0]

    return (max_norm, min_norm_key)

def discretize(H, H1, bins=10, plot=False, lower=0, upper=100, threshold=2 , verbose = False):
    # Combine data from both sequences for consistent binning
    H_range = np.concatenate((H, H1), axis=0)
    # Compute percentile-based boundaries for binning
    lower_bound = np.percentile(H_range, lower, axis=0)
    upper_bound = np.percentile(H_range, upper, axis=0)
    num_bins = bins
    # Determine bins based on the type of bins requested
    if isinstance(bins, int):
        # Linearly spaced bins
        bins = [np.linspace(lower_bound[i], upper_bound[i], num=bins - 1) for i in range(H.shape[1])]
    elif isinstance(bins, (list, np.ndarray, tuple)):
        # If bins is an array/list, create specified number of bins for each dimension
        bins = [np.linspace(lower_bound[i], upper_bound[i], num=bins[i] - 1) for i in range(H.shape[1])]
    else:
        # Non-uniform binning methods
        if bins == 'blocks':
            bins = [bayesian_blocks(H_range[:, i], fitness='events') for i in range(H.shape[1])]
        elif bins == 'knuth':
            bins = [knuth_bin_width(H_range[:, i], return_bins=True)[1] for i in range(H.shape[1])]
        else:
            bins = [np.histogram_bin_edges(H_range[:, i], bins=bins) for i in range(H.shape[1])]
    ngrid = len(bins[0]) - 1

    def get_discrete_states_and_counts(seq, bins):
        # Discretize each dimension of the sequence
        seq_discretized = np.array([np.digitize(seq[:, dim], bins[dim]) for dim in range(seq.shape[1])]).T
        
        # Count occurrences of each state
        state_counts = {}
        for state_tuple in map(tuple, seq_discretized):
            state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1
        
        # Filter out states with counts less than the threshold
        frequent_states = {state: count for state, count in state_counts.items() if count > 0}
        infrequent_states = {state: count for state, count in state_counts.items() if count <= 0}

        # Merge infrequent states to the nearest frequent state
        for inf_state, inf_count in infrequent_states.items():
            distances = {frq_state: distance.euclidean(inf_state, frq_state) for frq_state in frequent_states.keys()}
            nearest_state = min(distances, key=distances.get)
            frequent_states[nearest_state] += inf_count

        return frequent_states, seq_discretized
    
    # Discretize both sequences and calculate state counts
    sc, sd = get_discrete_states_and_counts(H, bins)
    sc1, sd1 = get_discrete_states_and_counts(H1, bins)
    
    # Combine states found in both sequences
    combined_states = set(sc.keys()).union(sc1.keys())
    # Generate all possible states based on the bin edges
    all_states = set(itertools.product(*[range(num_bins[i]) for i in range(H.shape[1])]))


    # Add back all states, including those with small entries, and give 0 as entry number if not found in sc or sc1
    for state in all_states:
        sc.setdefault(state, 0)
        sc1.setdefault(state, 0)



    def compute_transition_matrices(state_counts, seq_discretized, threshold=1e-3, plot=False):
        # Map states to indices
        state_index = {state: idx for idx, state in enumerate(state_counts.keys())}
        transition_matrix = np.zeros((len(state_index), len(state_index)), dtype=int)
        all_states = []
        for (from_state, to_state) in zip(seq_discretized[:-1], seq_discretized[1:]):
            if tuple(from_state) in state_index and tuple(to_state) in state_index:
                transition_matrix[state_index[tuple(from_state)], state_index[tuple(to_state)]] += 1
                all_states.append(state_index[tuple(to_state)])
        
        # Apply threshold by setting transitions below the threshold to zero
        transition_matrix[transition_matrix <= threshold] = 0
        
        transition_prob_matrix = np.nan_to_num(transition_matrix / transition_matrix.sum(axis=1, keepdims=True), nan=0)
        if plot:
            plt.figure()
            plt.hist(all_states, bins=len(state_index.values()), 
                     color='grey', alpha=0.6, density=True)
            plt.axis('off')
            plt.show()
            
            fig = plt.figure()
            cmap = sns.color_palette("light:grey", as_cmap=True)
            sns.heatmap(transition_prob_matrix, annot=False, cmap=cmap, square=True, vmin=0, vmax=1, 
                        cbar= False, linewidths=1)
            plt.xlabel("$h_i$", fontsize=15, labelpad=5)
            plt.ylabel("$h_{i-1}$", fontsize=15, labelpad=5, rotation=90, va='center')
            plt.xticks([])
            plt.yticks([])
            plt.show()

        return transition_matrix, transition_prob_matrix
    
    # Compute transition matrices for both sequences
    tm, tpm = compute_transition_matrices(sc, sd, threshold, plot)
    tm1, tpm1 = compute_transition_matrices(sc1, sd1, threshold, plot)
    
    return tm, tpm, tm1, tpm1

def run_with_retry(func, params, retries=100, delay=0.1):
    import time
    for i in range(retries):
        try:
            H_1_test, H_1_train, lam_reg, zeros, min_state, config = params
            p_val = func(H_1_test, H_1_train, lam_reg, zeros, min_state, config)
            return p_val  # If it succeeds, break out of the loop
        except Exception as e:
            print(f"Error: {e}, retrying {i + 1}/{retries}...")
            time.sleep(delay)
            if i == retries - 1:
                raise  # Re-raise the exception after the last retry
        

