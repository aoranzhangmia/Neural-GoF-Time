import numpy as np
import torch
import torch.nn as nn
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import argparse
# import seaborn as sns
import pandas as pd
# %matplotlib notebook
# %matplotlib widget
# !pip install clustpy
import pickle
import random
from IPython.display import clear_output
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pickle


class VanillaHawkes:
    def __init__(self, mu, beta, num_sequences, T, lam_bar, min_length):
        self.mu = mu
        self.alpha = beta
        self.beta = beta
        self.seq_length = num_sequences
        self.T = T
        self.lam_bar = lam_bar
        self.min_length = min_length
    def intensity(self, t, past_events):
        if len(past_events) > 0:
            return self.mu + np.sum(self.beta * np.exp(- self.beta * (t - past_events)))
        else:
            return self.mu

    def trigger(self, dt):
        return self.alpha * np.exp(-self.beta*(dt))

    def papangelou(self, t, X):
        """
        Papangelou conditional intensity function at time t given points X.
        This is different from the conditional intensity (self.intensity).

        Args:
            t: array(dim), a new time point.
            X: array((..., dim)), existing time-points in a sample.
        """
        assert X.shape[1] == 1

        times = X.ravel()   # array
        denom = [self.intensity(u, X[X<u]) for u in times[times > t]]  # Check
        g_t = self.trigger(times[times > t] - t)
        log_frac = np.sum(np.log(denom + g_t) - np.log(denom))

        # Integrated intensity term
        log_Lterm = -self.alpha/self.beta * (1.-np.exp(-(self.T[1]-t)*self.beta))

        # Intensity term
        log_lterm = np.log(self.intensity(t, X[X<t]))
        # print(log_Lterm)

        return np.exp(log_Lterm + log_lterm + log_frac)


    def homo_pp(self):
        n = self.T[1] - self.T[0]
        N = np.random.poisson(size = 1, lam = self.lam_bar * n)  \
            # n = N(T) ~ poisson(lam * n)
        times = np.random.uniform(self.T[0], self.T[1], N)
        # print(times)
        times = times[times.argsort()]
        # print(times)
        return times

    def simulate_inhomo_pp(self):
        times = []
        lam = []
        homo_points = self.homo_pp()
        for i in range(homo_points.shape[0]):
            cur_t = homo_points[i]
            hist_t = times
            cur_lam = self.intensity(cur_t, hist_t)
            d = np.random.uniform()
            if cur_lam > self.lam_bar:
                # print("exceeds")
                return (None, None)
            if d <= cur_lam / (self.lam_bar):
                # print("d too small")
                times.append(cur_t)
                lam.append(cur_lam)
        # print(times)
        return times, lam

    def simulate_multiple_seq(self):
        all_lams = []
        all_times = []
        lens = []
        i = 0
        while i < self.seq_length:
            times, lam = self.simulate_inhomo_pp()
            if (times is None) or len(times) < self.min_length:
                # print("too short")
                continue
            all_lams.append(lam)
            all_times.append(times)
            lens.append(len(times))
            i += 1
        time_array = np.zeros((len(lens), max(lens)), dtype = "float32")
        lams_array = np.zeros((len(lens), max(lens)), dtype = "float32")
        for i in range(len(lens)):
            time_array[i, :lens[i]] = all_times[i]
            lams_array[i, :lens[i]] = all_lams[i]
        return time_array, lams_array, np.asarray(lens)
    
class SelfExcitingProcess:
    def __init__(self, mu, alpha, beta, num_sequences, T, lam_bar, min_length):
        self.mu = mu  # Base rate
        self.alpha = alpha  # Excitement factor
        self.beta = beta  # Decay rate
        self.seq_length = num_sequences
        self.T = T
        self.lam_bar = lam_bar
        self.min_length = min_length

    def intensity(self, t, past_events):
        if len(past_events) > 0:
            # print(past_events, t)
            return self.mu + np.sum(self.alpha * np.exp(-self.beta * (t - np.array(past_events))))
        else:
            return self.mu

    def trigger(self, dt):
        return self.alpha * np.exp(-self.beta*(dt))

    def papangelou(self, t, X):
        assert X.shape[1] == 1

        times = X.ravel()   # array
        denom = [self.intensity(u, X[X<u]) for u in times[times > t]]  # Check
        g_t = self.trigger(times[times > t] - t)
        log_frac = np.sum(np.log(denom + g_t) - np.log(denom))

        # Integrated intensity term
        log_Lterm = -self.alpha/self.beta * (1.-np.exp(-(self.T[1]-t)*self.beta))

        # Intensity term
        log_lterm = np.log(self.intensity(t, X[X<t]))
        # print(log_Lterm)

        return np.exp(log_Lterm + log_lterm + log_frac)

    def homo_pp(self):
        n = self.T[1] - self.T[0]
        N = np.random.poisson(self.lam_bar * n)
        print(N)
        times = np.random.uniform(self.T[0], self.T[1], N)
        times.sort()
        return times

    def simulate_inhomo_pp(self):
        times = []
        lam = []
        homo_points = self.homo_pp()
        for cur_t in homo_points:
            cur_lam = self.intensity(cur_t, times)
            if cur_lam > self.lam_bar:
                print("intensity exceeds upper bound")
                return (None, None)
            d = np.random.uniform( )
            if d <= cur_lam / self.lam_bar:
                times.append(cur_t)
                lam.append(cur_lam)
        return times, lam

    def simulate_multiple_seq(self):
        all_lams = []
        all_times = []
        lens = []
        i = 0
        while i < self.seq_length:
            times, lam = self.simulate_inhomo_pp()
            if times is None or (len(times) < self.min_length):
                print(0 if times is None else len(times))
                continue
            all_lams.append(lam)
            all_times.append(times)
            lens.append(len(times))
            i += 1
        time_array = np.zeros((len(lens), max(lens)), dtype="float32")
        lams_array = np.zeros((len(lens), max(lens)), dtype="float32")
        for i, length in enumerate(lens):
            time_array[i, :length] = all_times[i]
            lams_array[i, :length] = all_lams[i]
        return time_array, lams_array, np.asarray(lens)

class SelfCorrectingProcess:
    def __init__(self, mu, alpha, beta, num_sequences, T, lam_bar, min_len):
        self.mu = mu  # Base rate
        self.alpha = alpha  # Damping factor
        self.beta = beta  # Recovery rate
        self.seq_length = num_sequences
        self.T = T
        self.lam_bar = lam_bar
        self.min_len = min_len

    def intensity(self, t, past_events):
        return np.exp(self.mu + self.alpha * (t - self.beta * len(past_events)))

    def trigger(self, dt):
        return np.exp(-self.beta * dt)

    def papangelou(self, t, X):
        assert X.shape[1] == 1

        times = X.ravel()  # Flatten array to 1D
        denom = [self.intensity(u, X[X < u]) for u in times[times > t]]
        g_t = self.trigger(times[times > t] - t)
        log_frac = np.sum(np.log(denom + g_t) - np.log(denom))

        # Integrated intensity term
        integrated_intensity = np.sum([self.intensity(u, X[X < u]) for u in times if u <= t])
        log_Lterm = -integrated_intensity

        # Intensity term
        log_lterm = np.log(self.intensity(t, X[X < t]))

        return np.exp(log_Lterm + log_lterm + log_frac)

    def homo_pp(self):
        n = self.T[1] - self.T[0]
        N = np.random.poisson(self.lam_bar * n)
        times = np.random.uniform(self.T[0], self.T[1], N)
        times.sort()
        return times

    def simulate_inhomo_pp(self):
        times = []
        lam = []
        homo_points = self.homo_pp()
        # print(homo_points)
        for cur_t in homo_points:
            cur_lam = self.intensity(cur_t, times)

            if cur_lam > self.lam_bar:  # Still reject if intensity exceeds upper bound
                print("intensity exceeds upper bound")
                return (None, None)
            d = np.random.uniform()
            # print(d,cur_lam / self.lam_bar )
            if d < cur_lam / self.lam_bar:
                times.append(cur_t)
                lam.append(cur_lam)
        return times, lam

    def simulate_multiple_seq(self):
        all_lams = []
        all_times = []
        lens = []
        i = 0
        while i < self.seq_length:
            times, lam = self.simulate_inhomo_pp()
            # print(times)
            if times is None or not (len(times) < self.min_len):  # Ensure the sequence has enough events
                print(f"Too short: {0 if times == None else len(times)}")
                continue
            all_lams.append(lam)
            all_times.append(times)
            lens.append(len(times))
            i += 1
        time_array = np.zeros((len(lens), max(lens)), dtype="float32")
        lams_array = np.zeros((len(lens), max(lens)), dtype="float32")
        for i, length in enumerate(lens):
            time_array[i, :length] = all_times[i]
            lams_array[i, :length] = all_lams[i]
        return time_array, lams_array, np.asarray(lens)
    
def concat_array(data, lens, cluster_len):
    num_seq = sum(cluster_len)
    num = np.cumsum(cluster_len)
    num = np.hstack(([0], num))
    max_len = max([max(l) for l in lens])
    concat = np.zeros((num_seq, max_len))
    for i in range(len(data)):
      concat[num[i]:num[i+1], :data[i].shape[1]] = data[i]
    return concat

def concat_2d_array(data, lens, cluster_len):
    num_seq = sum(cluster_len)
    num = np.cumsum(cluster_len)
    num = np.hstack(([0], num))
    max_len = max([max(l) for l in lens])
    concat = np.zeros((num_seq, max_len, 3))
    for i in range(len(data)):
      concat[num[i]:num[i+1], :data[i].shape[1], :] = data[i]
    return concat

def plot_1d_pointprocess(model, points, T, ngrid=100):
    ts = np.linspace(T[0], T[1], ngrid)
    lamvals = []
    for t in ts:
        his_t = points[(points <= t) * (points > 0)]
        lamval = model.intensity(t, his_t)
        lamvals.append(lamval)

    evals = []
    for t in points:
        his_t = points[(points <= t) * (points > 0)]
        lamval = model.intensity(t, his_t)
        evals.append(lamval)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(ts, lamvals, linestyle="--", color="lightgrey")
    # ax.fill_between(ts, 0, lamvals, color="grey", alpha=0.3)  # Shaded area
    ax.scatter(points, evals, c="r", s = 1.5)
    for point in points:
        ax.vlines(point, ymin=min(lamvals), ymax=min(lamvals) +
                  ((max(lamvals)- min(lamvals))*0.01),
                  colors='blue', linestyles='solid', linewidth=0.5)
    plt.ylabel('Conditional Intensity')
    plt.xlabel('Time')
    plt.show()

    return lamvals, evals

if __name__ == '__main__':
    # parameter initialization
    base_mu = 1
    alpha = 1
    beta = 1.25
    se = SelfExcitingProcess(base_mu, alpha, beta, 100, [0., 100.], 1e+2, 800)
    se_time_train, se_lam_train, se_len_train = se.simulate_multiple_seq()
    se_time_test, se_lam_test, se_len_test = se.simulate_multiple_seq()
    sc = SelfCorrectingProcess(2.5, 0.05, 0.25, 100, [0., 100.], 1e+2, 500)
    sc_time_train, sc_lam_train, sc_len_train = \
    sc.simulate_multiple_seq()
    sc_time_test, sc_lam_test, sc_len_test = \
        sc.simulate_multiple_seq()
    _ = plot_1d_pointprocess(se, se_time_train[19], [0.,100.], 300)
    _ = plot_1d_pointprocess(sc, sc_time_train[15], [0.,100.], 300)
    data_to_save = {
    'se_time_train': se_time_train,
    'se_lam_train': se_lam_train,
    'se_len_train': se_len_train,
    'se_time_test': se_time_test,
    'se_lam_test': se_lam_test,
    'se_len_test': se_len_test,
    'sc_time_train': sc_time_train,
    'sc_lam_train': sc_lam_train,
    'sc_len_train': sc_len_train,
    'sc_time_test': sc_time_test,
    'sc_lam_test': sc_lam_test,
    'sc_len_test': sc_len_test
    }
    with open('training_data.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)
    print("Data has been serialized and saved to 'training_data.pkl'")