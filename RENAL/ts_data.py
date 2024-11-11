import numpy as np
import torch
import torch.nn as nn
import os
import random
import matplotlib
import matplotlib.pyplot as plt
import argparse
import pickle
import random
from IPython.display import clear_output
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.arima_process as ap
from arch import arch_model


def generate_ar1_sequence(phi, sigma, seq_len):
    # Initialize the array for the AR(1) process
    y = np.zeros(seq_len)

    # Generate white noise
    e = np.random.normal(0, sigma, seq_len)

    # Simulate the AR(1) process
    for t in range(1, seq_len):
        y[t] = phi * y[t-1] + e[t]

    # Create an array with time steps and the AR(1) values
    sequence = np.column_stack((np.arange(seq_len), y))
    return sequence

def generate_multiple_ar1_sequences(num, seq_len, phi, sigma, plot = True):
    sequences = np.zeros((num, seq_len, 2))
    for i in range(num):
        sequences[i] = generate_ar1_sequence(phi, sigma, seq_len)
    if plot:
        plt.figure()
        for i in range(100):
            plt.plot(sequences[i][:, 0], sequences[i][:, 1], label=f'Sequence {i+1}')
        plt.title('Simulated AR(1) Sequences')
        plt.xlabel('Time')
        plt.ylabel('Value')
        # plt.legend()
        plt.show()

    return sequences

def generate_arma_sequence(ar_params, ma_params, sigma, seq_len):
    # Define the AR and MA parameters
    ar = np.r_[1, -np.array(ar_params)]  # Add the zero-lag and negate AR params
    ma = np.r_[1, np.array(ma_params)]   # Add the zero-lag for MA params

    # Generate white noise
    e = np.random.normal(0, sigma, seq_len)

    # Simulate the ARMA process
    y = ap.arma_generate_sample(ar, ma, seq_len, scale=sigma)

    # Create an array with time steps and the ARMA values
    sequence = np.column_stack((np.arange(seq_len), y))
    return sequence

def generate_multiple_arma_sequences(num, seq_len, ar_params, ma_params, sigma, plot = True):
    sequences = np.zeros((num, seq_len, 2))
    for i in range(num):
        sequences[i] = generate_arma_sequence(ar_params, ma_params, sigma, seq_len)
    plt.figure()
    for i in range(10):
        plt.plot(sequences[i][:, 0], sequences[i][:, 1], label=f'Sequence {i+1}')
    plt.title('Simulated ARMA Sequences')
    plt.xlabel('Time')
    plt.ylabel('Value')
    # plt.legend()
    plt.show()

    return sequences

def generate_garch_sequence(mu, omega, alpha, gamma, beta, eta, lambda_, seq_len):
    # Define the GJR-GARCH model (including asymmetric effects)
    model = arch_model(None, vol='Garch', p=len(alpha), o=len(gamma), q=len(beta), power=eta)

    # Simulate the GARCH process
    params = np.array([mu, omega] + alpha + gamma + beta + [lambda_])
    sim_data = model.simulate(params=params, nobs=seq_len)

    # Create an array with time steps and the GARCH values
    sequence = np.column_stack((np.arange(seq_len), sim_data['data']))
    return sequence

def generate_multiple_garch_sequences(num, seq_len, mu, omega, alpha, gamma, beta, eta, lambda_, plot=True):
    sequences = np.zeros((num, seq_len, 2))
    for i in range(num):
        sequences[i] = generate_garch_sequence(mu, omega, alpha, gamma, beta, eta, lambda_, seq_len)
    if plot:
        plt.figure(figsize=(15, 10))
        for i in range(2):
            plt.plot(sequences[i][:, 0], sequences[i][:, 1], label=f'Sequence {i+1}')
        plt.title('Simulated GARCH Sequences')
        plt.xlabel('Time')
        plt.ylabel('Value')
        # plt.legend()
        plt.show()

    return sequences

if __name__ == '__main__':
    num = 100 
    seq_len = 1000  
    phi = 0.5  # |phi| < 1 means stationary
    sigma = 1  

    ar1_1 = generate_multiple_ar1_sequences(num, seq_len, phi, sigma)
    ar1_2 = generate_multiple_ar1_sequences(num, seq_len, 0.2, 0.8)
    
    # Parameters
    num = 100  # Number of sequences
    seq_len = 1000  # Length of each sequence
    ar_params = [0.5, 0.4]  # AR parameters (e.g., AR(2) with phi_1=0.75 and phi_2=-0.25)
    ma_params = [0.65]  # MA parameters (e.g., MA(1) with theta_1=0.65)
    sigma = 1  # Standard deviation of the white noise

    # Generate the sequences
    arma_1 = generate_multiple_arma_sequences(num, seq_len, ar_params, ma_params, sigma)

    r_params = [0.5, -0.4]  # AR parameters (e.g., AR(2) with phi_1=0.5 and phi_2=-0.4)
    ma_params = [0.8, -0.5]  # MA parameters (e.g., MA(2) with theta_1=0.3 and theta_2=-0.2)
    arma_2 = generate_multiple_arma_sequences(num, seq_len, ar_params, ma_params, sigma)
    
    num = 100  # Number of sequences
    seq_len = 1000  # Length of each sequence
    mu = 0.029365
    omega = 0.044374
    alpha = [0.044344]
    gamma = [0.02]
    beta = [0.931280]
    eta = 1.0  # Power parameter for GARCH, default is usually 1
    lambda_ = -0.041616

    garch = generate_multiple_garch_sequences(num, seq_len, mu, omega, alpha, gamma, beta, eta, lambda_)


    
    