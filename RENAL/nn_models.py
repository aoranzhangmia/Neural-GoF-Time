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

def Exponential_Hawkes(ts, seqs, mu, alpha, beta):
    lams = []

    for t in ts:
        mask = (seqs > 0) * (seqs < t)  
        lams.append(mu + np.sum(alpha * np.exp(-beta * (t - seqs[mask]))))

    return np.array(lams).reshape(1,-1) 

def config_generate(T0,
                    T1,
                    int_res='100',
                    nsample='50',
                    n_class='2',
                    hid_dim='1',
                    event_dim='1',
                    lr='0.01',
                    epoch='10',
                    batch_size='64',
                    opt="Adam",
                    momentum='0.0',
                    train_type="event",
                    base_mu = '0.0',
                    alpha = '0.0',
                    beta = '0.0', 
                    weight_decay = '0.0',
                    max_bin = '0',
                    lam_reg = '0.0',
                                        spatial_res = '100'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-T0", type=float)
    parser.add_argument("-T1", type=float)
    parser.add_argument("-tau_max", type=float)
    parser.add_argument("-int_res", type=int)
    parser.add_argument("-nsample", type=int)
    parser.add_argument("-max_bin", type=int)
    parser.add_argument("-n_class",
                        help="Number of types in the dataset, default is 2", type=int, default=2)
    parser.add_argument("-n_cluster", type=int, default=1)
    parser.add_argument("-hid_dim", type=int, default=1)
    parser.add_argument("-event_dim", type=int, default=1)
    parser.add_argument("-n_layers", type=int, default=1)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-momentum", type=float, default=0.0)
    parser.add_argument("-epoch", type=int, default=30)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-opt", type=str, default='Adam')
    parser.add_argument("-train_type", type=str, default='event')
    parser.add_argument("-train_buffer", action='store_true')
    parser.add_argument("-no-train_buffer", dest='train_buffer', action='store_false')
    parser.add_argument("-base_mu", type=float)
    parser.add_argument("-alpha", type=float)
    parser.add_argument("-beta", type=float)
    parser.add_argument("-weight_decay", type=float)
    parser.add_argument("-lam_reg", type=float)
    parser.set_defaults(train_buffer=True)
    parser.add_argument("-spatial_res", type=int)


    arg_list = ['-T0', T0,
                '-T1', T1,
                '-int_res', int_res,
                '-spatial_res', spatial_res,
                '-nsample', nsample,
                '-n_class', n_class,
                '-hid_dim', hid_dim,
                '-event_dim', event_dim,
                '-lr', lr,
                '-epoch', epoch,
                '-batch_size', batch_size,
                '-opt', opt,
                '-momentum', momentum,
                '-train_type', train_type,
                '-base_mu', base_mu,
                '-alpha', alpha,
                '-beta', beta,
                '-weight_decay', weight_decay,
                '-max_bin', max_bin,
                '-lam_reg', lam_reg
                ]

    config = parser.parse_args(arg_list)

    return config

def generate_initial_weights_and_biases(num_layers, input_dim, hidden_dim, output_dim):
    input_dim = input_dim + output_dim # include target as input
    weights = {}
    biases = {}
    for i in range(num_layers):
        layer_input_dim = input_dim if i == 0 else hidden_dim
        weights[f'W_i_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim, layer_input_dim + hidden_dim))
        weights[f'W_f_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim, layer_input_dim + hidden_dim))
        weights[f'W_c_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim, layer_input_dim + hidden_dim))
        weights[f'W_o_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim, layer_input_dim + hidden_dim))
        weights[f'W_y_{i}'] = torch.nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        
        biases[f'b_i_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim))
        biases[f'b_f_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim))
        biases[f'b_c_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim))
        biases[f'b_o_{i}'] = torch.nn.Parameter(torch.Tensor(hidden_dim))
        biases[f'b_y_{i}'] = torch.nn.Parameter(torch.Tensor(output_dim))
        # Initialize weights
        for name, param in weights.items():
            if name.endswith(f'_{i}'):  # Initialize only current layer's weights
                torch.nn.init.xavier_uniform_(param)
        for name, param in biases.items():
            if name.endswith(f'_{i}'):  # Initialize only current layer's biases
                torch.nn.init.zeros_(param)
                
    return weights, biases

class SimpleLSTM(nn.Module):
    def __init__(self, config, input_dim, output_dim, weights, biases, hid_dim = None, single = False):
        super(SimpleLSTM, self).__init__()
        self.input_dim = input_dim + 1
        self.output_dim = output_dim
        if hid_dim == None:
            self.hidden_dim = config.hid_dim
        else:
            self.hidden_dim = hid_dim
        self.batch_size = config.batch_size
        if single:
            # Parameters for input gate
            self.W_i = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim + self.hidden_dim))
            self.b_i = nn.Parameter(torch.Tensor(self.hidden_dim))
            
            # Parameters for forget gate
            self.W_f = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim + self.hidden_dim))
            self.b_f = nn.Parameter(torch.Tensor(self.hidden_dim))
            
            # Parameters for output gate
            self.W_o = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim + self.hidden_dim))
            self.b_o = nn.Parameter(torch.Tensor(self.hidden_dim))
            
            # Parameters for cell state
            self.W_c = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim + self.hidden_dim))
            self.b_c = nn.Parameter(torch.Tensor(self.hidden_dim))
            
            # Parameters for final gate
            self.W_y = nn.Parameter(torch.Tensor(self.output_dim, self.hidden_dim))
            self.b_y = nn.Parameter(torch.Tensor(self.output_dim))
        else:
        
            self.W_i = weights['W_i']
            self.W_f = weights['W_f']
            self.W_c = weights['W_c']
            self.W_o = weights['W_o']
            self.W_y = weights['W_y']
            
            self.b_i = biases['b_i']
            self.b_f = biases['b_f']
            self.b_c = biases['b_c']
            self.b_o = biases['b_o']
            self.b_y = biases['b_y']
            
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        self.init_weights()
        self.criterion = nn.MSELoss()
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        # self.dropout = nn.Dropout(p=0.1)
        self.scheduler = self.set_scheduler(self.optimizer)
        
    def set_optimizer(self, opt, lr, momentum=None, weight_decay=1e-5):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'Adadelta':
            return torch.optim.Adadelta(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'Adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("No such optimizer!")
        
    def set_scheduler(self, optimizer, mode='min', factor=0.1, patience=20, verbose=True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
        return scheduler
    
    def init_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, x_t, h_t, c_t):
        """ Forward pass for LSTM which processes one timestep """
        # x_t = self.dropout(x_t)
        z_t = torch.cat((h_t, x_t), dim=1)
        
        i_t = torch.sigmoid(torch.matmul(z_t, self.W_i.t()) + self.b_i)  
        f_t = torch.sigmoid(torch.matmul(z_t, self.W_f.t()) + self.b_f)
        o_t = torch.sigmoid(torch.matmul(z_t, self.W_o.t()) + self.b_o)
        c_tilde_t = torch.tanh(torch.matmul(z_t, self.W_c.t()) + self.b_c)

        c_t = f_t * c_t + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        y_t = torch.matmul(h_t, self.W_y.t()) + self.b_y
        return h_t, c_t, y_t
    
    def train(self, x, y, plot, init_states=None):
        """ Train the LSTM over all timesteps in the input """
        # Assuming y is of shape (batch, seq_len, 1) and x is of shape (batch, seq_len, input_dim)
        # Concatenate y shifted by one timestep to x along the feature dimension
        shifted_y = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        # print(shifted_y)
        x_with_y = torch.cat([x, shifted_y], dim=-1)  # Now x_with_y has one additional dimension
        # print(x_with_y)

        seq_size = x_with_y.shape[1]
        # print(seq_size)
        h = []
        y_pred = []
        h_t, c_t = (torch.rand(self.batch_size, self.hidden_dim).to(x.device),
                        torch.rand(self.batch_size, self.hidden_dim).to(x.device))

        for t in range(seq_size):
            # print(x_with_y[:, t, :])
            h_t, c_t, y_t = self.forward(x_with_y[:, t, :], h_t, c_t)
            h.append(h_t)
            y_pred.append(y_t)

        h_seq = torch.stack(h, dim=1)
        y_seq = torch.stack(y_pred, dim=1)
        if plot:
            plt.figure()
            plt.plot(x.detach().numpy()[0], y.detach().numpy()[0], color = 'blue')
            plt.plot(x.detach().numpy()[0], y_seq.detach().numpy()[0], color='red')
            plt.tight_layout()
            plt.show()
        loss = self.criterion(y_seq, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item(), h_seq.detach().numpy(), y_seq.detach().numpy()

    def test(self, x, y, init_states=None):
        """ Evaluate the model without updating weights """
        # Include previous target values in the input sequence
        shifted_y = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        x_with_y = torch.cat([x, shifted_y], dim=-1)

        seq_size = x_with_y.shape[1]
        y_pred = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.batch_size, self.hidden_dim).to(x_with_y.device),
                        torch.zeros(self.batch_size, self.hidden_dim).to(x_with_y.device))
        else:
            h_t, c_t = init_states

        with torch.no_grad():
            for t in range(seq_size):
                h_t, c_t, y_t = self.forward(x_with_y[:, t, :], h_t, c_t)
                y_pred.append(y_t)

        y_seq = torch.stack(y_pred, dim=1)
        loss = self.criterion(y_seq, y)
        return loss.item(), y_seq.detach().numpy()

    def get_H_mat(self, x, y, init_states=None):
        """ Retrieve all hidden states across the sequence """
        # Include previous target values in the input sequence
        shifted_y = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        x_with_y = torch.cat([x, shifted_y], dim=-1)

        seq_size = x_with_y.shape[1]
        hidden_states = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.batch_size, self.hidden_dim).to(x_with_y.device),
                        torch.zeros(self.batch_size, self.hidden_dim).to(x_with_y.device))
        else:
            h_t, c_t = init_states

        with torch.no_grad():
            for t in range(seq_size):
                h_t, c_t, _ = self.forward(x_with_y[:, t, :], h_t, c_t)
                hidden_states.append(h_t)

        h_seq = torch.stack(hidden_states, dim=1)
        return h_seq.detach().numpy()

def model_train(train_data, test_data, model, config, mae_eval=False, ts=None,
                true_lam=None, plot = True):
    """Training process"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = model.to(device)
    train_llks = []
    test_llks = []
    test_maes = []
    n_events_train = (train_data > 0).sum()
    n_events_test = (test_data > 0).sum()

    n_batches = int(train_data.shape[0] / config.batch_size) 
    batch_idx = np.arange(train_data.shape[0])
    random.shuffle(batch_idx)

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

    epc = 0
    while epc < config.epoch:
        # print(f"Epoch {epc}")
        train_loss = 0
        for b in range(n_batches):
            idx = np.arange(config.batch_size * b, config.batch_size * (b + 1))
            loss = model.train(train_data[idx])
            if isinstance(loss, tuple):
                loss = loss[0]
            train_loss += loss
            train_data_loss = model.test(train_data, False)

        test_data_loss = model.test(test_data, False)
        if isinstance(train_data_loss, tuple):
            train_data_loss = train_data_loss[0]
        if isinstance(test_data_loss, tuple):
            test_data_loss = test_data_loss[0]
        

        train_llks.append(-train_data_loss / n_events_train)
        test_llks.append(-test_data_loss / n_events_test)
        
        if np.isnan(train_llks[-1]):
            return model, train_llks, test_llks, test_maes
        
        if mae_eval:
            print(f"{epc + 1} epoch train lld: ", (train_llks[-1]))
            print(f"{epc + 1} epoch test lld: ", (test_llks[-1]))

        epc += 1
        ts = np.linspace(config.T0, config.T1, 100)

    if plot:
           
            fig, axs = plt.subplots(1, 3, figsize=(12,3))
            axs[0].plot(test_llks)
            axs[0].set_title(f'Log-likelihood Convergence Plot (Test)')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Test lld')
            true_lams = Exponential_Hawkes(ts, train_data.detach().numpy(), config.base_mu, config.alpha, config.beta)
            nh_lams = model.sample_intensity(ts, train_data, config.base_mu)
            plot_idx = 0
            axs[1].plot(ts, nh_lams[plot_idx], color="darkred", alpha=0.5, linewidth=3, label="NH")
            axs[1].legend(fontsize=20)
            axs[1].set_xlabel(r"$t$", fontsize=15)
            axs[1].set_ylabel(r"$\lambda(t)$", fontsize=15)
            axs[1].set_title("Conditional Intensity", fontsize = 15)
            axs[2].plot(ts, true_lams[plot_idx], color="darkred", alpha=0.5, linewidth=3, label="NH")
            axs[2].legend(fontsize=20)
            axs[2].set_xlabel(r"$t$", fontsize=15)
            axs[2].set_ylabel(r"$\lambda(t)$", fontsize=15)
            axs[2].set_title("True Conditional Intensity", fontsize = 15)
            plt.tight_layout()
            plt.show()

    print("training done!")

    return model, train_llks, test_llks, test_maes

class NeuralHawkes(nn.Module):
    """
    Neural Hawkes Network (https://github.com/Hongrui24/NeuralHawkesPytorch; https://github.com/hongyuanmei/neurawkes) 
    proposed by Hongyuan Mei and Jason Eisner (2017).
    https://proceedings.neurips.cc/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html
    """
    def __init__(self, config):
        super(NeuralHawkes, self).__init__()
        self.T0 = config.T0
        self.T1 = config.T1
        self.hid_dim = config.hid_dim
        self.batch_size = config.batch_size
        self.config = config
        self.base_mu = config.base_mu
        self.weight_decay = config.weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM cell parameters
        self.W = nn.Linear(2 * self.hid_dim, 7 * self.hid_dim)
            # i, f, z, o, i_bar, f_bar, delta
        self.emb = nn.Embedding(2, self.hid_dim) 
        self.w = nn.Linear(self.hid_dim, 1) 
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        # Output (λ_k(t))
        # lambda_k_t = torch.nn.functional.softplus(self.W_k(h_t1))
        self.scheduler = self.set_scheduler(self.optimizer)

    def set_optimizer(self, opt, lr, momentum):
      if opt == 'SGD':
        return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=self.weight_decay,
                               momentum = self.config.momentum)
      elif opt == 'Adam':
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay = self.weight_decay)
      elif opt == 'Adadelta':
        return torch.optim.Adadelta(self.parameters(), lr=lr, weight_decay = self.weight_decay)
      elif opt == 'Adagrad':
        return torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=self.weight_decay)
      else:
        ValueError("No such optimizer!")
        
    def set_scheduler(self, optimizer, mode='max', factor=0.1, patience=5, verbose=True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
        return scheduler


    def init_states(self, batch_size):
        # o, c_bar_1, c_1, delta_1
        return (torch.zeros(batch_size, self.hid_dim) ,
                torch.zeros(batch_size, self.hid_dim) ,
                torch.zeros(batch_size, self.hid_dim) ,
                torch.zeros(batch_size, self.hid_dim) )

    def intensity(self, H):
        """
        Calculate intensity from hidden states
        """
        # print(nn.functional.softplus(self.wa(H)) + self.baserate_mu)
        return nn.functional.softplus(self.w(H)) + self.base_mu

    def decay(self, o, c_bar_1, c_1, delta_1, time_lapse):
        # decay
        c_t = c_bar_1 + (c_1 - c_bar_1) * torch.exp(-delta_1 * time_lapse.unsqueeze(-1))
        # print( o, c_bar_1, c_1, delta_1, time_lapse)
        # k = nn.Embedding(1, 1)(torch.tensor([0]))
        h_t = o * (2 * torch.sigmoid(2 * c_t) - 1)
        return c_t, h_t


    def cell(self, o, c_bar_1, c_1, delta_1, time_lapse, event):
        # decay
        # print(time_lapse)
        c_t, h_t = self.decay(o, c_bar_1, c_1, delta_1, time_lapse)

        # print("h_t")

        # forward
        emb_event_t = nn.Embedding(2, self.hid_dim) (event)
        # print(event, h_t)
        hidden = torch.cat((emb_event_t, h_t), dim=-1) 
        # print(hidden)
        # print(hidden.shape)
        gates = self.W(hidden)
        # print(gates)
        # Gates calculations (5a)-(5d)
        i_1, f_1, z_1, o_1, i_bar_1, f_bar_1, d = torch.chunk(gates, 7, -1)

        i_1 = torch.sigmoid(i_1)
        f_1 = torch.sigmoid(f_1)
        z_1 = 2 * torch.sigmoid(z_1) - 1
        o_1 = torch.sigmoid(o_1)
        d =  nn.functional.softplus(d)
        c_bar_1 = f_bar_1 * c_bar_1 + i_bar_1 * z_1

        # Output (λ_k(t))
        # lambda_k_t = torch.nn.functional.softplus(self.W_k(h_t1))

        return o_1, c_bar_1, c_t, d, h_t

    def scan(self, X, plot = False):
        h_list = []
        output_state_list = []
        # time = X[:,X>0]
        # print(X.shape)
        event = torch.zeros(X.shape, dtype = int) 

        time_duration = torch.diff(X, dim = -len(X.shape)+1,
                                   prepend=torch.zeros((X.shape[0],1),dtype = int) ) 
 
        time_duration[time_duration < 0] = 0

        c, c_bar, o, delta = self.init_states(X.shape[0])

        for i in range(X.shape[1]):
            o, c_bar, c, delta, h = self.cell(o, c_bar, c, delta, time_duration[:,i], event[:,i])
            h_list.append(h)
            output_state_list.append(torch.stack([c, c_bar, o, delta]))
        h_seq = torch.stack(h_list, dim=1)
        output_seq = torch.stack(output_state_list, dim=-2) # [ 4, batch_size, seq_len, self.hid_dim ]
        return h_seq, output_seq

    def mtmc_loss(self, h, output, X):
      """
      Calculate loss using MTMC
      """
      c, c_bar, o, delta = torch.chunk(output, 4, 0)
      c = torch.squeeze(c, 0)
      c_bar = torch.squeeze(c_bar, 0)
      o = torch.squeeze(o, 0)
      delta = torch.squeeze(delta, 0)
      batch_size = X.shape[0]

      lambda_k = self.intensity(h)    # [ batch_size, seq_len+seq_len_buff, self.n_class ]
      # print(lambda_k[:,:,0].shape,lambda_k[X > 0].shape)
      event_log = lambda_k
      event_log = torch.sum(torch.log(lambda_k)[X>0])


      # Calculate simulated loss from MCMC method
      startT = self.T0
      endT = self.T1
      ts_int = torch.linspace(startT, endT, self.config.int_res+1) 
      unit_len = (endT - startT) / self.config.int_res
      ts_int = (ts_int + unit_len / 2)[:-1]   # [ self.config.int_res ]

      time = X
      # print(X)
      time_expand = time.unsqueeze(-1).repeat(1, 1, self.config.int_res)
      # print(time_expand)
    #   print(((time_expand < ts_int) * (time_expand > 0)).sum(-2).shape)
      prev_t_idx = ((time_expand < ts_int) * (time_expand > 0)).sum(-2).long()-1
              # [ batch_size, int_res ]
    #   print(prev_t_idx)

      batch_idx = torch.arange(batch_size).unsqueeze(-1).repeat(1, self.config.int_res) 
      prev_t = time[batch_idx, prev_t_idx]  # [ batch_size, int_res ]
      # print(prev_t)
      time_lapse = ts_int - prev_t  # [ batch_size, int_res ]
    #   print(time_lapse.shape)
    #   print(c[batch_idx, prev_t_idx].shape)

      _, h_d_seq = self.decay(c[batch_idx, prev_t_idx],
                              c_bar[batch_idx, prev_t_idx],
                              o[batch_idx, prev_t_idx],
                              delta[batch_idx, prev_t_idx],
                              time_lapse) # [ batch_size, int_res, self.hid_dim ]

      # print(h_d_seq)
      sim_lambda_k = self.intensity(h_d_seq).transpose(1, 2)
              # [ batch_size, self.n_class, int_res ]
    #   print(sim_lambda_k)
      int_lam = sim_lambda_k.sum() * unit_len
    #   print(int_lam)
      loglikelihood = event_log - int_lam
      # print(loglikelihood.item())
      return -loglikelihood, h_d_seq.detach().cpu().numpy()

    def train(self, X):
        # time = X[X>0].reshape(batch_size,-1)
        X = X 
        mask = X > 0
        # print(X)
        h, output = self.scan(X)
        loss, _ = self.mtmc_loss(h, output, X)
        # print(loss.item())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), h.cpu().detach().numpy()


    def get_H_mat(self, X, plot):
        X = torch.tensor(X, dtype = torch.float32) 
        X = X 
        # print(self.device)
        time = X
        mask = X > 0

        with torch.no_grad():
            h, output = self.scan(X)
            # loss, _ = self.mtmc_loss(h, output, time, mask)
            if plot:
              time_duration = torch.diff(X, dim = -1,
                                         prepend=torch.zeros((X.shape[0],1),
                                                             dtype = int) ) 
              # time_duration = torch.cat((torch.zeros(time.shape[0], 1), time), dim=1)
              time_duration[time_duration < 0] = 0
              fig, axs = plt.subplots(1, 2, figsize=(8, 3))
              duration_plot = time_duration.flatten()
              # print(X.flatten())
              # axs[0].hist(duration_plot[X.flatten()>0].cpu().numpy(), bins=100, color='red', alpha=0.7)
              axs[0].scatter(range(duration_plot[X.flatten()>0].cpu().numpy().shape[0]),
                                  duration_plot[X.flatten()>0].cpu().numpy(), color='red', alpha=0.7, s = 0.5)
              axs[0].set_title('Duration Plot')
              # h_draw = (h.flatten())[X.flatten()>0].cpu().detach().numpy()
              # h_temp = h_draw.flatten()
              # h_plot = h_temp
              # # axs[1].hist(h_plot, bins=100, color='blue', alpha=0.7)
              # axs[1].scatter(range(h_plot.shape[0]), h_plot, color='blue', alpha=0.7, s = 0.5)
              # axs[1].set_title('h_plot')
              plt.show()
            # time.sleep(10)
        return h

    def test(self, X, plot = False):
        # time = X
        X = X 
        mask = X > 0

        with torch.no_grad():
            h, output = self.scan(X)
            loss, _ = self.mtmc_loss(h, output,X)
        if plot:
            # clear_output(wait=True)
            time_duration = torch.diff(X, dim = -1, prepend=torch.zeros((X.shape[0],1),dtype = int) ) 
            # time_duration = torch.cat((torch.zeros(time.shape[0], 1), time), dim=1)
            time_duration[time_duration < 0] = 0
            # clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            duration_plot = time_duration.flatten()
            # print(X.flatten())
            # axs[0].hist(duration_plot[X.flatten()>0].cpu().numpy(), bins=100, color='red', alpha=0.7)
            axs[0].scatter(range(duration_plot[X.flatten()>0].cpu().numpy().shape[0]),
                                 duration_plot[X.flatten()>0].cpu().numpy(), color='red', alpha=0.7, s = 0.5)
            axs[0].set_title('Duration Plot')
            h_draw = (h)[0, X.flatten()>0, :].cpu().detach().numpy()
            h_temp = h_draw
            h_plot = h_temp
            # axs[1].hist(h_plot, bins=100, color='blue', alpha=0.7)
            axs[1].plot(range(h_plot.shape[0]), h_plot, alpha=0.7)
            axs[1].set_title('h_plot')
            plt.tight_layout()
            plt.show()
            # time.sleep(10)
        return loss.item()


    def sample_intensity(self, ts, time, base_mu):
        time = torch.tensor(time, dtype=torch.float32)
        ts = torch.tensor(ts, dtype=torch.float32)
        # print(ts)
        batch_size = time.shape[0]
        with torch.no_grad():
            h, output = self.scan(time)
            c, c_bar, o, delta = torch.chunk(output, 4, 0)
                # [ batch_size, seq_len, self.hid_dim ]
            c = torch.squeeze(c, 0)
            c_bar = torch.squeeze(c_bar, 0)
            o = torch.squeeze(o, 0)
            delta = torch.squeeze(delta, 0)

            grid_len = len(ts)
            time_expand = time.unsqueeze(-1).repeat(1, 1, grid_len)
            # print(time_expand < ts)
            prev_t_idx = ((time_expand < ts) * (time_expand > 0)).sum(-2).long()-1
                                    # [ batch_size, grid_len ]
            batch_idx = torch.arange(batch_size).unsqueeze(-1).repeat(1, grid_len) 
            prev_t = time[batch_idx, prev_t_idx]  # [ batch_size, grid_len ]
            time_lapse = ts - prev_t  # [ batch_size, grid_len ]

            _, h_d_seq = self.decay(c[batch_idx, prev_t_idx],
                                    c_bar[batch_idx, prev_t_idx],
                                    o[batch_idx, prev_t_idx],
                                    delta[batch_idx, prev_t_idx],
                                    time_lapse) # [ batch_size, grid_len, self.hid_dim ]

            lambda_k = self.intensity(h_d_seq)# [ batch_size, grid_len, self.n_class ]
        return lambda_k.sum(-1).cpu().numpy() 


class NHSTPP(nn.Module):
    def __init__(self, config):
        super(NHSTPP, self).__init__()
        self.T0 = config.T0
        self.T1 = config.T1
        self.hid_dim = config.hid_dim
        self.batch_size = config.batch_size
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM cell parameters
        self.W = nn.Linear(2 * self.hid_dim, 7 * self.hid_dim)
        self.emb = nn.Embedding(2, self.hid_dim) 
        self.w = nn.Linear(self.hid_dim, 1)
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)
        self.scheduler = self.set_scheduler(self.optimizer)

    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-2)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == 'Adadelta':
            return torch.optim.Adadelta(self.parameters(), lr=lr)
        elif opt == 'Adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=lr)
        else:
            raise ValueError("No such optimizer!")
        
    def set_scheduler(self, optimizer, mode='max', factor=0.1, patience=5, verbose=True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)
        return scheduler

    def init_states(self, batch_size):
        return (torch.zeros(batch_size, self.hid_dim),
                torch.zeros(batch_size, self.hid_dim),
                torch.zeros(batch_size, self.hid_dim),
                torch.zeros(batch_size, self.hid_dim))

    def intensity(self, H, S):
        """
        Calculate intensity from hidden states and spatial coordinates
        """
        combined = torch.cat((H, S), dim=-1)
        # print(nn.functional.softplus(self.w(H)))
        return nn.functional.softplus(self.w(H))

    def decay(self, o, c_bar_1, c_1, delta_1, time_lapse):
        c_t = c_bar_1 + (c_1 - c_bar_1) * torch.exp(-delta_1 * time_lapse.unsqueeze(-1))
        h_t = o * (2 * torch.sigmoid(2 * c_t) - 1)
        return c_t, h_t

    def cell(self, o, c_bar_1, c_1, delta_1, time_lapse, event):
        c_t, h_t = self.decay(o, c_bar_1, c_1, delta_1, time_lapse)
        emb_event_t = nn.Embedding(2, self.hid_dim)(event)
        hidden = torch.cat((emb_event_t, h_t), dim=-1)
        gates = self.W(hidden)
        i_1, f_1, z_1, o_1, i_bar_1, f_bar_1, d = torch.chunk(gates, 7, -1)
        i_1 = torch.sigmoid(i_1)
        f_1 = torch.sigmoid(f_1)
        z_1 = 2 * torch.sigmoid(z_1) - 1
        o_1 = torch.sigmoid(o_1)
        d = nn.functional.softplus(d)
        c_bar_1 = f_bar_1 * c_bar_1 + i_bar_1 * z_1
        return o_1, c_bar_1, c_t, d, h_t

    def scan(self, X, plot=False):
        h_list = []
        output_state_list = []

        time = X[:, :, 0]
        spatial = X[:, :, 1:]

        event = torch.zeros(time.shape, dtype=int)
        time_duration = torch.diff(time, dim=-1, prepend=torch.zeros((time.shape[0], 1), dtype=int))
        time_duration[time_duration < 0] = 0
        c, c_bar, o, delta = self.init_states(time.shape[0])

        for i in range(time.shape[1]):
            o, c_bar, c, delta, h = self.cell(o, c_bar, c, delta, time_duration[:, i], event[:, i])
            h_list.append(h)
            output_state_list.append(torch.stack([c, c_bar, o, delta]))
        h_seq = torch.stack(h_list, dim=1)
        output_seq = torch.stack(output_state_list, dim=-2)

        return h_seq, output_seq

    def mtmc_loss(self, h, output, X):
        """
        Calculate loss using MTMC for spatio-temporal point processes
        """
        time = X[:, :, 0]
        spatial = X[:, :, 1:]

        c, c_bar, o, delta = torch.chunk(output, 4, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)
        batch_size = time.shape[0]

        lambda_k = self.intensity(h, spatial)  
        event_log = torch.sum(torch.log(lambda_k)[time > 0])

        startT = self.T0
        endT = min(self.T1, time.max())
        ts_int = torch.linspace(startT, endT, self.config.int_res + 1)
        unit_len = (endT - startT) / self.config.int_res
        ts_int = (ts_int + unit_len / 2)[:-1]

        time_expand = time.unsqueeze(-1).repeat(1, 1, self.config.int_res)
        prev_t_idx = ((time_expand < ts_int) * (time_expand > 0)).sum(-2).long()
        batch_idx = torch.arange(batch_size).unsqueeze(-1).repeat(1, self.config.int_res)
        prev_t = time[batch_idx, prev_t_idx]
        time_lapse = ts_int - prev_t

        _, h_d_seq = self.decay(c[batch_idx, prev_t_idx],
                                c_bar[batch_idx, prev_t_idx],
                                o[batch_idx, prev_t_idx],
                                delta[batch_idx, prev_t_idx],
                                time_lapse)

        spatial_grid_x = torch.linspace(0, 1, self.config.spatial_res)
        spatial_grid_y = torch.linspace(0, 1, self.config.spatial_res)
        spatial_grid = torch.stack(torch.meshgrid(spatial_grid_x, spatial_grid_y), -1).view(-1, 2)
        spatial_len = 1.0 / self.config.spatial_res
        unit_area = spatial_len ** 2

        spatial_expand = spatial.unsqueeze(-2).unsqueeze(-2) 
        spatial_lapse = spatial_grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.config.int_res, 1, 1)

        h_d_seq_expand = h_d_seq.unsqueeze(-2).repeat(1, 1, spatial_grid.shape[0], 1)

        lambda_k_spatial = self.intensity(h_d_seq_expand, spatial_lapse).view(batch_size, -1)

        int_lam = lambda_k_spatial.sum() * unit_len * unit_area
        loglikelihood = event_log - int_lam
        return -loglikelihood, h_d_seq.detach().cpu().numpy()


    def train(self, X):
        h, output = self.scan(X)
        loss, _ = self.mtmc_loss(h, output, X)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), h.cpu().detach().numpy()

    def get_H_mat(self, X, plot):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            h, output = self.scan(X)
            if plot:
                time = X[:, :, 0]
                time_duration = torch.diff(time, dim=-1, prepend=torch.zeros((time.shape[0], 1), dtype=int))
                time_duration[time_duration < 0] = 0
                fig, axs = plt.subplots(1, 2, figsize=(8, 3))
                duration_plot = time_duration.flatten()
                axs[0].scatter(range(duration_plot[time.flatten() > 0].cpu().numpy().shape[0]),
                               duration_plot[time.flatten() > 0].cpu().numpy(), color='red', alpha=0.7, s=0.5)
                axs[0].set_title('Duration Plot')
                h_draw = (h)[:, time.flatten() > 0, :].cpu().detach().numpy()
                h_plot = h_draw
                axs[1].plot((range(h_plot[0].shape[0])), 
                            h_plot[0, :, :], alpha=0.7)
                axs[1].set_title('h_plot')
                plt.show()
        return h

    def test(self, X, plot=False):
        with torch.no_grad():
            h, output = self.scan(X)
            loss, _ = self.mtmc_loss(h, output, X)
        if plot:
            time = X[:, :, 0]
            time_duration = torch.diff(time, dim=-1, prepend=torch.zeros((time.shape[0], 1), dtype=int))
            time_duration[time_duration < 0] = 0
            fig, axs = plt.subplots(1, 2, figsize=(8, 3))
            duration_plot = time_duration.flatten()
            axs[0].scatter(range(duration_plot[time.flatten() > 0].cpu().numpy().shape[0]),
                           duration_plot[time.flatten() > 0].cpu().numpy(), color='red', alpha=0.7, s=0.5)
            axs[0].set_title('Duration Plot')
            h_draw = (h)[:, time.flatten() > 0, :].cpu().detach().numpy()
            h_plot = h_draw
            axs[1].plot((range(h_plot[0].shape[0])), 
                        h_plot[0, :, :], alpha=0.7)
            axs[1].set_title('h_plot')
            plt.tight_layout()
            plt.show()
        return loss.item()

    def sample_intensity(self, ts, X):
        ts = torch.tensor(ts, dtype=torch.float32)
        time = X[:, :, 0]
        spatial = X[:, :, 1:]
        batch_size = time.shape[0]
        with torch.no_grad():
            h, output = self.scan(X)
            c, c_bar, o, delta = torch.chunk(output, 4, 0)
            c = torch.squeeze(c, 0)
            c_bar = torch.squeeze(c_bar, 0)
            o = torch.squeeze(o, 0)
            delta = torch.squeeze(delta, 0)

            grid_len = len(ts)
            time_expand = time.unsqueeze(-1).repeat(1, 1)
            prev_t_idx = ((time_expand < ts) * (time_expand > 0)).sum(-2).long() - 1
            batch_idx = torch.arange(batch_size).unsqueeze(-1).repeat(1, grid_len)
            prev_t = time[batch_idx, prev_t_idx]
            time_lapse = ts - prev_t

            _, h_d_seq = self.decay(c[batch_idx, prev_t_idx],
                                    c_bar[batch_idx, prev_t_idx],
                                    o[batch_idx, prev_t_idx],
                                    delta[batch_idx, prev_t_idx],
                                    time_lapse)

            lambda_k = self.intensity(h_d_seq, spatial)
        return lambda_k
