# RENAL Goodness-of-Fit Test
Code for the REcurrent NeurAL (RENAL) Goodness-of-Fit test proposed in the paper *Recurrent Neural Goodness-of-Fit Test for Time Series* by Aoran Zhang, Wenbin Zhou, Liyan Xie, and Shixiang Zhu.

See below for information about the framework and implementations.

## Architecture of RENAL Framework
![architecture](https://github.com/user-attachments/assets/1451d019-6898-4a24-a992-aa8d2f6aa6ee)
Real-world observations are compared to model-generated sequences, with darker blue indicating better fits. We first use a recurrent neural network $\phi$ to extract conditionally independent history embeddings. We then construct their transition probability matrices using these embeddings and evaluate the fit with a chi-square discrepancy test. A tutorial on using RENAL can be found in `example_usage.ipynb`.

## File Usage
- `nn_model.py` includes neural network models used in the framework, such as RNNs and LSTMs.
- `tpp.py` and `ts_data.py`  contain codes for generating and visualizing temporal point processes and time series data.
- `embedding_binning.py` implements the embedding binning process to transform continuous state spaces into discrete state spaces and estimate transition probability matrices.
- `example_usage.ipynb` demonstrates how to use the framework with example data.

- `ksd.py`/`mmd.py`/`linear_time.py`/`kernelgof.py`/`util.py` contain baseline implementations for kernel-based goodness-of-fit tests.

## Reference
[Zhang A, Zhou W, Xie L, Zhu S. Recurrent Neural Goodness-of-Fit Test for Time Series](https://arxiv.org/abs/2410.13986).
