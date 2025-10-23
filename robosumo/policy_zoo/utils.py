"""
A variety of utilities (PyTorch version).
This is a PyTorch implementation of the utilities from utils_modern.py.
"""
import numpy as np
import torch
import torch.nn as nn


def dense(x, weight, bias=None):
    """
    Linear layer using PyTorch.
    
    Args:
        x: Input tensor
        weight: Weight matrix
        bias: Optional bias vector
    
    Returns:
        Output tensor
    """
    output = torch.matmul(x, weight)
    if bias is not None:
        output = output + bias
    return output


class DiagonalGaussian:
    """Diagonal Gaussian distribution for stochastic policies."""
    
    def __init__(self, mean, logstd):
        """
        Initialize DiagonalGaussian.
        
        Args:
            mean: Mean of the distribution
            logstd: Log standard deviation
        """
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)
    
    def sample(self):
        """Sample from the distribution."""
        return self.mean + self.std * torch.randn_like(self.mean)
    
    def mode(self):
        """Return the mode (mean) of the distribution."""
        return self.mean


class RunningMeanStd(nn.Module):
    """
    Running mean and standard deviation tracker.
    Uses buffers so statistics are properly saved/loaded with the model.
    """
    
    def __init__(self, epsilon=1e-2, shape=()):
        """
        Initialize RunningMeanStd.
        
        Args:
            epsilon: Small constant for numerical stability
            shape: Shape of the statistics to track
        """
        super().__init__()
        self.epsilon = epsilon
        self.shape = shape if isinstance(shape, tuple) else tuple(shape)
        
        # Use register_buffer so these are saved with the model but not trained
        if self.shape == () or self.shape == (0,):
            # Scalar case
            self.register_buffer('_sum', torch.tensor(0.0))
            self.register_buffer('_sumsq', torch.tensor(epsilon))
            self.register_buffer('_count', torch.tensor(epsilon))
        else:
            # Vector case
            self.register_buffer('_sum', torch.zeros(self.shape))
            self.register_buffer('_sumsq', torch.full(self.shape, epsilon))
            self.register_buffer('_count', torch.tensor(epsilon))
    
    def forward(self):
        """
        Compute current mean and standard deviation.
        
        Returns:
            Tuple of (mean, std)
        """
        mean = self._sum / self._count
        var_est = (self._sumsq / self._count) - mean ** 2
        std = torch.sqrt(torch.maximum(var_est, torch.tensor(1e-2, device=var_est.device)))
        return mean, std
    
    def __call__(self):
        """Allow calling the module directly."""
        return self.forward()


def load_params(path):
    """
    Load parameters from a numpy file.
    
    Args:
        path: Path to the .npy file
    
    Returns:
        Numpy array of parameters
    """
    return np.load(path)


def load_from_tf_params(policy, tf_params):
    """
    Load TensorFlow flat parameters into PyTorch MLPPolicy.
    
    TensorFlow saves weights as a FLAT numpy array. We need to extract and reshape
    each parameter according to the expected sizes.
    
    Order of parameters in flat array:
    1. Normalization params (if normalize=True):
       - ret_rms: sum (), sumsq (), count ()
       - ob_rms: sum (obs_dim,), sumsq (obs_dim,), count ()
    2. Value network:
       - vffc1/w (obs_dim, 64), vffc1/b (64,)
       - vffc2/w (64, 64), vffc2/b (64,)
       - vffinal/w (64, 1), vffinal/b (1,)
    3. Policy network:
       - polfc1/w (obs_dim, 64), polfc1/b (64,)
       - polfc2/w (64, 64), polfc2/b (64,)
       - polfinal/w (64, act_dim), polfinal/b (act_dim,)
       - logstd (1, act_dim)
    
    PyTorch Linear layers store weights transposed compared to TensorFlow.
    TensorFlow: weight is (input_dim, output_dim)
    PyTorch Linear: weight is (output_dim, input_dim)
    
    Args:
        policy: PyTorch MLPPolicy instance
        tf_params: Flat numpy array of all TensorFlow parameters
    """
    idx = 0
    
    # Get dimensions from policy
    ob_dim = policy.vf_fc1.in_features
    hidden1 = policy.vf_fc1.out_features  # Should be 64
    hidden2 = policy.vf_fc2.out_features  # Should be 64
    act_dim = policy.pol_final.out_features
    
    # Load normalization parameters if present
    if policy.normalized:
        # ret_rms: sum (), sumsq (), count () - all scalars
        policy.ret_rms._sum.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        policy.ret_rms._sumsq.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        policy.ret_rms._count.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        
        # ob_rms: sum (ob_dim,), sumsq (ob_dim,), count ()
        policy.ob_rms._sum.copy_(torch.from_numpy(tf_params[idx:idx+ob_dim]).float())
        idx += ob_dim
        policy.ob_rms._sumsq.copy_(torch.from_numpy(tf_params[idx:idx+ob_dim]).float())
        idx += ob_dim
        policy.ob_rms._count.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
    
    # Load value network
    # vffc1/w (ob_dim, hidden1)
    w_size = ob_dim * hidden1
    vffc1_w = tf_params[idx:idx+w_size].reshape(ob_dim, hidden1)
    policy.vf_fc1.weight.data.copy_(torch.from_numpy(vffc1_w.T).float())
    idx += w_size
    
    # vffc1/b (hidden1,)
    policy.vf_fc1.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+hidden1]).float())
    idx += hidden1
    
    # vffc2/w (hidden1, hidden2)
    w_size = hidden1 * hidden2
    vffc2_w = tf_params[idx:idx+w_size].reshape(hidden1, hidden2)
    policy.vf_fc2.weight.data.copy_(torch.from_numpy(vffc2_w.T).float())
    idx += w_size
    
    # vffc2/b (hidden2,)
    policy.vf_fc2.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+hidden2]).float())
    idx += hidden2
    
    # vffinal/w (hidden2, 1)
    w_size = hidden2 * 1
    vffinal_w = tf_params[idx:idx+w_size].reshape(hidden2, 1)
    policy.vf_final.weight.data.copy_(torch.from_numpy(vffinal_w.T).float())
    idx += w_size
    
    # vffinal/b (1,)
    policy.vf_final.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+1]).float())
    idx += 1
    
    # Load policy network
    # polfc1/w (ob_dim, hidden1)
    w_size = ob_dim * hidden1
    polfc1_w = tf_params[idx:idx+w_size].reshape(ob_dim, hidden1)
    policy.pol_fc1.weight.data.copy_(torch.from_numpy(polfc1_w.T).float())
    idx += w_size
    
    # polfc1/b (hidden1,)
    policy.pol_fc1.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+hidden1]).float())
    idx += hidden1
    
    # polfc2/w (hidden1, hidden2)
    w_size = hidden1 * hidden2
    polfc2_w = tf_params[idx:idx+w_size].reshape(hidden1, hidden2)
    policy.pol_fc2.weight.data.copy_(torch.from_numpy(polfc2_w.T).float())
    idx += w_size
    
    # polfc2/b (hidden2,)
    policy.pol_fc2.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+hidden2]).float())
    idx += hidden2
    
    # polfinal/w (hidden2, act_dim)
    w_size = hidden2 * act_dim
    polfinal_w = tf_params[idx:idx+w_size].reshape(hidden2, act_dim)
    policy.pol_final.weight.data.copy_(torch.from_numpy(polfinal_w.T).float())
    idx += w_size
    
    # polfinal/b (act_dim,)
    policy.pol_final.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+act_dim]).float())
    idx += act_dim
    
    # logstd (1, act_dim)
    logstd_size = 1 * act_dim
    logstd = tf_params[idx:idx+logstd_size].reshape(1, act_dim)
    policy.logstd.data.copy_(torch.from_numpy(logstd).float())
    idx += logstd_size


def load_lstm_from_tf_params(policy, tf_params):
    """
    Load TensorFlow flat parameters into PyTorch LSTMPolicy.
    
    Order of parameters in flat array:
    1. Normalization params (if normalize=True):
       - ret_rms: sum (), sumsq (), count ()
       - ob_rms: sum (obs_dim,), sumsq (obs_dim,), count ()
    2. Embedding layer:
       - embed_fc/w (obs_dim, embed_dim)
       - embed_fc/b (embed_dim,)
    3. Value LSTM:
       - lstmv/kernel (embed_dim, 4*lstm_dim) - input weights
       - lstmv/recurrent_kernel (lstm_dim, 4*lstm_dim) - hidden weights
       - lstmv/bias (4*lstm_dim,)
    4. Value final layer:
       - vf_final/w (lstm_dim, 1)
       - vf_final/b (1,)
    5. Policy LSTM:
       - lstmp/kernel (embed_dim, 4*lstm_dim)
       - lstmp/recurrent_kernel (lstm_dim, 4*lstm_dim)
       - lstmp/bias (4*lstm_dim,)
    6. Policy final layer:
       - pol_final/w (lstm_dim, act_dim)
       - pol_final/b (act_dim,)
    7. Logstd:
       - logstd (1, act_dim)
    
    Args:
        policy: PyTorch LSTMPolicy instance
        tf_params: Flat numpy array of all TensorFlow parameters
    """
    idx = 0
    
    # Get dimensions
    ob_dim = policy.embed_fc.in_features
    embed_dim = policy.embed_fc.out_features
    lstm_dim = policy.lstm_dim
    act_dim = policy.pol_final.out_features
    
    # Load normalization parameters if present
    if policy.normalized:
        # ret_rms: sum (), sumsq (), count ()
        policy.ret_rms._sum.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        policy.ret_rms._sumsq.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        policy.ret_rms._count.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
        
        # ob_rms: sum (ob_dim,), sumsq (ob_dim,), count ()
        policy.ob_rms._sum.copy_(torch.from_numpy(tf_params[idx:idx+ob_dim]).float())
        idx += ob_dim
        policy.ob_rms._sumsq.copy_(torch.from_numpy(tf_params[idx:idx+ob_dim]).float())
        idx += ob_dim
        policy.ob_rms._count.copy_(torch.tensor(float(tf_params[idx])))
        idx += 1
    
    # Load embedding layer
    w_size = ob_dim * embed_dim
    embed_w = tf_params[idx:idx+w_size].reshape(ob_dim, embed_dim)
    policy.embed_fc.weight.data.copy_(torch.from_numpy(embed_w.T).float())
    idx += w_size
    
    b_size = embed_dim
    policy.embed_fc.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+b_size]).float())
    idx += b_size
    
    # Load Value LSTM
    # kernel (embed_dim, 4*lstm_dim) -> weight_ih (4*lstm_dim, embed_dim)
    w_size = embed_dim * 4 * lstm_dim
    kernel = tf_params[idx:idx+w_size].reshape(embed_dim, 4*lstm_dim)
    policy.vf_lstm.weight_ih.data.copy_(torch.from_numpy(kernel.T).float())
    idx += w_size
    
    # recurrent_kernel (lstm_dim, 4*lstm_dim) -> weight_hh (4*lstm_dim, lstm_dim)
    w_size = lstm_dim * 4 * lstm_dim
    recurrent_kernel = tf_params[idx:idx+w_size].reshape(lstm_dim, 4*lstm_dim)
    policy.vf_lstm.weight_hh.data.copy_(torch.from_numpy(recurrent_kernel.T).float())
    idx += w_size
    
    # bias (4*lstm_dim,)
    # TensorFlow uses single bias, PyTorch uses bias_ih and bias_hh
    # We set both to half the TensorFlow bias
    b_size = 4 * lstm_dim
    bias = torch.from_numpy(tf_params[idx:idx+b_size]).float()
    policy.vf_lstm.bias_ih.data.copy_(bias / 2)
    policy.vf_lstm.bias_hh.data.copy_(bias / 2)
    idx += b_size
    
    # Load Value final layer
    w_size = lstm_dim * 1
    vf_final_w = tf_params[idx:idx+w_size].reshape(lstm_dim, 1)
    policy.vf_final.weight.data.copy_(torch.from_numpy(vf_final_w.T).float())
    idx += w_size
    
    b_size = 1
    policy.vf_final.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+b_size]).float())
    idx += b_size
    
    # Load Policy LSTM
    # kernel (embed_dim, 4*lstm_dim) -> weight_ih (4*lstm_dim, embed_dim)
    w_size = embed_dim * 4 * lstm_dim
    kernel = tf_params[idx:idx+w_size].reshape(embed_dim, 4*lstm_dim)
    policy.pol_lstm.weight_ih.data.copy_(torch.from_numpy(kernel.T).float())
    idx += w_size
    
    # recurrent_kernel (lstm_dim, 4*lstm_dim) -> weight_hh (4*lstm_dim, lstm_dim)
    w_size = lstm_dim * 4 * lstm_dim
    recurrent_kernel = tf_params[idx:idx+w_size].reshape(lstm_dim, 4*lstm_dim)
    policy.pol_lstm.weight_hh.data.copy_(torch.from_numpy(recurrent_kernel.T).float())
    idx += w_size
    
    # bias (4*lstm_dim,)
    b_size = 4 * lstm_dim
    bias = torch.from_numpy(tf_params[idx:idx+b_size]).float()
    policy.pol_lstm.bias_ih.data.copy_(bias / 2)
    policy.pol_lstm.bias_hh.data.copy_(bias / 2)
    idx += b_size
    
    # Load Policy final layer
    w_size = lstm_dim * act_dim
    pol_final_w = tf_params[idx:idx+w_size].reshape(lstm_dim, act_dim)
    policy.pol_final.weight.data.copy_(torch.from_numpy(pol_final_w.T).float())
    idx += w_size
    
    b_size = act_dim
    policy.pol_final.bias.data.copy_(torch.from_numpy(tf_params[idx:idx+b_size]).float())
    idx += b_size
    
    # Load logstd
    logstd_size = 1 * act_dim
    logstd = tf_params[idx:idx+logstd_size].reshape(1, act_dim)
    policy.logstd.data.copy_(torch.from_numpy(logstd).float())
    idx += logstd_size

