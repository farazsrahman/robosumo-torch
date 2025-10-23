"""
Policy classes (PyTorch version with GPU support).
This is a PyTorch implementation of the policies from policy_modern.py.
"""
import torch
import torch.nn as nn
import numpy as np
import logging

# Try relative import first, fall back to absolute import
try:
    from .utils import RunningMeanStd, DiagonalGaussian
except ImportError:
    import os
    import importlib.util
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("utils", os.path.join(current_dir, "utils.py"))
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    RunningMeanStd = utils_module.RunningMeanStd
    DiagonalGaussian = utils_module.DiagonalGaussian


class Policy:
    """Base policy class."""
    
    def reset(self, **kwargs):
        pass
    
    def act(self, observation):
        raise NotImplementedError


class MLPPolicy(nn.Module, Policy):
    """
    Multi-Layer Perceptron policy with separate value and policy networks.
    
    Architecture:
    - Value network: obs → FC(64) → FC(64) → FC(1)
    - Policy network: obs → FC(64) → FC(64) → FC(action_dim)
    - All hidden layers use tanh activation
    - Optional observation normalization with running mean/std
    """
    
    def __init__(self, ob_space, ac_space, hiddens=[64, 64], normalize=False, device=None):
        """
        Initialize MLPPolicy.
        
        Args:
            ob_space: Observation space (gym.spaces.Box)
            ac_space: Action space (gym.spaces.Box)
            hiddens: List of hidden layer sizes (default: [64, 64])
            normalize: Whether to use observation normalization
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
        """
        super().__init__()
        
        self.recurrent = False
        self.normalized = normalize
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Extract dimensions
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.shape[0]
        
        # Observation normalization
        if self.normalized:
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
            self.ret_rms = RunningMeanStd(shape=())
        
        # Value network
        self.vf_fc1 = nn.Linear(ob_dim, hiddens[0])
        self.vf_fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.vf_final = nn.Linear(hiddens[1], 1)
        
        # Policy network
        self.pol_fc1 = nn.Linear(ob_dim, hiddens[0])
        self.pol_fc2 = nn.Linear(hiddens[0], hiddens[1])
        self.pol_final = nn.Linear(hiddens[1], ac_dim)
        
        # Log standard deviation (learned parameter)
        self.logstd = nn.Parameter(torch.zeros(1, ac_dim))
        
        # Move to device
        self.to(self.device)
    
    @torch.no_grad()
    def act(self, observation, stochastic=True):
        """
        Generate action from observation.
        
        Args:
            observation: Numpy array of shape (obs_dim,)
            stochastic: Whether to sample stochastically or return mean
        
        Returns:
            Tuple of (action, info_dict)
            - action: Numpy array of shape (action_dim,)
            - info: Dict with 'vpred' key
        """
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs = torch.from_numpy(observation).float()
        else:
            obs = observation.float()
        
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move to device
        obs = obs.to(self.device)
        
        # Normalize observation if enabled
        if self.normalized:
            mean, std = self.ob_rms()
            obs = torch.clamp((obs - mean) / std, -5.0, 5.0)
        
        # Value network forward pass
        v = torch.tanh(self.vf_fc1(obs))
        v = torch.tanh(self.vf_fc2(v))
        vpredz = self.vf_final(v).squeeze()
        
        # Apply return normalization if enabled
        if self.normalized:
            ret_mean, ret_std = self.ret_rms()
            vpred = vpredz * ret_std + ret_mean
        else:
            vpred = vpredz
        
        # Policy network forward pass
        p = torch.tanh(self.pol_fc1(obs))
        p = torch.tanh(self.pol_fc2(p))
        mean = self.pol_final(p)
        
        # Create diagonal Gaussian distribution
        pd = DiagonalGaussian(mean, self.logstd)
        
        # Sample action
        if stochastic:
            action = pd.sample()
        else:
            action = pd.mode()
        
        # Convert to numpy and remove batch dimension
        action_np = action.squeeze().cpu().numpy()
        vpred_item = vpred.item() if vpred.dim() == 0 else vpred[0].item()
        
        return action_np, {'vpred': vpred_item}
    
    def get_device(self):
        """Return the device this policy is on."""
        return self.device


class LSTMPolicy(nn.Module, Policy):
    """
    LSTM policy with separate value and policy networks.
    
    Architecture:
    - Embedding layer (FC): obs_dim -> hiddens[0]
    - Value LSTM: hiddens[0] -> hiddens[1]
    - Value head: hiddens[1] -> 1
    - Policy LSTM: hiddens[0] -> hiddens[1]  
    - Policy head: hiddens[1] -> act_dim
    """
    
    def __init__(self, ob_space, ac_space, hiddens=[64, 64], normalize=False, device=None):
        """
        Initialize LSTMPolicy.
        
        Args:
            ob_space: Observation space (gym.spaces.Box)
            ac_space: Action space (gym.spaces.Box)
            hiddens: List of hidden layer sizes (default: [64, 64])
                     hiddens[0] = embedding dimension
                     hiddens[1] = LSTM hidden size
            normalize: Whether to use observation normalization
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
        """
        super().__init__()
        
        self.recurrent = True
        self.normalized = normalize
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Extract dimensions
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.shape[0]
        embed_dim = hiddens[0]
        lstm_dim = hiddens[1]
        
        # Observation normalization
        if self.normalized:
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
            self.ret_rms = RunningMeanStd(shape=())
        
        # Embedding layer (shared)
        self.embed_fc = nn.Linear(ob_dim, embed_dim)
        
        # Value network
        self.vf_lstm = nn.LSTMCell(embed_dim, lstm_dim)
        self.vf_final = nn.Linear(lstm_dim, 1)
        
        # Policy network
        self.pol_lstm = nn.LSTMCell(embed_dim, lstm_dim)
        self.pol_final = nn.Linear(lstm_dim, ac_dim)
        
        # Log standard deviation (learned parameter)
        self.logstd = nn.Parameter(torch.zeros(1, ac_dim))
        
        # Initialize LSTM states
        # State is [c_value, h_value, c_policy, h_policy]
        self.zero_state = np.zeros(4 * lstm_dim, dtype=np.float32)
        self.state = self.zero_state.copy()
        self.lstm_dim = lstm_dim
        
        # Move to device
        self.to(self.device)
    
    @torch.no_grad()
    def act(self, observation, stochastic=True):
        """
        Generate action from observation.
        
        Args:
            observation: Numpy array of shape (obs_dim,)
            stochastic: Whether to sample stochastically or return mean
        
        Returns:
            Tuple of (action, info_dict)
            - action: Numpy array of shape (action_dim,)
            - info: Dict with 'vpred' and 'state' keys
        """
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            obs = torch.from_numpy(observation).float()
        else:
            obs = observation.float()
        
        # Add batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move to device
        obs = obs.to(self.device)
        
        # Normalize observation if enabled
        if self.normalized:
            mean, std = self.ob_rms()
            obs = torch.clamp((obs - mean) / std, -5.0, 5.0)
        
        # Embedding
        embed = torch.tanh(self.embed_fc(obs))
        
        # Extract LSTM states from numpy state
        lstm_dim = self.lstm_dim
        c_v = torch.from_numpy(self.state[0:lstm_dim]).float().unsqueeze(0).to(self.device)
        h_v = torch.from_numpy(self.state[lstm_dim:2*lstm_dim]).float().unsqueeze(0).to(self.device)
        c_p = torch.from_numpy(self.state[2*lstm_dim:3*lstm_dim]).float().unsqueeze(0).to(self.device)
        h_p = torch.from_numpy(self.state[3*lstm_dim:4*lstm_dim]).float().unsqueeze(0).to(self.device)
        
        # Value LSTM forward pass
        h_v_new, c_v_new = self.vf_lstm(embed, (h_v, c_v))
        vpredz = self.vf_final(h_v_new).squeeze()
        
        # Apply return normalization if enabled
        if self.normalized:
            ret_mean, ret_std = self.ret_rms()
            vpred = vpredz * ret_std + ret_mean
        else:
            vpred = vpredz
        
        # Policy LSTM forward pass
        h_p_new, c_p_new = self.pol_lstm(embed, (h_p, c_p))
        mean = self.pol_final(h_p_new)
        
        # Create diagonal Gaussian distribution
        pd = DiagonalGaussian(mean, self.logstd)
        
        # Sample action
        if stochastic:
            action = pd.sample()
        else:
            action = pd.mode()
        
        # Update internal state
        new_state = np.concatenate([
            c_v_new.squeeze().cpu().numpy(),
            h_v_new.squeeze().cpu().numpy(),
            c_p_new.squeeze().cpu().numpy(),
            h_p_new.squeeze().cpu().numpy(),
        ])
        self.state = new_state
        
        # Convert to numpy and remove batch dimension
        action_np = action.squeeze().cpu().numpy()
        vpred_item = vpred.item() if vpred.dim() == 0 else vpred[0].item()
        
        return action_np, {'vpred': vpred_item, 'state': new_state}
    
    def reset(self):
        """Reset LSTM hidden state."""
        self.state = self.zero_state.copy()
    
    def get_device(self):
        """Return the device this policy is on."""
        return self.device

