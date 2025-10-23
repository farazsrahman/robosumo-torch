# Import PyTorch policies
try:
    from .policy import LSTMPolicy
    from .policy import MLPPolicy
except ImportError:
    # Dependencies not available
    pass
