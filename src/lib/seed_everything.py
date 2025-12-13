import random
import os
import numpy as np
import torch

def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.
    """
    # 1. Set seed for Python's built-in random module
    random.seed(seed)
    
    # 2. Set seed for Python's hash randomization (dictionaries, sets, etc.)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Set seed for NumPy (which also covers Pandas)
    np.random.seed(seed)
    
    # 4. Set seed for PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 5. Set seed for PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # 6. Configure PyTorch to be deterministic
    # Note: This may impact performance but is necessary for 100% reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Usage
seed_everything(42)