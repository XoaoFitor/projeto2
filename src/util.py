import numpy as np, random, os

def set_seed(seed=42)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
