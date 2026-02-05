import numpy as np
from scipy.stats import entropy

def rolling_entropy(returns, window=50):
    entropies = []
    for i in range(window, len(returns)):
        hist, _ = np.histogram(
            returns.iloc[i-window:i].values.flatten(),
            bins=50,
            density=True
        )
        entropies.append(entropy(hist + 1e-12))
    return np.array(entropies)
