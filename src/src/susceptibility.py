import numpy as np

def susceptibility(returns, window=50):
    chi = []
    for i in range(window, len(returns)):
        C = np.corrcoef(returns.iloc[i-window:i].T)
        chi.append(np.sum(C**2))
    return np.array(chi)
