import numpy as np

def rmt_filter(returns):
    T, N = returns.shape
    Q = T / N

    C = np.corrcoef(returns.T)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    lambda_plus = (1 + np.sqrt(1/Q))**2
    lambda_minus = (1 - np.sqrt(1/Q))**2

    filtered_eigenvalues = [
        ev if (ev > lambda_plus or ev < lambda_minus) else 0
        for ev in eigenvalues
    ]

    C_filtered = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

    return C, C_filtered, eigenvalues, lambda_plus, lambda_minus
