import numpy as np

def ising_energy(spins, J):
    return -0.5 * np.sum(J * np.outer(spins, spins))

def metropolis_step(spins, J, beta=1.0):
    N = len(spins)
    i = np.random.randint(N)

    flipped = spins.copy()
    flipped[i] *= -1

    dE = ising_energy(flipped, J) - ising_energy(spins, J)

    if dE < 0 or np.random.rand() < np.exp(-beta * dE):
        spins = flipped
    return spins

def simulate_ising(J, steps=5000, beta=1.0):
    spins = np.random.choice([-1, 1], size=J.shape[0])
    magnetization = []

    for _ in range(steps):
        spins = metropolis_step(spins, J, beta)
        magnetization.append(np.mean(spins))

    return np.array(magnetization)
