import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Define the parameters
a_0 = 1  # Bohr radius in meter (m)
L = 20 * a_0  # Length of the spatial domain
x_min = -10 * a_0
x_max = 10 * a_0
dx = 0.06 * a_0  # Spatial step size
dt = 0.02  # Real time step size
N = int(L / dx)  # Number of spatial points
Nt = int(1 / dt)  # Number of time steps
x = np.linspace(x_min, x_max, N)  # Spatial grid
hbar = 1  # Reduced Planck's constant J*s
m_e = 1  # Electron mass in kg
computed_states = []  # To store computed wavefunctions over time
x_expectation_values = []  # To store expectation values of x

# Define the potential function V(x)
def V(x):
    a = 0.816  # Constant parameter
    return -1 / np.sqrt(x ** 2 + a ** 2)  # Potential formula

# Create the potential matrix (diagonal matrix with potential values)
def potential_matrix(V):
    return np.diag(V)

V_matrix = potential_matrix(V(x))

# Define the spatial discretization matrix (tridiagonal matrix)
def spatialdiscr_matrix(N, dx):
    diagonal = -2 * np.ones(N)
    off_diagonal = np.ones(N - 1)
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return (-1 / (2 * dx ** 2)) * matrix

S_matrix = spatialdiscr_matrix(N, dx)

# Define the discretized Hamiltonian
def H_dis():
    return S_matrix + V_matrix

H = H_dis()

# Crank-Nicolson matrices for real-time evolution (1 ± i * dt * H / 2)
def CN_matrices(H, dt):
    I = np.eye(N, dtype=complex)  # The identity matrix
    A = I + 1j * (dt / 2) * H  # Matrix A for real-time evolution
    B = I - 1j * (dt / 2) * H  # Matrix B for real-time evolution
    return A, B

A, B = CN_matrices(H, dt)

# Function to normalize the wavefunction
def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

def time_evolution(psi, A, B, Nt, dx):
    for _ in range(Nt):
        psi = np.linalg.solve(A, np.dot(B, psi))  # Time evolution step
        psi = normalize(psi, dx)  # Normalize wavefunction
        computed_states.append(psi)  # Append the wavefunction to computed_states
    return psi

# Define the initial wave packet (Gaussian)
def psi_0(N, x, dx, A, B, Nt):
    psi = N * np.exp(-x ** 2)  # Gaussian wave packet
    psi = normalize(psi, dx)  # Normalize the initial wavefunction
    computed_states.append(psi)  # Append the initial wavefunction
    for _ in range(Nt):
        psi = time_evolution(psi, A, B, 1, dx)  # Evolve in time, step by step
    return psi

#We compute the expectation value <psi(x,0)|H_0|psi(x,0)>
def psi0(x):
    psi= psi_0(N, x, dx, A, B, Nt)
    return psi
expectation_value=np.sum(dx * np.conj(psi0(x)) * V(x) * psi0(x))
second_derivative=(psi0(x)[2:] - 2 * psi0(x)[1:-1] + psi0(x)[:-2]) / dx**2
expectation_value= expectation_value - 1/2 * dx * np.sum(psi0(x)[1:-1] * second_derivative)
print(f'The expectation value is : <H_0> = {expectation_value:.15f}')

t=np.linspace(0,1,Nt)
psi_xt=psi = np.exp(-1j * expectation_value * t[:, np.newaxis]) * psi0(x)

#We check that <psi(x)|psi(x)>=1
val=np.sum(dx * np.abs(psi0(x))**2 )
print(val)

