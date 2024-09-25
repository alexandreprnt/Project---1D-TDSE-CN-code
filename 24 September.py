#Importation of the libraries needed
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#We define different parameters
a_0 = 5.2917721090e-11 #Bohr radius in meter(m) ## Should I set this constant to 1? ##
L = 200*a_0 #Length of the spatial domain
x_min = -100*a_0
x_max = 100*a_0
dx = 0.06*a_0 #Spatial step size
d_tau = 0.02 #imaginary Time step size
N = int(L/dx) #Number of spatial points
x = np.linspace(x_min, x_max, N) #Spatial grid
hbar = 1  # reduced Planck's constant J=1/s
m_e = 9.10938356e-31  # Electron mass in kg

#We define the potential function V(x)
def V(x):
    a = 0.81 #a constant parameter
    V =-1/np.sqrt(x**2 + a**2) #Potential formula
    return V

#We create the potential matrix where only the diagonal elements are non zero
def potential_matrix(V):
    V_matrix = np.diag(V)
    return V_matrix
V_matrix = potential_matrix(V(x))

#We define the spatial discretization matrix (tridiagonal matrix)
def spatialdiscr_matrix(N, dx):
    diagonal = -2 * np.ones(N)
    off_diagonal = np.ones(N - 1)
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    matrix= (-1/(2*dx**2))*matrix
    return matrix
S_matrix = spatialdiscr_matrix(N,dx)

#We define the discretized hamiltonian
def H_dis():
    H=S_matrix+V_matrix
    return H
H=H_dis()
print(H)

#We can define now the Crank-Nicolson time evolutions matrices (1±i*dt*H/2)=A or B
def CN_matrices(H,d_tau):
    I = np.eye(N, dtype=int) #the identity matrix
    A = I + (d_tau / 2) * H
    B = I - (d_tau / 2) * H
    return A,B
A,B = CN_matrices(H,d_tau)

# Normalization function
def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

# Time evolution using Crank-Nicolson method (Imaginary time)
#Nt : number of time step
def time_evolution(psi, A, B, Nt):
    for _ in range(Nt):
        psi = la.solve(A, np.dot(B, psi))
        psi = normalize(psi, dx)  # Normalize the wavefunction at each step
    return psi

## How should I define psi_0? As a sum of c_n and psi_n?##














