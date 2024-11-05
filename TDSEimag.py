#Importation of the libraries needed
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


#Previously we had L=200, now let's try with L=20
#We define different parameters
a_0 = 1 #Bohr radius in meter(m) ## Should I set this constant to 1? ##
L = 20*a_0 #Length of the spatial domain
x_min = -10*a_0
x_max = 10*a_0
dx = 0.06*a_0 #Spatial step size
d_tau = 0.02 #imaginary Time step size
N = int(L/dx) #Number of spatial points
Nt = int(1/d_tau) #Number of time step
x = np.linspace(x_min, x_max, N) #Spatial grid
hbar = 1  # reduced Planck's constant J=1/s
m_e = 1  # Electron mass in kg
#reduce to L=20

#Parameters that we will use later but which are relevant for making the hamiltonian
w_L = 0.057 #angular frequency
N_c= 10 #cycle
T = N_c * 2 * np.pi / w_L #period
I = 5e13/3.51e16 #intensity
t2=np.linspace(0,T,N)
E0 = np.sqrt(I)
amplitude=E0/(w_L **2)

#We create the functions f(t) and E(t)
#envelope
def f_t(t):
    return np.sin(np.pi*t/T)**2

def f_t2(t):
    return -f_t(t)
#Electric field
def E_t(t):
    return E0 * f_t(t) * np.sin(w_L * t)

#We define the potential function V(x)
def V(x):
    a = 0.816 #a constant parameter
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
def H_dis(t):
    interaction_term = np.diag(x * E_t(t))
    H=S_matrix + V_matrix + + interaction_term
    return H
H_0=H_dis(0)


#We can define now the Crank-Nicolson time evolutions matrices (1±i*dt*H/2)=A or B
def CN_matrices(H,d_tau):
    I = np.eye(N, dtype=int) #the identity matrix
    A = I + (d_tau / 2) * H
    B = I - (d_tau / 2) * H
    return A,B
A,B = CN_matrices(H_0,d_tau)


#1st method: diagonalisation of the hamiltonian
# Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
eigenvalues, eigenvectors = la.eigh(H_0)

# The ground state is the eigenvector corresponding to the lowest eigenvalue
psi1 = eigenvectors[:, 0]
E1 = eigenvalues[0]
print(E1)
# Normalize the ground state wave function
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx)
#Now I want to compare with a second method (that is imaginary time propagation)
# Normalization function
def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

# Project out the ground state component using Gram-Schmidt
def orthogonalize(psi, psi0):
    overlap = np.dot(psi0.conj(), psi) * dx
    psi_orth = psi - overlap * psi0
    return normalize(psi_orth,dx)

# Time evolution using Crank-Nicolson method (Imaginary time)
#Nt : number of time step
def time_evolution(psi, A, B, Nt, dx):
    for _ in range(Nt):
        psi = np.linalg.solve(A, np.dot(B, psi))  # Time evolution step
        psi = normalize(psi,dx)  # Normalize wavefunction
    return psi


def psi_0(N,x, dx, A, B, Nt):
    psi = N * np.exp(-x**2)  # Gaussian wave packet
    psi = normalize(psi, dx)  # Normalize the initial wavefunction
    psi_0 = time_evolution(psi, A, B, Nt, dx)  # Time evolution to find the psi_0
    return psi_0

psi_0 = psi_0(N, x, dx, A, B, Nt)

def compute_energy(psi, H):
    return np.real(np.dot(psi.conj(), np.dot(H, psi)) * dx)

# Define the variable x and the function exp(-x^2)
def hermite_polynomial(n, x):
    if n == 1:
        return np.ones_like(x)  # H_1(x) = 1
    elif n == 2:
        return x  # H_2(x) = x
    elif n == 3:
        return x ** 2 - 1  # H_3(x) = x^2 - 1
    else:
        # Use recurrence relation for n >= 4
        H_n_minus_2 = np.ones_like(x)  # H_1(x) = 1
        H_n_minus_1 = x  # H_2(x) = x

        for k in range(3, n + 1):
            H_n = x * H_n_minus_1 - (k - 1) * H_n_minus_2
            H_n_minus_2 = H_n_minus_1
            H_n_minus_1 = H_n
            return H_n

#n is the n_th states where n=1 is the ground state
def psi(N, x, dx, A, B, Nt, n, computed_states):
    psi = N * hermite_polynomial(n,x)  # Initial guess for psi_2
    psi = normalize(psi, dx)
    for _ in range(Nt):
        psi = time_evolution(psi, A, B, Nt, dx)  # Single time evolution step
        for prev_state in computed_states:
            psi = orthogonalize(psi, prev_state)        # Orthogonalize against psi_i
    return psi

computed_states = []

# Compute psi for the first three states
psi_1 = psi(N, x, dx, A, B, Nt, 1, computed_states)  # Ground state
computed_states.append(psi_1)  # Store computed state

E_1 = compute_energy(psi_1, H_0)
print(E_1)




