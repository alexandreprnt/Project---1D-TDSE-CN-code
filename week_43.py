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
#We define the potential function V(x)
def V(x):
    a = 0.816 #a constant parameter
    V =-1/np.sqrt(x**2 + a**2) #Potential formula
    return V
#We can plot the potential function to see how it is behaving
plt.plot(x, V(x), label="Potential V(x)")
plt.xlabel('x')
plt.ylabel('V(x)')
plt.title('Potential Function V(x)')
plt.grid(True)
plt.legend()
plt.show()
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


#We can define now the Crank-Nicolson time evolutions matrices (1±i*dt*H/2)=A or B
def CN_matrices(H,d_tau):
    I = np.eye(N, dtype=int) #the identity matrix
    A = I + (d_tau / 2) * H
    B = I - (d_tau / 2) * H
    return A,B
A,B = CN_matrices(H,d_tau)


#1st method: diagonalisation of the hamiltonian
# Diagonalize the Hamiltonian to get eigenvalues and eigenvectors
eigenvalues, eigenvectors = la.eigh(H)

# The ground state is the eigenvector corresponding to the lowest eigenvalue
psi1 = eigenvectors[:, 0]
E1 = eigenvalues[0]
# Normalize the ground state wave function
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx)
#We can now do also the same for the first excited state (E_2, psi_2) and the second excited state (E_3, psi_3)
E2, E3 = eigenvalues[1], eigenvalues[2]
psi2, psi3 =eigenvectors[:, 1], eigenvectors[:, 2]
#We normalize it
psi2 /= np.sqrt(np.sum(np.abs(psi2)**2) * dx)
psi3 /= np.sqrt(np.sum(np.abs(psi3)**2) * dx)

# We plot the potential and the wave functions of the first three states
plt.figure(figsize=(10, 8))
plt.plot(x, V(x), 'k', label="Potential V(x)")
plt.plot(x, np.abs(psi1)**2, label=f'Ground State |ψ₁(x)|² (E₁ = {E1:.4f})')
plt.plot(x, np.abs(psi2)**2, label=f'1st Excited State |ψ₂(x)|² (E₂ = {E2:.4f})')
plt.plot(x, np.abs(psi3)**2, label=f'2nd Excited State |ψ₃(x)|² (E₃ = {E3:.4f})')

# Plot formatting
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('First Three Eigenstates of the Hamiltonian by diagonalising it')
plt.legend()
plt.grid(True)
plt.show()

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
    psi = N*np.exp(-x ** 2)  # Gaussian wave packet
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

psi_2 = psi(N, x, dx, A, B, Nt, 2, computed_states)  # First excited state
computed_states.append(psi_2)

psi_3 = psi(N, x, dx, A, B, Nt, 3, computed_states)  # First excited state
computed_states.append(psi_3)

E_1 = compute_energy(psi_1, H)
E_2 = compute_energy(psi_2, H)
E_3 = compute_energy(psi_3, H)
print(f'First three energy states using diagonalization: E1 = {E1:.15f}, E2 = {E2:.15f}, E3 = {E3:.15f}')
print(f'First three energy states using imaginary time: E1 = {E_1:.15f}, E2 = {E_2:.15f}, E3 = {E_3:.15f}')
plt.figure(figsize=(10, 8))
plt.plot(x, V(x), 'k', label="Potential V(x)")
plt.plot(x, np.abs(psi_1)**2, label=f'|ψ₁|² Ground State (E₁ = {E_1:.4f})', color='blue', alpha=0.7)
plt.plot(x, np.abs(psi_2)**2, label=f'|ψ₂|² First Excited State (E₂ = {E_2:.4f})', color='orange', alpha=0.7)
plt.plot(x, np.abs(psi_3)**2, label=f'|ψ₃|² Second Excited State (E₃ = {E_3:.4f})', color='green', alpha=0.7)

plt.title('First three wavefunctions |ψ_n|² using the imaginary-time propagation')
plt.xlabel('Position x')
plt.ylabel('Probability Density |ψ|²')
plt.legend()
plt.grid()
plt.show()

