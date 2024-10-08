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

# Normalization function
def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

# Time evolution using Crank-Nicolson method (Imaginary time)
#Nt : number of time step
def time_evolution(psi, A, B, Nt, dx):
    for _ in range(Nt):
        psi = la.solve(A, np.dot(B, psi))
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    return psi


def psi_0(N,x, dx):
    psi = N*np.exp(-x ** 2)  # for gaussian wavepacket
    psi = normalize(psi, dx)  # Normalize the initial wavefunction
    psi_0 = time_evolution(psi, A, B, Nt,dx) #Time evolution to find the psi_0
    return psi_0

psi_0=psi_0(N,x,dx)

#2nd method
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

#Now I want to compare with the first method
def compute_energy(psi, H):
    return np.real(np.dot(psi.conj(), np.dot(H, psi)) * dx)

psi_1 = time_evolution(psi_0,A,B,Nt,dx)
ground_state_energy = compute_energy(psi_1, H)
print(f"Ground state energy E1: {compute_energy(psi_1, H):.5f}")
# Define projection to make psi orthogonal to previously computed states
def project_out(psi, *states):
    for state in states:
        overlap = np.sum(psi * np.conj(state)) * dx
        psi -= overlap * state
    return psi

# Modify time evolution function to orthogonalize against psi_1 and psi_2
def time_evolution_orthogonalized(psi, A, B, Nt, dx, *orthogonal_to):
    for _ in range(Nt):
        psi = la.solve(A, np.dot(B, psi))
        psi = project_out(psi, *orthogonal_to)  # Project out previous states
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalize
    return psi

# Find psi_2: First excited state
def psi_2(x, dx, A, B, Nt, psi_1):
    # Start with an initial guess for psi_2 (e.g., sin(x) or another function with a node)
    psi = x * np.exp(-x ** 2)
    psi = normalize(psi, dx)
    psi_2 = time_evolution_orthogonalized(psi, A, B, Nt, dx, psi_1)
    return psi_2

# Find psi_3: Second excited state
def psi_3(x, dx, A, B, Nt, psi_1, psi_2):
    # Start with an initial guess for psi_3 (e.g., x^2 * exp(-x^2) or a function with two nodes)
    psi = (x**2 - 1) * np.exp(-x ** 2)
    psi = normalize(psi, dx)
    psi_3 = time_evolution_orthogonalized(psi, A, B, Nt, dx, psi_1, psi_2)
    return psi_3

# Compute psi_2 and psi_3 using imaginary time evolution
psi2_imag = psi_2(x, dx, A, B, Nt, psi_0)
psi3_imag = psi_3(x, dx, A, B, Nt, psi_0, psi2_imag)

# Compute energies for psi_2 and psi_3
E2_imag = np.real(np.dot(np.conj(psi2_imag), np.dot(H, psi2_imag)) * dx)
E3_imag = np.real(np.dot(np.conj(psi3_imag), np.dot(H, psi3_imag)) * dx)

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(x, V(x), 'k', label="Potential V(x)")
plt.plot(x, np.abs(psi_0)**2, label=f'Ground State |ψ₁(x)|² (E₁ = {E1:.4f})')
plt.plot(x, np.abs(psi2_imag)**2, label=f'1st Excited State |ψ₂(x)|² (E₂ = {E2_imag:.4f})')
plt.plot(x, np.abs(psi3_imag)**2, label=f'2nd Excited State |ψ₃(x)|² (E₃ = {E3_imag:.4f})')

# Plot formatting
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('First Three Eigenstates of the Hamiltonian by Imaginary Time Evolution')
plt.legend()
plt.grid(True)
plt.show()
