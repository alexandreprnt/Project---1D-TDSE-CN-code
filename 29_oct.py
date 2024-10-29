import numpy as np
import matplotlib.pyplot as plt

# Constants
a_0 = 1  # Bohr radius in meters
L = 20 * a_0  # Length of the spatial domain
x_min = -10 * a_0
x_max = 10 * a_0
dx = 0.06 * a_0  # Spatial step size
dt = 0.02  # Time step size
N = int(L / dx)  # Number of spatial points
Nt = int(1 / dt)  # Number of time steps
x = np.linspace(x_min, x_max, N)  # Spatial grid
hbar = 1  # Reduced Planck's constant
m_e = 1  # Electron mass

# Initialize lists to store results
computed_states = []  # To store wavefunctions at each time step
x_expectation_values = []  # To store expectation values of x at each time step

# Potential function V(x)
def V(x):
    a = 0.816
    return -1 / np.sqrt(x ** 2 + a ** 2)

# Diagonal potential matrix
V_matrix = np.diag(V(x))

# Define the spatial discretization (second-derivative operator) matrix
def spatial_discr_matrix(N, dx):
    diagonal = -2 * np.ones(N)
    off_diagonal = np.ones(N - 1)
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return (-1 / (2 * dx ** 2)) * matrix

# Define the Hamiltonian
S_matrix = spatial_discr_matrix(N, dx)
H = S_matrix + V_matrix  # Hamiltonian matrix

# Crank-Nicolson matrices
def CN_matrices(H, dt):
    I = np.eye(N, dtype=complex)
    A = I + 1j * (dt / 2) * H
    B = I - 1j * (dt / 2) * H
    return A, B

A, B = CN_matrices(H, dt)

# Function to normalize the wavefunction
def normalize(psi, dx):
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm

# Function to initialize the Gaussian wave packet
def initial_wavefunction(x, dx):
    psi = np.exp(-x ** 2)  # Gaussian shape
    return normalize(psi, dx)

# Calculate expectation value of x for a given wavefunction
def x_expectation(psi, x, dx):
    return dx * np.sum(np.conj(psi) * x * psi).real

# Time evolution function
def time_evolution(psi, A, B, Nt, dx):
    for _ in range(Nt):
        psi = np.linalg.solve(A, np.dot(B, psi))  # Crank-Nicolson step
        psi = normalize(psi, dx)  # Normalize at each time step
        computed_states.append(psi)  # Store wavefunction
        x_exp = x_expectation(psi, x, dx)  # Compute expectation value
        x_expectation_values.append(x_exp)  # Store expectation value

    return psi


# Initialize the wavefunction
psi = initial_wavefunction(x, dx)
computed_states.append(psi)  # Store the initial wavefunction
x_exp = x_expectation(psi, x, dx)  # Initial expectation value
x_expectation_values.append(x_exp)

# Perform time evolution
final_psi = time_evolution(psi, A, B, Nt, dx)

# Create a time array corresponding to the expectation values
time_array = np.arange(0, (Nt+1) * dt, dt)

# Plot the expectation values of x over time
plt.plot(time_array, x_expectation_values)
plt.xlabel('Time')
plt.ylabel(r'$\langle x \rangle$')
plt.title(r'Expected Position $\langle x(t) \rangle$')
plt.grid()
plt.show()

# Time array corresponding to expectation values
time_array = np.arange(0, len(x_expectation_values) * dt, dt)

# Define the range of frequencies for the transform
omega = np.linspace(-10, 10, 1000)  # Adjust frequency range as needed

# Initialize the Fourier Transform result
X_omega = np.zeros_like(omega, dtype=complex)

# Compute the Fourier Transform using numerical integration (trapezoidal rule)
for i, w in enumerate(omega):
    # Integral approximation using the trapezoidal rule
    X_omega[i] = np.trapezoid(x_expectation_values * np.exp(-1j * w * time_array), dx=dt)

# Calculate the magnitude of the Fourier Transform
X_magnitude = np.abs(X_omega)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(omega, X_magnitude)
plt.xlabel(r'Frequency $\omega$ (rad/s)')
plt.ylabel(r'$|X(\omega)|$')
plt.title(r'Fourier Transform of Expected Position $\langle x(t) \rangle$')
plt.grid()
plt.xlim(-10, 10)  # Adjust limits as needed
plt.show()