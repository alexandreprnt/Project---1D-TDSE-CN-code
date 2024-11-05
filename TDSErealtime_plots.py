import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
from scipy.sparse.linalg import splu
from TDSEimag import *
# Parameters and constants
a_0 = 1  # Bohr radius in meters
L = 20 * a_0
x_min = -10 * a_0
x_max = 10 * a_0
dx = 0.06 * a_0
dt = 0.02
N = int(L / dx)
Nt = int(1 / dt)
x = np.linspace(x_min, x_max, N)
hbar = 1  # Reduced Planck's constant
m_e = 1  # Electron mass


# Define potential function
def V(x):
    a = 0.816
    return -1 / np.sqrt(x ** 2 + a ** 2)


# Create potential matrix
V_matrix = np.diag(V(x))


# Spatial discretization matrix (S_matrix)
def spatialdiscr_matrix(N, dx):
    diagonal = -2 * np.ones(N)
    off_diagonal = np.ones(N - 1)
    matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    return (-1 / (2 * dx ** 2)) * matrix


S_matrix = spatialdiscr_matrix(N, dx)

# Electric field and interaction term
w_L = 0.057
N_c = 10
T = N_c * 2 * np.pi / w_L
I = 5e13 / 3.51e16
E0 = np.sqrt(I)


def f_t(t):
    return np.sin(np.pi * t / T) ** 2


def E_t(t):
    return E0 * f_t(t) * np.sin(w_L * t)


# Time-dependent Hamiltonian matrix
def H_dis(t):
    X = np.diag(x)
    interaction_term = X * E_t(t)
    return S_matrix + V_matrix + interaction_term

# Modified Crank-Nicolson function
import numpy as np

def CrankNicolson(psi0, x, time_array, H_dis, print_norm=False):
    """
    Crank-Nicolson method for time-dependent Hamiltonian in 1D Schrodinger equation,
    using a two-step approach with H_current and H_next.

    Parameters:
        psi0 : ndarray
            Initial wave function.
        x : ndarray
            Spatial grid points.
        time_array : ndarray
            Array of time points at which to evaluate the Hamiltonian.
        H_dis : function
            Function to compute the Hamiltonian matrix at a given time.
        hbar : float, optional
            Reduced Planck constant (default is 1).
        print_norm : bool, optional
            Print the norm of the wave function at each step if True.

    Returns:
        np.ndarray: Real part of the wave function at all time steps.
    """
    J = x.size - 1
    Nt = len(time_array)  # Number of time steps based on time_array length
    PSI_t = np.zeros((Nt, J + 1), dtype=complex)  # Store all wavefunctions
    PSI_t[0, :] = psi0  # Initial wavefunction

    psi = psi0.copy()  # Initialize wave function

    for t in range(Nt - 1):
        # Calculate time step based on time_array
        dt = time_array[t + 1] - time_array[t]

        # Compute Hamiltonians at current and next time steps
        H_current = H_dis(time_array[t])
        H_next = H_dis(time_array[t + 1])

        # Construct matrices A and B based on current and next Hamiltonians
        A = np.eye(J + 1) - (1j * dt / (2 * hbar)) * H_current
        B = np.eye(J + 1) + (1j * dt / (2 * hbar)) * H_next

        # Solve A * psi_{n+1} = B * psi_{n} for psi_{n+1}
        psi = np.linalg.solve(A, B @ psi)
        PSI_t[t + 1, :] = psi

        if print_norm:
            norm = np.trapezoid(np.abs(psi) ** 2, x)
            print(f"Step {t}, Norm: {norm}")

    return np.real(PSI_t)

time_array = np.arange(0, T, 100*dt)  # Include T if dt fits perfectly

psi0 = psi_1 #from the previous file
# Run Crank-Nicolson simulation
PSI_t = CrankNicolson(psi0, x, time_array, H_dis, print_norm=True)



# Modified X_EXP function
def X_EXP(PSI_t, x):
    x_reshaped = x[np.newaxis, :]
    x_exp = dx * np.sum(np.conj(PSI_t) * x_reshaped * PSI_t, axis=1)
    return x_exp

X_exp = X_EXP(PSI_t,x)
optical_cycles= time_array / (2 * np.pi / w_L)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot E(t) in the first subplot
ax1.plot(optical_cycles, E_t(time_array), label='E(t)', color='blue')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('E(t)')
ax1.set_title("Electric Field E(t)")
ax1.grid(True)
ax1.legend()

# Plot <x(t)> in the second subplot
ax2.plot(optical_cycles, X_exp, label=r'<x(t)>', color='blue')
ax2.set_xlabel('Time in optical cycles')
ax2.set_ylabel('<x(t)>')
ax2.set_title("Expectation Value of Position Over Time")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

omega = np.linspace(0, 100, 1000)  # Adjust frequency range as needed

# Initialize the Fourier Transform result
X_omega = np.zeros_like(omega, dtype=complex)

# Compute the Fourier Transform using numerical integration (trapezoidal rule)
for i, w in enumerate(omega):
    # Integral approximation using the trapezoidal rule
    X_omega[i] = np.trapezoid(X_exp * np.exp(-1j * w * time_array), dx=dt)

# Calculate the magnitude of the Fourier Transform
X_magnitude = np.abs(X_omega)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot FFT of <x(t)> in the first subplot
ax1.plot(omega, X_magnitude**2)
ax1.set_yscale('log')
ax1.set_xlabel('Frequency omega (rad/s)')
ax1.set_ylabel('|X(omega)|^2')
ax1.set_title("Fourier Transform of Expected Position <x(t)>")
ax1.set_xlim(0, 10)
ax1.grid(True)
ax1.legend()

# Plot <x(t)> in the second subplot
ax2.plot(optical_cycles, X_exp, label=r'<x(t)>', color='blue')
ax2.set_xlabel('Time in optical cycles')
ax2.set_ylabel('<x(t)>')
ax2.set_title("Expectation Value of Position Over Time")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()


def V_EXP(PSI_t, x, dx, dt):
    x_exp_series = X_exp
    v_exp = np.diff(x_exp_series) / dt
    v_exp = np.append(v_exp, np.nan)
    return v_exp

V_exp = V_EXP(PSI_t, x, dx, dt)

def A_EXP(PSI_t, x, dx, dt):
    x_exp_series = X_exp
    v_exp = np.diff(x_exp_series) / dt
    a_exp = np.diff(v_exp) / dt  # Calculate finite difference for <v(t)>
    a_exp = np.append(a_exp, [np.nan, np.nan])
    return a_exp

A_exp = A_EXP(PSI_t, x, dx, dt)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot <x(t)>
ax1.plot(optical_cycles, X_exp, label='<x(t)>', color='blue')
ax1.set_xlabel('Time in optical cycles')
ax1.set_ylabel('<x(t)>')
ax1.set_title("Position Expectation Value <x(t)>")
ax1.grid(True)
ax1.legend()

# Plot <v(t)>
ax2.plot(optical_cycles, V_exp, label='<v(t)>', color='red')
ax2.set_xlabel('Time in optical cycles')
ax2.set_ylabel('<v(t)>')
ax2.set_title("Velocity Expectation Value <v(t)>")
ax2.grid(True)
ax2.legend()

# Plot <a(t)>
ax3.plot(optical_cycles, A_exp, label='<a(t)>', color='green')
ax3.set_xlabel('Time in optical cycles')
ax3.set_ylabel('<a(t)>')
ax3.set_title("Acceleration Expectation Value <a(t)>")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot FFT of <x(t)> in the first subplot
ax1.plot(omega, X_magnitude**2, color='blue')
ax1.set_yscale('log')
ax1.set_xlabel('Frequency omega (rad/s)')
ax1.set_ylabel('|X(omega)|^2 in log scale')
ax1.set_title("Fourier Transform of Expected Position <x(t)> in log scale")
ax1.set_xlim(0, 10)
ax1.grid(True)
ax1.legend()

# Plot <x(t)> in the second subplot
ax2.plot(omega, X_magnitude**2, color='red')
ax2.set_yscale('linear')
ax2.set_xlabel('Frequency omega (rad/s)')
ax2.set_ylabel('|X(omega)|^2 in linear scale')
ax2.set_title("Fourier Transform of Expected Position <x(t)> in linear scale")
ax2.set_xlim(0, 10)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()