import numpy as np
import matplotlib.pyplot as plt
from TDSErealtime_plots import *

# We perform FFT to obtain High Harmonic Generation

def HHG(function):
    fhat = np.fft.fft(function)  # Compute the FFT
    freqs = np.fft.fftfreq(len(function), d=dt)  # frequencies
    omega = 2 * np.pi * freqs
    amplitude_spectrum = fhat
    pos_freqs = freqs > 0
    return omega[pos_freqs] / w_L, amplitude_spectrum[pos_freqs]

omega_X_exp, HHG_X_exp = HHG(X_exp)

HHG_V_exp = X_exp[-1] * np.exp(-1j * omega_X_exp * T) + 1j * omega_X_exp * HHG_X_exp
#omega_V_exp, HHG_V_exp = HHG(V_exp[:-1])
#omega_A_exp, HHG_A_exp = HHG(A_exp[:-2])
HHG_A_exp = V_exp[-2] * np.exp(-1j * omega_X_exp * T) + 1j * omega_X_exp * HHG_V_exp

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot FFT of <x(t)>
ax1.plot(omega_X_exp, np.abs(HHG_X_exp)**2, color='blue')
ax1.set_yscale('log')
ax1.set_xlabel('Harmonic Order')
ax1.set_xticks(range(39))
ax1.set_xlim(0,38)
ax1.set_ylabel('|X(omega)|^2 in log scale')
ax1.set_title('Harmonic Spectrum of <x(t)>')
ax1.grid(True)

# Plot FFT of <v(t)>
ax2.plot(omega_X_exp, np.abs(HHG_V_exp)**2, color='red')
ax2.set_yscale('log')
ax2.set_xlabel('Harmonic Order')
ax2.set_xticks(range(39))
ax2.set_xlim(0,38)
ax2.set_ylabel('|X(omega)|^2 in log scale')
ax2.set_title('Harmonic Spectrum of <v(t)>')
ax2.grid(True)

# Plot FFT of <a(t)>
ax3.plot(omega_X_exp, np.abs(HHG_A_exp)**2, color='green')
ax3.set_yscale('log')
ax3.set_xlabel('Harmonic Order')
ax3.set_xticks(range(39))
ax3.set_xlim(0,38)
ax3.set_ylabel('|X(omega)|^2 in log scale')
ax3.set_title('Harmonic Spectrum of <a(t)>')
ax3.grid(True)

# Display the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
