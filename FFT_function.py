import numpy as np
import matplotlib.pyplot as plt

# We create a function sin(1/2 t) and plot the fast fourier transform of it
T = 2 * np.pi #Period
t = np.linspace(0, 10 * T, 500)

f_t= np.sin(t/2)

#We perform the FFT
fhat = np.fft.fft(f_t)  # Compute the FFT
freqs = np.fft.fftfreq(len(f_t), d=(t[1] - t[0]))# frequencies
omega = 2 * np.pi * freqs
amplitude_spectrum = np.abs(fhat) #the magnitude of the FFT function is giving the amplitude spectrum

#We plot the function and its amplitude spectrum
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, f_t, label=r'$f(t) = \sin(t/2)$')
plt.title("Time Domain Signal")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(omega, amplitude_spectrum, 'b', markerfmt=" ", basefmt="-b")
plt.title("Frequency Domain (FFT)")
plt.xlabel("Angular Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(-1, 1)

plt.tight_layout()
plt.show()