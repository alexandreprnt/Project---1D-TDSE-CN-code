from week_43 import *

E_1_values = []
d_tau1=d_tau
t = np.linspace(0, 300, 400)
# Loop over each timestep and calculate E_1
for t in range(300):  # Nt-1 because we need t+dt for each step
    psi_t_dt = psi_1[t] # psi_1 at time t+dt
    norm_squared = np.vdot(psi_t_dt, psi_t_dt)  # Inner product: norm squared of psi_1(t+dt)
    E_1 = -np.log(norm_squared) / (2 * d_tau)  # Formula for E_1
    E_1_values.append(E_1)
    t= t + 0.02

# Save E_1 values to a text file
with open('E_1_values.txt', 'w') as f:
    for i, E_1 in enumerate(E_1_values):
        f.write(f"Timestep {i}, E_1 = {E_1}\n")

# Plot E_1 as a function of timestep
timesteps = np.arange(0, 300)
plt.plot(timesteps, E_1_values)
plt.xlabel('Time')
plt.ylabel('E_1')
plt.title('E_1 vs Time')
plt.grid(True)
plt.show()
