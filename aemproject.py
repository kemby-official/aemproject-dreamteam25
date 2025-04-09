import numpy as np
import matplotlib.pyplot as plt

# Constants
MU = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6  # Earth radius (m)
ALTITUDE_LEO = 500e3  # LEO altitude (m)

# Variables

initial_pos = [R_EARTH + ALTITUDE_LEO, 0]
initial_vel = [0, 11200]

def compute_acceleration(position):
    """
    Parameters:
        position: [x, y] in meters
    Returns:
        [a_x, a_y] in m/s^2, represented by numpy array
    """
    r_vec = np.array(position)
    r = np.linalg.norm(r_vec)
    acceleration = -MU / r ** 3 * r_vec
    return acceleration

def simulate_spacecraft(initial_pos, initial_vel, dt=1, t_total=10000):
    """
    Simulates for t_total seconds (1.5 hrs = ~1 LEO orbit).
    Parameters: initial_pos in meters, initial_vel in m/s, dt in seconds, t_total
    Return: trajectory
    """
    pos = np.array(initial_pos, dtype='float64')
    vel = np.array(initial_vel, dtype='float64')
    trajectory = [pos.copy()]
    t = 0
    distance = 10 # when distance <=6, it prints the escape message correctly, but when it's >6 it doesn't. UGH! 
    escaped = False

    while t <= t_total:
        r = np.linalg.norm(pos)

        # Check for crash
        if r <= R_EARTH:
            print('Spacecraft crashed!')
            break
        # Check for escape
        if r >= distance * R_EARTH and not escaped: # <- right here, officer!
            print(f'Spacecraft escaped at t = {t} s!')
            escaped = True

        # Compute acceleration
        acc = compute_acceleration(pos)

        # Euler integration
        vel += acc * dt
        pos += vel * dt

        trajectory.append(pos.copy())
        t += dt

    return np.array(trajectory)

def calculate_energy(trajectory, dt):
    """Computes kinetic, potential, and total energy, given a trajectory"""
    velocities = np.diff(trajectory, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)
    kinetic = 0.5 * speeds ** 2
    positions = trajectory[:-1]
    distances = np.linalg.norm(positions, axis=1)
    potential = -MU / distances
    total = kinetic + potential
    return kinetic, potential, total

# Simulate
trajectory = simulate_spacecraft(initial_pos, initial_vel, dt=10, t_total=6000)
kinetic, potential, total_energy = calculate_energy(trajectory, dt=10)

# Plot trajectory
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="LEO Orbit")
plt.gca().add_patch(plt.Circle((0, 0), R_EARTH, color='blue', alpha=0.2))
plt.title("LEO Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# Plot energy
plt.figure()
plt.plot(kinetic, label="Kinetic")
plt.plot(potential, label="Potential")
plt.plot(total_energy, label="Total")
plt.title("Energy Conservation in LEO")
plt.xlabel("Time Step")
plt.ylabel("Energy (J/kg)")
plt.legend()
plt.grid(True)
plt.show()
