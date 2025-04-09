import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Constants
MU = 3.986e14          # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6      # Earth radius (m)
ALTITUDE_LEO = 400e3   # LEO altitude (m)

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
    escaped = False

    while t <= t_total:
        r = np.linalg.norm(pos)

        # Check for crash
        if r <= R_EARTH:
            print('Spacecraft crashed!')
            break
        # Check for escape
        if r > 10 * R_EARTH and not escaped:
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
    
MU = 3.986e14          # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6      # Earth radius (m)
ALTITUDE_LEO = 400e3   # LEO altitude (m)
# Initial conditions for LEO
initial_pos = [R_EARTH + ALTITUDE_LEO, 0.0]
initial_vel = [0.0, 7670.0]  # m/s

# Simulate
trajectory = simulate_spacecraft(initial_pos, initial_vel, dt=10, t_total=6000)

# Plot

plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="LEO Orbit")
# plot your trajectory here 

plt.gca().add_patch(plt.Circle((0, 0), R_EARTH, color='blue', alpha=0.2))  # please includie this line to plot the earth
plt.title("LEO Trajectory")
plt.legend()
plt.grid(True)
plt.show()
