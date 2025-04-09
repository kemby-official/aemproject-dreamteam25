import numpy as np
import matplotlib.pyplot as plt
%matplotlib
inline

# Constants
MU = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6  # Earth radius (m)
ALTITUDE_LEO = 400e3  # LEO altitude (m)


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
    Parameterss: initial_pos in meters, initial_vel in m/s, dt in seconds, t_total
    Return: trajectory
    """
    pos = np.array(initial_pos, dtype='float64')
    vel = np.array(initial_vel, dtype='float64')
    trajectory = [pos.copy()]
    t = 0
    escaped = False

    while t <= t_total:
        r = np.linalg.norm(pos)

        # check for crash
        if r <= R_EARTH:
            print('Spacecraft crashed!')
            break
        # check for escape (arbitrarily, r > 10*R_EARTH)
        if r > 10 * R_EARTH and not escaped:
            print(f'Spacecraft escaped at t = {t} s!')
            escaped = True

        # compute acceleration
        acc = compute_acceleration(pos)

        # Euler integration: updates velocity and position
        vel += acc * dt
        pos += vel * dt

        # record trajectory
        trajectory.append(pos.copy())

        t += dt

    return np.array(trajectory)  # using numpy array to store the trajecotry


def calculate_energy(trajectory, dt):
    """Computes kinetic, potential, and total energy, given a trajectory"""
    velocities =  # Approximate velocity using difference, which can be computed using np.diff

    return kinetic, potential, total