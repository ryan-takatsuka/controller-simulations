import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel
import control
from scipy import signal
import controller_design

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

# Parameters for the simulation
time_vector = np.linspace(0, 10, 10000)  # time vector for sim
x0 = np.array([[np.deg2rad(6), 0, 0, 0]]).T  # initial states
dt = 1/60  # sampling time for the sensor
velocity = 10  # velocity of the bike

# Create the bike model
bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# ----------------------------------------------------------------------
# Pole placement controller design
# ----------------------------------------------------------------------
# Define controller parameters
overshoot = 5  # [%]
natural_freq = 1  # [Hz]

# Create controller
sys = bike.continuous_ss()
poles = controller_design.calc_poles(overshoot, natural_freq, 4)
K = signal.place_poles(sys.A, sys.B, poles)
K = K.gain_matrix
print(K)

# Simulate the response
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0, noise=1e-3)

# Plot the results
plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=10)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')
plt.title('Pole Placement: States')

plt.figure()
plt.step(sensor_time, u, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.title('Pole Placement: Control input')
# plt.show()


# ----------------------------------------------------------------------
# LQR controller design
Q = np.eye(bike.A.shape[0]) * 1
R = np.eye(bike.D.shape[1]) * 0.0000001

# Create the controll
sys = bike.discrete_ss(dt)
K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
sys = bike.continuous_ss()
print(K)
# Simulate the response
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0, noise=1e-3)

# Plot the results
plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=10)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')

plt.figure()
plt.step(sensor_time, u, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.show()
