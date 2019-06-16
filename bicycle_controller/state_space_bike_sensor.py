import numpy as np
from matplotlib import pyplot as plt
import bike_parameters as params
from BikeModel import BikeModel
import controller_design

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

# Parameters for the simulation
time_vector = np.linspace(0, 40, 10000)  # time vector for sim
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
# LQR controller design
Q = np.eye(bike.A.shape[0]) * 1
R = np.eye(bike.D.shape[1]) * 0.01

# Create the controll
sys = bike.discrete_ss(dt)
K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
# K = np.zeros((1,4))
sys = bike.continuous_ss()
print(K)
# Simulate the response
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0, noise=1e-2)

# Plot the results
plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=2)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')
plt.title('Roll Angle Sensor')

plt.figure()
plt.plot(t, np.rad2deg(states[:,1]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,1]), label=('Measured states'), s=2)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Steer angle [deg]')
plt.title('Steer Angle Sensor')

plt.figure()
plt.step(sensor_time, u, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.title('Input with noisy sensors')
plt.show()
