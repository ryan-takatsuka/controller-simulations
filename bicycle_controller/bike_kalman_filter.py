'''
This script analyzes the bike model with a reduced number of sensors.  
This assumes that not all the state are being measured, and some
need to be estimated.  It also assumes that there is gaussian noise in
the sensor readings that needs to be filtered out.

For this simulation, it is assumed that there is a steer rate and roll
rate sensor (gyroscopes) because those states are cheap and easy to measure.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel
import control
from scipy import signal
import controller_design
from sensor import Sensor
from filterpy import common

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

# Parameters for the simulation
time_vector = np.linspace(0, 10, 10000)  # time vector for sim
x0 = np.array([[np.deg2rad(6), 0, 0, 0]]).T  # initial states
dt = 1/60  # sampling time for the sensor
velocity = 10  # velocity of the bike
r_var = 1e-2
q_var = 1

# Create the sensors
roll_gyro_sensor = Sensor(r_var)
steer_gyro_sensor = Sensor(r_var)
sensors = [roll_gyro_sensor, steer_gyro_sensor]

# Create the bike model
# C = np.array([[1, 0, 0, 0],
# 			  [0, 1, 0, 0]])
C = np.array([[0, 0, 1, 0],
			  [0, 0, 0, 1]])
D = np.zeros(2)
bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
bike.update_C(C)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# ----------------------------------------------------------------------
# Open Loop Kalman filter
# ----------------------------------------------------------------------
# Create controller
Q_c = np.eye(4)
# Q_c[0,0] *= 100
R_c = np.eye(1) * 0.001
sys = bike.discrete_ss(dt)
# K, S, E = controller_design.dlqr(sys.A, sys.B, Q_c, R_c)
K = np.zeros((1,4))
sys = bike.continuous_ss()
print(K)

# Design the process noise matrix
Q_k = bike.calc_process_noise(q_var, dt)

# Simulate the response
t, states, est_states, sensor_time, measurements, u = controller_design.simulate_kalman(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, r_var, q_var, x0=x0, sensors=sensors, Q=Q_k)

# Plot the results
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states1'), c='k')
plt.plot(t, np.rad2deg(states[:,1]), label=('Real states2'), c='k')
plt.plot(t, np.rad2deg(states[:,2]), label=('Real states3'), c='k')
plt.plot(t, np.rad2deg(states[:,3]), label=('Real states4'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=2, c='r')
plt.scatter(sensor_time, np.rad2deg(measurements[:,1]), label=('Measured states'), s=2, c='r')
plt.plot(sensor_time, np.rad2deg(est_states[:,0]), '--', label=('Measured states1'), c=colors[0], drawstyle='steps')
plt.plot(sensor_time, np.rad2deg(est_states[:,1]), '--', label=('Measured states2'), c=colors[1], drawstyle='steps')
plt.plot(sensor_time, np.rad2deg(est_states[:,2]), '--', label=('Measured states3'), c=colors[2], drawstyle='steps')
plt.plot(sensor_time, np.rad2deg(est_states[:,3]), '--', label=('Measured states4'), c=colors[4], drawstyle='steps')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('State variable value [deg, deg/s]')
plt.title('LQR Control: States')

plt.figure()
plt.step(sensor_time, u, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.title('LQR Control: Control input')
plt.show(0)

# Roll plots
plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label='Real Roll Angle', c='k')
# plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label='Measured Roll Angle', c='r', s=2)
plt.plot(sensor_time, np.rad2deg(est_states[:,0]), '--', label='Estimated Roll Angle', drawstyle='steps')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')
plt.title('LQR Control: States')
plt.show()