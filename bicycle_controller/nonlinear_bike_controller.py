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
velocity = 5*np.cos(0.1*2*np.pi*time_vector)+5+5  # velocity of the bike

# Create the bike model
bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity[0])
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# ----------------------------------------------------------------------
# LQR controller design
Q = np.eye(bike.A.shape[0]) * 1
R = np.eye(bike.D.shape[1]) * 0.01
noise = 0

# Reference state
x_ref = np.zeros((1, 4))
u_ref = 0
# Create the controll
sys_d = bike.discrete_ss(dt)
K, S, E = controller_design.dlqr(sys_d.A, sys_d.B, Q, R)
sys_c = bike.continuous_ss()
print(K)

# Simulate the response
time = time_vector
sensor_time = [0]
states = [x0]
measurements = [x0]
control_input = [-K@measurements[-1]]
for idx, t in enumerate(time[1:]):
	bike.set_velocity(velocity[idx+1])
	states_dot = bike.A@states[-1] + np.dot(bike.B, control_input[-1])
	states.append(states[-1] + states_dot * (t-time[idx]))

	if (t-sensor_time[-1])>=dt:
		sys_d = bike.discrete_ss(dt)
		K, S, E = controller_design.dlqr(sys_d.A, sys_d.B, Q, R)

		measurements.append(states[-1] + np.random.randn(4,1)*noise)
		control_input.append(u_ref - K@(measurements[-1]))
		sensor_time.append(t)

# Convert output variables to arrays
states = np.asarray(states)
measurements = np.asarray(measurements)
control_input = np.squeeze(np.asarray(control_input))

# Plot the results
plt.figure()
plt.plot(time, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=2)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')

plt.figure()
plt.step(sensor_time, control_input, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
# plt.show()

plt.figure()
plt.step(time_vector, velocity, label=('Control input'), c='k')
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.show()
