import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel
import control
from scipy import signal
import controller_design
from sensor import Sensor

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

# Parameters for the simulation
time_vector = np.linspace(0, 10, 10000)  # time vector for sim
x0 = np.array([[np.deg2rad(6), 0, 0, 0]]).T  # initial states
dt = 1/60  # sampling time for the sensor
velocity = 10  # velocity of the bike
r_var = 1e-2
q_var = 1

# Create the bike model
bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# Create the sensors
roll_angle_sensor = Sensor(r_var)
steer_angle_sensor = Sensor(r_var)
roll_gyro_sensor = Sensor(r_var)
steer_gyro_sensor = Sensor(r_var)
sensors = [roll_angle_sensor, steer_angle_sensor, roll_gyro_sensor, steer_gyro_sensor]


# ----------------------------------------------------------------------
# LQR controller design
Q = np.eye(bike.A.shape[0]) * 1
R = np.eye(bike.D.shape[1]) * 0.01

# Design the process noise matrix
Q_k = bike.calc_process_noise(q_var, dt)

# Create the controller
sys = bike.discrete_ss(dt)
K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
sys = bike.continuous_ss()

# Simulate the response
t, states, est_states, sensor_time, measurements, u = controller_design.simulate_kalman(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, r_var, q_var, x0=x0, sensors=sensors, Q=Q_k)

# Plot the results
titles = ['Roll Angle', 'Steer Angle', 'Roll Angular Velocity', 'Steer Angular Velocity']
ylabels = ['Angle [deg]', 'Angle [deg]', 'Angular Velocity [deg/sec]', 'Angular Velocity [deg/sec]']
plt.figure()
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.plot(t, np.rad2deg(states[:,i]), label=('Real states'), c='k')
	plt.scatter(sensor_time, np.rad2deg(measurements[:,i]), label=('Measured states'), s=2)
	plt.step(sensor_time, np.rad2deg(est_states[:,i]), label=('Estimated states'), c='r')
	if i==3:
		plt.legend()
	plt.grid()
	plt.ylabel(ylabels[i])
	if i==2 or i==3:
		plt.xlabel('Time [sec]')

	# plt.ylabel('Roll angle [deg]')
	plt.title(titles[i])


plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(sensor_time, np.rad2deg(measurements[:,0]), label=('Measured states'), s=2)
plt.step(sensor_time, np.rad2deg(est_states[:,0]), label=('Estimated states'), c='r')
plt.xlabel('Time [sec]')
plt.ylabel('Angle [deg]')
plt.title('LQR Controller with Kalman Filter')
plt.legend()
plt.grid()


plt.figure()
plt.step(sensor_time, u, label=('Control input'), c='k')
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Torque [Nm]')
plt.title('Control Input')
plt.show()
