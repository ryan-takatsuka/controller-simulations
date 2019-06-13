import numpy as np
from matplotlib import pyplot as plt
import bike_parameters as params
from BikeModel import BikeModel
import control
from scipy import signal
import controller_design

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

velocity = 10
real_eigs_list = []  # a matrix that holds the real part of the eigenvalues for each velocity
imag_eigs_list = []  # a matrix that holds the imaginary part of the eigenvalues for each velocity
P_rank_list = []  # a matrix that holds the rank of the controllabilty matrices for each velocity
Q_rank_list = []  # a matrix that holds the rank of the observabillity matrices for each velocity
time_vector = np.linspace(0, 5, 10000)
x0 = np.array([[np.deg2rad(6), 0, 0, 0]])
dt = 0.1

bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# ----------------------------------------------------------------------
# Open Loop
sys = bike.continuous_ss()
K = np.zeros((1,4))
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0.T, noise=0)

plt.figure(1)
ylabels = ['Angle [deg]', 'Angle [deg]', 'Angular Rate [deg/s]', 'Angular Rate [deg/s]']
titles = ['Roll Angle', 'Steer Angle', 'Roll Rate', 'Steer Rate']
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.step(sensor_time, np.rad2deg(measurements[:,i]), label=('Open Loop'))
	plt.title(titles[i])
	if i==2 or i==3:
		plt.xlabel('Time [sec]')
	plt.ylabel(ylabels[i])
	plt.grid()

plt.figure(2)
plt.step(sensor_time, u, label='Open Loop')
plt.title('Control Input')
plt.xlabel('Time [sec]')
plt.ylabel('Steering Torque [N-m]')
plt.grid()

# ----------------------------------------------------------------------
# Closed loop time response with initial steer angle
# LQR Control
q_mag = [1, 1e-5, 1]
r_mag = [1, 1, 1e-5]

for i in range(len(q_mag)):

	Q = np.eye(bike.A.shape[0]) * q_mag[i]
	R = np.eye(bike.D.shape[1]) * r_mag[i]
	dt = 0.1

	sys = bike.discrete_ss(dt)
	K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
	sys = bike.continuous_ss()
	t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0.T, noise=0)

	plt.figure(1)
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.step(sensor_time, np.rad2deg(measurements[:,i]), label=('LQR, Q=' + str(np.max(Q)) + ', R=' + str(np.max(R))))

	plt.figure(2)
	plt.step(sensor_time, u, label=('LQR, Q=' + str(np.max(Q)) + ', R=' + str(np.max(R))))

# -----------------------------------------------------------
# Closed loop time response with initial steer angle
# pole placement
sys = bike.continuous_ss()
poles = controller_design.calc_poles(5, 1, 4)
K = signal.place_poles(sys.A, sys.B, poles)
K = K.gain_matrix
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0.T, noise=0)

plt.figure(1)
for i in range(4):
	plt.subplot(2,2,i+1)
	plt.step(sensor_time, np.rad2deg(measurements[:,i]), label=('Pole Placement'))
	if i==3:
		plt.legend()

plt.figure(2)
plt.step(sensor_time, u, label='Pole Placement')
plt.legend()

plt.show()
