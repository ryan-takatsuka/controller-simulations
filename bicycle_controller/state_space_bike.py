import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

velocity = np.array([0.5,3,7,10]) # velocity range vector
real_eigs_list = [] # a matrix that holds the real part of the eigenvalues for each velocity
imag_eigs_list = [] # a matrix that holds the imaginary part of the eigenvalues for each velocity
P_rank_list = [] # a matrix that holds the rank of the controllabilty matrices for each velocity
Q_rank_list = [] # a matrix that holds the rank of the observabillity matrices for each velocity
time_vector = np.linspace(0, 10, 10000)

bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# Open loop time response with initial steer angle
u = np.zeros((time_vector.size, 1))
x0 = np.array([0, np.deg2rad(10), 0, 0])
plt.figure()
for index, v in enumerate(velocity):
	bike.set_velocity(v)
	T_c, xout_c = bike.continuous_response(time_vector, u, x0=x0)
	T_d, xout_d = bike.discrete_response(time_vector, u, x0=x0, dt=0.1)
	plt.plot(T_c,np.rad2deg(xout_c[:,0]), label=('Velocity: ' + str(v) + ' m/s'))
plt.ylim([-10, 90])
plt.legend()
plt.grid()
plt.xlabel('Test')


# Closed loop time response with initial steer angle
# Q = np.eye(bike.A.shape[0])
# R = np.eye(bike.D.size)
# plt.figure()
# for index, v in enumerate(velocity):
# 	bike.set_velocity(v)
# 	system, K = bike.lqr_controller(Q, R, dt=0.1)
# 	# T_c, xout_c = bike.continuous_response(time_vector, u, x0=x0, system=system)
# 	T_d, xout_d = bike.discrete_response(time_vector, u, x0=x0, dt=0.1, system=system)
# 	plt.step(T_d,np.rad2deg(xout_d[:,2]), label=('Velocity: ' + str(v) + ' m/s'))
# plt.ylim([-10, 90])
# plt.legend()
# plt.grid()
# plt.xlabel('Test')


# plt.show()

from controller_design import simulate
Q = np.eye(bike.A.shape[0])
R = np.eye(bike.D.shape[1])
plt.figure()
velocity = velocity[1:2]
for index, v in enumerate(velocity):
	bike.set_velocity(v)
	system, K = bike.lqr_controller(Q, R, dt=0.1)
	t, x, u1 = simulate(bike.A, bike.B, bike.C, bike.D, K, u, time_vector, x0=x0, dt=0.1)
	# T_c, xout_c = bike.continuous_response(time_vector, u, x0=x0, system=system)
	t_, x_ = bike.discrete_response(time_vector, u, x0=x0, dt=0.1, system=system)
	plt.step(t,np.rad2deg(x[:,0]), label=('Velocity: ' + str(v) + ' m/s'))
	plt.step(t_,np.rad2deg(x_[:,0]), label=('Velocity: ' + str(v) + ' m/s'))

plt.ylim([-10, 90])
plt.legend()
plt.grid()
plt.xlabel('Test')
plt.show()


# print((bike.B @ u.T).T)