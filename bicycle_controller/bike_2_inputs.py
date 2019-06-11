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

velocity = np.array([0.5,3,7,11]) # velocity range vector
real_eigs_list = [] # a matrix that holds the real part of the eigenvalues for each velocity
imag_eigs_list = [] # a matrix that holds the imaginary part of the eigenvalues for each velocity
P_rank_list = [] # a matrix that holds the rank of the controllabilty matrices for each velocity
Q_rank_list = [] # a matrix that holds the rank of the observabillity matrices for each velocity
time_vector = np.linspace(0, 30, 10000)

bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# Open loop time response with initial steer angle
u = np.zeros((time_vector.size, 2))
x0 = np.array([np.deg2rad(6), 0, 0, 0])
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
u = np.zeros((time_vector.size, 2))
Q = np.eye(bike.A.shape[0])*0
R = np.eye(bike.D.shape[1])
dt = 0.1

control_plot = []
state_plot = []
for index, v in enumerate(velocity):
	bike.set_velocity(v)
	# K, S, E = control.lqr(sys.A, sys.B, Q, R)
	# sys_cl = signal.StateSpace((sys.A - sys.B@K), sys.B, sys.C, sys.D).to_discrete(dt)
	# tout, y, x = signal.lsim(sys_cl, u, time_vector, X0=x0)

	# sys = bike.discrete_ss(dt)
	# K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
	# sys_cl = signal.StateSpace((sys.A - sys.B@K), sys.B, sys.C, sys.D, dt=dt)
	# tout, y, x = signal.dlsim(sys_cl, np.zeros(1000), x0=x0)

	sys = bike.discrete_ss(dt)
	K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
	print(sys)
	s = signal.StateSpace(sys.A, sys.B, sys.C, sys.D, dt=dt)
	time, states, control_input = controller_design.simulate_discrete(s.A, s.B, s.C, s.D, dt, K, np.zeros(100), x0=x0)
	control_plot.append((time, control_input, v))
	state_plot.append((time, states, v))


plt.figure()
for plot in state_plot:
	plt.step(plot[0], np.rad2deg(plot[1][:,0]), label=('Velocity: ' + str(plot[2]) + ' m/s'))
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')

plt.figure()
for plot in control_plot:
	plt.step(plot[0], plot[1], label=('Velocity: ' + str(plot[2]) + ' m/s'))
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Steering Torque [Nm]')



plt.show()