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
time_vector = np.linspace(0, 20, 10000)

bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
print(bike.A)
print(bike.B)
print(bike.C)
print(bike.D)

# ----------------------------------------------------------------------
# Closed loop time response with initial steer angle
# LQR Control
u = np.zeros((time_vector.size, 1))
Q = np.eye(bike.A.shape[0])*0.001
R = np.eye(bike.D.shape[1])
x0 = np.array([[np.deg2rad(6), 0, 0, 0]]).T
dt = 1/60
velocity = 10
state_plot = []
measurements_plot = []
bike.set_velocity(velocity)

sys = bike.continuous_ss()
poles = controller_design.calc_poles(1, 0.1, 4)
K = signal.place_poles(sys.A, sys.B, poles)
K = K.gain_matrix
# K, S, E = controller_design.lqr(sys.A, sys.B, Q, R)
t, states, sensor_time, measurements, u = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0, noise=0)
print(len(sensor_time))
print(K)

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