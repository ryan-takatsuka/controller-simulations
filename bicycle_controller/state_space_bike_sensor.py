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
time_vector = np.linspace(0, 20, 100000)

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
dt = 0.1
velocity = 9
state_plot = []
measurements_plot = []
bike.set_velocity(velocity)

sys = bike.discrete_ss(dt)
K, S, E = controller_design.dlqr(sys.A, sys.B, Q, R)
t, states, measurements = controller_design.simulate(sys.A, sys.B, sys.C, sys.D, K, dt, time_vector, x0=x0)

plt.figure()
plt.plot(t, np.rad2deg(states[:,0]), label=('Real states'), c='k')
plt.scatter(t, np.rad2deg(measurements[:,0]), label=('Measured states'), s=10)
plt.legend()
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Roll angle [deg]')

plt.show()