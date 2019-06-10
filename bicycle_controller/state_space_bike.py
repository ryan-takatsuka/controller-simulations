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

u = np.zeros(time_vector.size)
x0 = np.array([0, 0, 0, np.deg2rad(-20)])
print(u)
plt.figure()
for index, v in enumerate(velocity):
	bike.set_velocity(v)
	# T, yout = bike.impulse_response(time_vector, 0)
	T, xout = bike.initial_response(time_vector, u, x0=x0)

	plt.plot(T,np.rad2deg(xout[:,2]), label=('Velocity: ' + str(v) + ' m/s'))


plt.ylim([-10, 90])
plt.legend()
plt.grid()
plt.xlabel('Test')
plt.show()