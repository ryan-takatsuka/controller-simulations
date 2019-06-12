import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

velocity = np.linspace(0,10,1000) # velocity range vector
real_eigs_list = [] # a matrix that holds the real part of the eigenvalues for each velocity
imag_eigs_list = [] # a matrix that holds the imaginary part of the eigenvalues for each velocity
P_rank_list = [] # a matrix that holds the rank of the controllabilty matrices for each velocity
Q_rank_list = [] # a matrix that holds the rank of the observabillity matrices for each velocity

bike =BikeModel(params.M, params.C_1, params.K_0, params.K_2)

# Calculate the state space matrices for each velocity in the velocity vector
for v in velocity:
	bike.set_velocity(v)

	P, P_rank = bike.calc_controllability()
	Q, Q_rank = bike.calc_observability()
	eigs = bike.calc_eigs()

	P_rank_list.append(P_rank)
	Q_rank_list.append(Q_rank)
	real_eigs_list.append(np.real(eigs))
	imag_eigs_list.append(np.imag(eigs))

real_eigs, imag_eigs = bike.sort_eigs(real_eigs_list, imag_eigs_list)
stable_index = bike.calc_stable_range(real_eigs)

# Plot stuff
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(velocity, real_eigs[:,0], label='eig1', )
plt.plot(velocity, real_eigs[:,1], label='eig2')
plt.plot(velocity, real_eigs[:,2], label='eig3')
plt.plot(velocity, real_eigs[:,3], '--', label='eig4')
plt.scatter(velocity[stable_index[0]], real_eigs[stable_index[0],3], 
				edgecolor='k', facecolors='none', zorder=5, s=70)
plt.scatter(velocity[stable_index[1]], real_eigs[stable_index[1],1], 
				edgecolor='k', facecolors='none', zorder=5, s=70)
plt.title('Real component')
# plt.xlabel('Velocity [m/s]')
plt.ylabel('Eigenvalue magnitude')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(velocity, imag_eigs[:,0], label='eig1')
plt.plot(velocity, imag_eigs[:,1], '--', label='eig2')
plt.plot(velocity, imag_eigs[:,2], label='eig3')
plt.plot(velocity, imag_eigs[:,3], label='eig4')
plt.title('Imaginary component')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Eigenvalue magnitude')
plt.grid()
plt.legend()
plt.show()

# plt.figure(2)
# plt.plot(velocity,P_rank_list, label='Controllability Matrix')
# plt.plot(velocity,Q_rank_list, label='Observability Matrix')
# plt.xlabel('Velocity [m/s]')
# plt.ylabel('Rank of the matrix (n=4)')
# plt.title('Controllability and Observability for varying velocities')
# plt.legend(shadow=True)

# plt.show()
