import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import bike_parameters as params
from BikeModel import BikeModel

# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

velocity = np.linspace(0,10,10000) # velocity range vector
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


# for index, velocity in enumerate(v):
# 	M_inv = np.linalg.inv(M) # inverse of the mass matrix
# 	K = (g*K_0+velocity**2*K_2) # overall stiffness matrix
# 	A = np.concatenate((-velocity*np.dot(M_inv,C_1), -1*np.dot(M_inv,K)), axis=1)
# 	A = np.concatenate((A,np.concatenate((I_2,zero_2), axis=1))) # A matrix

# 	B = np.concatenate((M_inv,zero_2)) # B matrix
# 	C = np.array([0,0,1,0]) # C matrix
# 	D = np.array([0,0]) # D matrix 

# 	eigs = np.linalg.eigvals(A) # calculate eigen values of the system
# 	total_real_eigs.append(np.real(eigs)) # real parts of the eigenvalues
# 	total_imag_eigs.append(np.imag(eigs)) # imaginary parts of the eigenvalues

# 	A_2 = np.dot(A,A) # A^2
# 	A_3 = np.dot(A,A_2) # A^3
# 	P = np.concatenate((B, np.dot(A,B), np.dot(A_2,B), np.dot(A_3,B)),axis=1) # controllability matrix
# 	P_rank_list.append(np.linalg.matrix_rank(P)) # rank of the controllability matrix
# 	Q = np.stack((C, np.dot(C,A), np.dot(C,A_2), np.dot(C,A_3))) # observabillity matrix
# 	Q_rank_list.append(np.linalg.matrix_rank(Q)) # rank of the observability matrix


real_eigs, imag_eigs, cross_index = bike.sort_eigs(real_eigs_list, imag_eigs_list)
stable_index = bike.calc_stable_range(real_eigs)

print(stable_index)

# Plot stuff
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(velocity, real_eigs[:,0], label='eig1', )
plt.plot(velocity, real_eigs[:,1], '--', label='eig2')
plt.plot(velocity, real_eigs[:,2], label='eig3')
plt.plot(velocity, real_eigs[:,3], label='eig4')
plt.scatter(velocity[stable_index[0]], real_eigs[stable_index[0],0], 
				edgecolor='k', facecolors='none', zorder=5, s=70)
plt.scatter(velocity[stable_index[1]], real_eigs[stable_index[1],2], 
				edgecolor='k', facecolors='none', zorder=5, s=70)
plt.title('Real component')
# plt.xlabel('Velocity [m/s]')
plt.ylabel('Eigenvalue magnitude')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(velocity, imag_eigs[:,0], label='eig1')
plt.plot(velocity, imag_eigs[:,1], label='eig2')
plt.plot(velocity, imag_eigs[:,2], label='eig3')
plt.plot(velocity, imag_eigs[:,3], '--',  label='eig4')
plt.title('Imaginary component')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Eigenvalue magnitude')
plt.grid()
plt.legend()
plt.show()

# plt.figure(1)
# plt.plot(velocity,imag_eigs_list, markersize=0.9, label='Imaginary part of the eigenvalues')
# plt.scatter(velocity,real_eigs[:,0], s=1, label='Real part of the eigenvalues')
# plt.xlabel('Velocity [m/s]')
# plt.ylabel('Eigenvalues')
# plt.title('Variation of system eigenvalues with forward bike velocity')
# plt.grid(True)
# red_patch = patches.Patch(color='red', label='Real part of the eigenvalues')
# green_patch = patches.Patch(color='green', label='Imaginary part of the eigenvalues')
# plt.legend(handles=[red_patch, green_patch], shadow=True)






plt.figure(2)
plt.plot(velocity,P_rank_list, label='Controllability Matrix')
plt.plot(velocity,Q_rank_list, label='Observability Matrix')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Rank of the matrix (n=4)')
plt.title('Controllability and Observability for varying velocities')
plt.legend(shadow=True)

plt.show()
