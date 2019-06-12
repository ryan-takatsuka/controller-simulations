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

real_eigs, imag_eigs, cross_index = bike.sort_eigs(real_eigs_list, imag_eigs_list)
stable_index = bike.calc_stable_range(real_eigs)

for i in cross_index:
	print(velocity[i[0]])


real_eigs_array = np.asarray(real_eigs_list)
eig1_real = [real_eigs_array[0,0]]
eig2_real = [real_eigs_array[0,1]]
eig3_real = [real_eigs_array[0,2]]
eig4_real = [real_eigs_array[0,3]]

eigs_guess = np.zeros((4,4))

diff_eigs = np.zeros((4,4))
for eigs in real_eigs_array[1:,:]:
	eigs_sort = np.sort(eigs)

	eigs_guess[0,:] = (eigs-eig1_real[-1]) + eigs
	eigs_guess[1,:] = (eigs-eig2_real[-1]) + eigs
	eigs_guess[2,:] = (eigs-eig3_real[-1]) + eigs
	eigs_guess[3,:] = (eigs-eig4_real[-1]) + eigs

	error = (eigs - eig1_real[-1])**2
	eig_prev_index = np.argmin(error)
	eig1_real.append(eigs[eig_prev_index])

	error = (eigs - eig2_real[-1])**2
	error_guess = (eigs_guess - eig2_real[-1])**2
	eig_guess_index = np.mod(np.argmin(error_guess), 4)
	eig_prev_index = np.argmin(error)
	eig_index = eig_prev_index
	error_sort = np.sort(error)
	if np.abs(error_sort[0]-error_sort[1]) < 0.01:
		eig_index = eig_guess_index		
	eig2_real.append(eigs[eig_index])

	error = (eigs - eig3_real[-1])**2
	error_guess = (eigs_guess - eig3_real[-1])**2
	eig_guess_index = np.mod(np.argmin(error_guess), 4)
	eig_prev_index = np.argmin(error)
	if np.min(np.diff(eigs_sort)) < 0.0001:
		eig = eigs_sort[np.argmin(np.diff(eigs_sort))]		
		eig3_real.append(eig)
	else:
		eig3_real.append(eigs[eig_prev_index])

	error = (eigs - eig4_real[-1])**2
	eig_prev_index = np.argmin(error)
	error_sort = np.sort(error)
	if np.min(np.diff(eigs_sort)) < 0.0001:
		eig = eigs_sort[np.argmin(np.diff(eigs_sort))]		
		eig4_real.append(eig)
	else:
		eig4_real.append(eigs[eig_prev_index])

new_real_eigs = [eig1_real, eig2_real, eig3_real, eig4_real]
print(new_real_eigs)

plt.figure()
plt.plot((eig1_real), label='1')
plt.plot((eig2_real), label='2')
plt.plot((eig3_real), label='3')
plt.plot((eig4_real), label='4')
plt.legend()
# plt.show()
# plt.figure()
# plt.plot(diff_eigs_array)
plt.show()

plt.figure(2)
plt.scatter(velocity, real_eigs_array[:,0], s=1)
plt.scatter(velocity, real_eigs_array[:,1], s=1)
plt.scatter(velocity, real_eigs_array[:,2], s=1)
plt.scatter(velocity, real_eigs_array[:,3], s=1)


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
# plt.show()

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






# plt.figure(2)
# plt.plot(velocity,P_rank_list, label='Controllability Matrix')
# plt.plot(velocity,Q_rank_list, label='Observability Matrix')
# plt.xlabel('Velocity [m/s]')
# plt.ylabel('Rank of the matrix (n=4)')
# plt.title('Controllability and Observability for varying velocities')
# plt.legend(shadow=True)

# plt.show()
