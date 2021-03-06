import numpy as np
import control
from scipy import signal
from filterpy import common

class BikeModel:
	def __init__(self, M, C1, K0, K2, v=1.0):
		''' Create the bikie model using the matrix parameters. '''

		# Set the input parameters
		self.M = np.array(M)
		self.C1 = np.array(C1)
		self.K0 = np.array(K0)
		self.K2 = np.array(K2)
		self.v = v
		self.g = 9.81 # gravity

		# Create the state space model
		self.calc_state_space_vars()
		self.continuous_ss()

	def calc_state_space_vars(self):
		''' Calculate the state space variables '''

		zero_mat = np.zeros((2,2))
		I_mat = np.eye(2)

		M_inv = np.linalg.inv(self.M) # inverse of the mass matrix
		K = (self.g*self.K0+self.v**2*self.K2) # overall stiffness matrix
		self.A = np.concatenate((-1*(M_inv@K), -self.v*(M_inv@self.C1)), axis=1)
		self.A = np.concatenate((np.concatenate((zero_mat, I_mat),axis=1),self.A), axis=0) # A matrix		
		self.B = np.concatenate((np.zeros((2,2)),M_inv)) # B matrix
		# self.C = np.array([[0,0,1,0],
		# 				   [0,0,0,1]]) # C matrix
		self.C = np.eye(4)
		self.D = np.zeros((4,2)) # D matrix

		# Ignore roll torque input
		self.B = np.array([self.B[:,1]]).T
		self.D = np.array([self.D[:,1]]).T

	def update_C(self, C_new):
		''' Update the C matrix in the state space model '''

		self.C = C_new
		self.D = np.zeros((self.C.shape[0],1))
		self.continuous_ss()

	def update_D(self, D_new):
		''' update the D matrix in the state space model '''

		self.D = D_new
		self.continuous_ss()

	def set_velocity(self, v):
		''' set a new velocity for the model '''

		self.v = v
		self.calc_state_space_vars()

	def calc_controllability(self):
		''' Calculate the controllability matrix '''

		A_2 = np.dot(self.A, self.A) # A^2 matrix
		A_3 = np.dot(self.A, A_2) # A^3 matrix

		P = np.concatenate((self.B, np.dot(self.A,self.B), np.dot(A_2,self.B), 
									np.dot(A_3,self.B)),axis=1) # controllability matrix
		P_rank = np.linalg.matrix_rank(P) # Calculate the rank of the matrix

		return P, P_rank

	def calc_observability(self):
		''' Calculate the observability matrix '''

		A_2 = np.dot(self.A, self.A) # A^2 matrix
		A_3 = np.dot(self.A, A_2) # A^3 matrix

		Q = np.stack((self.C, np.dot(self.C,self.A), np.dot(self.C,A_2), 
							np.dot(self.C,A_3))) # observabillity matrix
		Q_rank = np.linalg.matrix_rank(Q) # Calculate the rank of the matrix

		return Q, Q_rank

	def calc_eigs(self):
		''' Calculate the eigenvalues of the A matrix '''

		return np.linalg.eigvals(self.A)

	def sort_eigs(self, real_eigs_list, imag_eigs_list):
		''' sort the eigenvalues for plotting '''

		# Convert the eigenvalue lists into arrays
		real_eigs = np.asarray(real_eigs_list)
		imag_eigs = np.asarray(imag_eigs_list)

		# Initialize the arrays for the eigenvalues
		real_eigs_array = np.asarray(real_eigs_list)
		imag_eigs_array = np.asarray(imag_eigs_list)

		eig1_real = [real_eigs_array[0,0]]
		eig2_real = [real_eigs_array[0,1]]
		eig3_real = [real_eigs_array[0,2]]
		eig4_real = [real_eigs_array[0,3]]

		eigs_imag_sorted = np.zeros(imag_eigs_array.shape)
		eigs_imag_sorted[:,0] = imag_eigs_array[:,0]

		eigs_guess = np.zeros((4,4))
		diff_eigs = np.zeros((4,4))

		# Interate through and calculate the sorted eigenvalues
		for idx, eigs in enumerate(real_eigs_array[1:,:]):
			idx = idx+1
			eigs_sort_real = np.sort(eigs)
			eigs_sort_index = np.argsort(eigs)

			eigs_guess[0,:] = (eigs-eig1_real[-1]) + eigs
			eigs_guess[1,:] = (eigs-eig2_real[-1]) + eigs
			eigs_guess[2,:] = (eigs-eig3_real[-1]) + eigs
			eigs_guess[3,:] = (eigs-eig4_real[-1]) + eigs

			# Eigenvalue number 1
			error = (eigs - eig1_real[-1])**2
			eig_prev_index = np.argmin(error)
			eig1_real.append(eigs[eig_prev_index])
			# eigs_imag_sorted[idx,0] = imag_eigs_array[idx,eig_prev_index]
			eigs_imag_sorted[idx,0] = 0 # should be zero

			# Eigenvalue number 2
			error = (eigs - eig2_real[-1])**2
			error_guess = (eigs_guess - eig2_real[-1])**2
			eig_guess_index = np.mod(np.argmin(error_guess), 4)
			eig_prev_index = np.argmin(error)
			eig_index = eig_prev_index
			error_sort = np.sort(error)
			if np.abs(error_sort[0]-error_sort[1]) < 0.01:
				eig_index = eig_guess_index		
			eig2_real.append(eigs[eig_index])
			eigs_imag_sorted[idx,1] = imag_eigs_array[idx,eig_index] # should be zero
			eigs_imag_sorted[idx,1] = 0 # should be zero

			# Eigenvalue number 3
			error = (eigs - eig3_real[-1])**2
			error_guess = (eigs_guess - eig3_real[-1])**2
			eig_guess_index = np.mod(np.argmin(error_guess), 4)
			eig_prev_index = np.argmin(error)
			if np.min(np.diff(eigs_sort_real)) < 0.0001:
				eig_index = np.argmin(np.diff(eigs_sort_real))
				eig3_real.append(eigs[eigs_sort_index[eig_index]])
				eigs_imag_sorted[idx,2] = imag_eigs_array[idx,eigs_sort_index[eig_index]]
			else:
				eig3_real.append(eigs[eigs_sort_index[eig_index]])
				eigs_imag_sorted[idx,2] = imag_eigs_array[idx,eig_prev_index]


			# Eigenvalue number 4
			error = (eigs - eig4_real[-1])**2
			eig_prev_index = np.argmin(error)
			error_sort = np.sort(error)
			if np.min(np.diff(eigs_sort_real)) < 0.0001:
				eig_index = np.argmin(np.diff(eigs_sort_real))
				eig4_real.append(eigs_sort_real[eig_index])
				eigs_imag_sorted[idx,3] = -imag_eigs_array[idx,eigs_sort_index[eig_index]]
			else:
				eig4_real.append(eigs[eig_prev_index])
				eigs_imag_sorted[idx,3] = imag_eigs_array[idx,eig_prev_index]


		eigs_real_sorted = np.asarray([eig1_real, eig2_real, eig3_real, eig4_real]).T

		return eigs_real_sorted, eigs_imag_sorted

	def calc_stable_range(self, real_eigs):
		''' Calculate the stable velocity range where the real components
		of the eigenvalues are less than 0 '''

		CROSS_FLAG = False
		stable_index = []
		for idx, eigs in enumerate(real_eigs):
			if eigs[0]<0 and eigs[1]<0 and eigs[2]<0 and eigs[3]<0 and not CROSS_FLAG:
				CROSS_FLAG = True
				stable_index.append(idx)
			if (eigs[0]>0 or eigs[1]>0 or eigs[2]>0 or eigs[3]>0) and CROSS_FLAG:
				CROSS_FLAG = False
				stable_index.append(idx)

		return stable_index

	def continuous_ss(self):
		''' Create the state space system object '''

		self.sys_c = signal.StateSpace(self.A, self.B, self.C, self.D)
		return self.sys_c

	def discrete_ss(self, dt):
		''' Convert the continuous system to a discrete system '''

		# system = signal.cont2discrete((self.A, self.B, self.C, self.D), dt)
		self.sys_d = signal.StateSpace(self.A, self.B, self.C, self.D).to_discrete(dt)

		return self.sys_d

	def continuous_response(self, time, u, x0=None):
		''' Calculate the reponse with the specified time and
		input vector '''

		system = self.continuous_ss()
		tout, y, x = signal.lsim(system, u, time, X0=x0)
		return tout, x

	def discrete_response(self, time, u, x0=None, dt=1):
		''' Calculate the reponse with the specified time and
		input vector '''

		system = self.discrete_ss(dt)
		tout, y, x = signal.dlsim(system, u, time, x0=x0)
		return tout, x

	def calc_process_noise(self, q_var, dt):
		''' Calculate the process noise used in a Kalman filter '''

		# For this model, the process noise consists of 2 2nd order
		# white noise models.  These correspond to the 2nd
		# order roll variable and the 2nd order steering angle

		# Design the process noise matrix
		Q = np.zeros((4,4))
		q0 = common.Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
		Q[0,0] = q0[0,0]
		Q[0,2] = q0[0,1]
		Q[1,1] = q0[0,0]
		Q[1,3] = q0[0,1]
		Q[2,0] = q0[1,0]
		Q[2,2] = q0[1,1]
		Q[3,1] = q0[1,0]
		Q[3,3] = q0[1,1]

		return Q


if __name__ == "__main__":
	import bike_parameters as params

	bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
	bike.set_velocity(2)

	print(bike.v)