import numpy as np
import control
from scipy import signal

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

		# Calculate the crossover points for the eigenvalues
		min_val = 1e-2 # threshold for determining if values are equal
		FIRST_CROSS = False # flag for the first crossover
		SECOND_CROSS = False # flag fort the second crossover
		cross_index = [] # initial crossover index list
		for idx, eigs in enumerate(real_eigs_list):
			eigs = np.sort(eigs)
			eigs_diff = np.sort(np.abs(np.diff(eigs)))
			eigs_diff_index = np.argsort(np.abs(np.diff(eigs)))
			if eigs_diff[0] < min_val and not FIRST_CROSS:
				cross_index.append([idx, eigs[eigs_diff_index[0]]])
				FIRST_CROSS = True
			if FIRST_CROSS and not SECOND_CROSS and eigs_diff[0]<min_val and eigs_diff[1]<min_val:
				cross_index.append([idx, eigs[eigs_diff_index[0]]])
				SECOND_CROSS = True

		print(cross_index)

		# Sort the real eigenvalues to track each individual value
		new_real_eigs = np.zeros(real_eigs.shape)
		for idx, eigs in enumerate(real_eigs_list):
			for lam in eigs:
				if idx < cross_index[0][0]: # section before first crossover
					if lam >= cross_index[0][1]:
						new_real_eigs[idx,0] = lam
					if lam <= cross_index[0][1] and lam > 0:
						new_real_eigs[idx,1] = lam
					if lam <= 0 and lam >= np.min(real_eigs[0,:]):
						new_real_eigs[idx,2] = lam
					if lam <= np.min(real_eigs[0,:]):
						new_real_eigs[idx,3] = lam
				if idx >= cross_index[0][0] and idx < cross_index[1][0]: # middle section
					if lam >= cross_index[1][1]:
						new_real_eigs[idx,0] = lam
						new_real_eigs[idx,1] = lam
					if lam <= 0 and lam >= np.min(real_eigs[0,:]):
						new_real_eigs[idx,2] = lam
					if lam <= np.min(real_eigs[0,:]):
						new_real_eigs[idx,3] = lam
				if idx >= cross_index[1][0]: # final section
					if lam < cross_index[1][1] and lam > new_real_eigs[cross_index[1][0],3]:
						new_real_eigs[idx,0] = lam
						new_real_eigs[idx,1] = lam
					if lam >= cross_index[1][1]:
						new_real_eigs[idx,2] = lam
					if lam <= new_real_eigs[cross_index[1][0],3]:
						new_real_eigs[idx,3] = lam

		# Sort the imaginary eigenvalues
		new_imag_eigs = np.zeros(imag_eigs.shape)
		for idx, eigs in enumerate(imag_eigs_list):
			for lam in eigs:
				if idx < cross_index[0][0]:
					new_imag_eigs[idx,0] = lam
					new_imag_eigs[idx,1] = lam
					new_imag_eigs[idx,2] = lam
					new_imag_eigs[idx,3] = lam
				if idx >= cross_index[0][0]:
					if lam > min_val:
						new_imag_eigs[idx,0] = lam
					if lam < -min_val:
						new_imag_eigs[idx,1] = lam
					if lam > -min_val and lam < min_val:
						new_imag_eigs[idx,2] = lam
						new_imag_eigs[idx,3] = lam

		return new_real_eigs, new_imag_eigs, cross_index

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


if __name__ == "__main__":
	import bike_parameters as params

	bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
	bike.set_velocity(2)

	print(bike.v)