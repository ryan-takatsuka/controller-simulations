from __future__ import division, print_function, absolute_import

import numpy as np
import scipy
from filterpy.kalman import KalmanFilter
from filterpy import common

def simulate_kalman(A, B, C, D, K, dt, time, 
					r_var, q_var, noise=0, x0=None,
					sensors=None, Q=None, x_ref=None):
	''' Simulate a continuous system with discrete sensors and controller '''

	# Calculate the reference control input for the specified state target
	if x_ref is None:
		x_ref = np.zeros((1, A.shape[1]))
	u_ref = calc_uref(A, B, K, x_ref)

	dim_z = C.shape[0]
	dim_x = A.shape[0]

	sys_d = scipy.signal.StateSpace(A, B, C, D).to_discrete(dt)
	kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
	kf.F = sys_d.A
	kf.B = sys_d.B
	kf.H = sys_d.C
	kf.R *= r_var
	kf.Q = common.Q_discrete_white_noise(dim=dim_x, dt=dt, var=q_var)
	if Q is not None:
		kf.Q = Q
	kf.P *= 1000

	sensor_time = [0]
	states = [x0]
	est_states = [x0]
	measurements = [C @ x0]
	control_input = [-K@est_states[-1]]
	for idx, t in enumerate(time[1:]):
		states_dot = A@states[-1] + np.dot(B, control_input[-1])
		states.append(states[-1] + states_dot*(t-time[idx]))

		if (t-sensor_time[-1])>=dt:
			# Measure the data and apply the kalman filter
			y = C @ states[-1]
			for idx, sensor in enumerate(sensors):
				y[idx,:] = sensor.readSingleValue(y[idx,:])
			measurements.append(y)
			kf.predict(u=control_input[-1])
			kf.update(measurements[-1])

			# Add varaibles to list
			est_states.append(kf.x)
			control_input.append(-K@est_states[-1])
			sensor_time.append(t)

			# Convert output variables to arrays
	states = np.asarray(states)
	est_states = np.asarray(est_states)
	measurements = np.asarray(measurements)
	control_input = np.squeeze(np.asarray(control_input))
	return time, states, est_states, sensor_time, measurements, control_input


def simulate(A, B, C, D, K, dt, time, noise=0, x0=None, x_ref=0):
	''' Simulate a continuous system with discrete sensors and controller '''

	# Calculate the reference control input for the specified state target
	u_ref = calc_uref(A, B, K, x_ref)

	if x0 is None:
		x0 = np.zeros((A.shape[0], 1))

	sensor_time = [0]
	states = [x0]
	measurements = [x0]
	control_input = [-K@measurements[-1]]
	for idx, t in enumerate(time[1:]):
		states_dot = A@states[-1] + np.dot(B, control_input[-1])
		states.append(states[-1] + states_dot*(t-time[idx]))

		if (t-sensor_time[-1])>=dt:
			measurements.append(states[-1] + np.random.randn()*noise)
			# print(measurements[-1])
			control_input.append(u_ref-K@(measurements[-1]))
			sensor_time.append(t)

	# Convert output variables to arrays
	states = np.asarray(states)
	measurements = np.asarray(measurements)
	control_input = np.squeeze(np.asarray(control_input))
	return time, states, sensor_time, measurements, control_input


def simulate_discrete(a, b, c, d, dt, K, u, t=None, x0=None):
	''' simulate the closed_loop response '''

	if t is None:
		out_samples = max(u.shape)
		stoptime = (out_samples - 1) * dt
	else:
		stoptime = t[-1]
		out_samples = int(np.floor(stoptime / dt)) + 1

	# Pre-build output arrays
	xout = np.zeros((out_samples, a.shape[0]))
	yout = np.zeros((out_samples, c.shape[0]))
	tout = np.linspace(0.0, stoptime, num=out_samples)
	uout = np.zeros((out_samples, d.shape[1]))

	# Check initial condition
	if x0 is None:
		xout[0,:] = np.zeros((a.shape[1],))
	else:
		xout[0,:] = np.asarray(x0)

	# Pre-interpolate inputs into the desired time steps
	if t is None:
		u_dt = u
	else:
		if len(u.shape) == 1:
			u = u[:, np.newaxis]

		u_dt_interp = interp1d(t, u.transpose(), copy=False, bounds_error=True)
		u_dt = u_dt_interp(tout).transpose()

	# Simulate the system
	for i in range(0, out_samples - 1):
		uout[i,:] = -K @ xout[i,:]
		xout[i+1,:] = a @ xout[i,:] + np.dot(b, uout[i,:]).reshape(1,-1)
		# xout[i+1,:] = (a-b@K) @ xout[i,:] + (b * u_dt[i]).reshape(1,-1)
		# xout[i+1,:] = np.dot(a-b@K, xout[i,:]) + np.dot(b, u_dt[i]).reshape(1,-1)
		# xout[i+1,:] = np.dot(a, xout[i,:]) + np.dot(b, u_dt[i]).reshape(1,-1)
	uout[-1] = -K @ xout[-1,:]

	return tout, xout, uout


def lqr(A,B,Q,R):
	"""Solve the continuous time lqr controller.
	 
	dx/dt = A x + B u
	 
	cost = integral x.T*Q*x + u.T*R*u
	"""
	#ref Bertsekas, p.151
	 
	#first, try to solve the ricatti equation
	X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
	 
	#compute the LQR gain
	K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
	 
	eigVals, eigVecs = scipy.linalg.eig(A-B*K)
	 
	return K, X, eigVals
 
def dlqr(A,B,Q,R):
	"""Solve the discrete time lqr controller.
	 
	x[k+1] = A x[k] + B u[k]
	 
	cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
	"""
	#ref Bertsekas, p.151
	 
	#first, try to solve the ricatti equation
	X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	 
	#compute the LQR gain
	K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
	 
	eigVals, eigVecs = scipy.linalg.eig(A-B*K)
	 
	return K, X, eigVals

def calc_poles(overshoot, natural_freq, num_poles):
	''' Calculate the poles from the specified overshoot and natural frequency '''

	overshoot = overshoot / 100
	zeta = np.sqrt((np.log(overshoot)**2) / (np.pi**2 + np.log(overshoot)**2))
	wn = natural_freq * 2 * np.pi

	poles = []
	poles.append((-zeta + np.sqrt(zeta**2 - 1 + 0j))*wn)
	poles.append((-zeta - np.sqrt(zeta**2 - 1 + 0j))*wn)

	for idx in range(num_poles-2):
		poles.append(-(idx+4)*wn)

	return poles

def calc_Nbar(A, B, C, D, K):
	''' Calculate the scaling factor for a reference input to a state-space
	system with the specified controller, K.  This only works for a single 
	specified reference state, which means C has to be of size [1,n]. '''

	s = A.shape[0]
	Z = np.append(np.zeros(s), np.array(1))
	N0 = np.concatenate((A, B), axis=1)
	N0 = np.concatenate((N0, np.concatenate((C, D), axis=1)))
	N = np.linalg.inv(N0) @ Z.T
	Nx = N[0:s]
	Nu = N[s]
	Nbar = Nu + K @ Nx
	return Nbar[0,0]

def calc_uref(A, B, K, x_ref):
	''' Calculate the required reference input from the reference state.
	The reference state can only contain 1 non-zero element.  The remaining elements must be zero. '''

	xref_sort = np.sort(x_ref)
	if np.sum(np.abs(xref_sort[0:-1])) != 0:
		print('There can only be 1 non-zero element in x_ref')

	C = x_ref / np.max(x_ref)
	if len(C.shape) == 1:
		C = np.array([C])
	D = np.zeros((1, B.shape[1]))

	Nbar = calc_Nbar(A, B, C, D, K)
	u_ref = Nbar * np.max(x_ref)

	return u_ref
