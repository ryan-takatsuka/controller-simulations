from __future__ import division, print_function, absolute_import

import numpy as np
import scipy


def rk4(f, x0, y0, x1, n):
	vx = [0] * (n + 1)
	vy = [0] * (n + 1)
	h = (x1 - x0) / float(n)
	vx[0] = x = x0
	vy[0] = y = y0
	for i in range(1, n + 1):
		k1 = h * f(x, y)
		k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
		k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
		k4 = h * f(x + h, y + k3)
		vx[i] = x = x0 + i * h
		vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
	return vx, vy
 
def f(A, B, C, D, K):
	return x * sqrt(y)

def simulate(A, B, C, D, K, dt, time, noise=0, x0=None):
	''' Simulate a continuous system with discrete sensors and controller '''

	sensor_time = [0]
	states = [x0]
	measurements = [x0]
	control_input = [-K@measurements[-1]]
	for idx, t in enumerate(time[1:]):
		states_dot = A@states[-1] + np.dot(B, control_input[-1])
		states.append(states[-1] + states_dot*(t-time[idx]))

		if (t-sensor_time[-1])>=dt:
			measurements.append(states[-1] + np.random.randn()*noise)
			control_input.append(-K@measurements[-1])
			sensor_time.append(t)


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