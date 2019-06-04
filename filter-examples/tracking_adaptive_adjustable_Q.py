# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Kalman filter test for Hyleas

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
from filterpy import common


f = KalmanFilter(dim_x=2, dim_z=1)

# Adaptive filter parameters
Q_scale_factor = 100
eps_max = 5

# Simulation parameters
n_iters = 100*10 # number of time steps (num of iterations)
timestep = 1 / 125 # [s]
time = np.arange(0, n_iters*timestep, timestep) # time vector
sz = (n_iters,) # size of array

# Create the actual motion data
b, a = signal.butter(2, 0.08, 'low') # design filter for realistic positions
n_partial = int(n_iters/10)
p_true = np.zeros(n_partial)
p_true = np.concatenate((p_true, 100*np.sin(0.5*2*np.pi*time[0:n_partial])))
p_true = np.concatenate((p_true, 100*np.sin(5*2*np.pi*time[0:n_partial])))
p_true = np.concatenate((p_true, p_true[-1]*np.ones(n_partial)))
p_true = np.concatenate((p_true, np.linspace(p_true[-1], p_true[-1]-50, n_partial)))
p_true = np.concatenate((p_true, np.linspace(p_true[-1], 0, n_partial)))
p_true = np.concatenate((p_true, np.zeros(n_partial)))
p_true = np.concatenate((p_true, np.zeros(n_partial)))
p_true = np.concatenate((p_true, np.zeros(n_partial)))
p_true = np.concatenate((p_true, np.zeros(n_partial)))
p_true = signal.lfilter(b, a, p_true)

z = p_true + np.random.normal(0,0.1,size=sz) # noise added to measurement

# Initialize the filterpy kalman filter
f.x = np.array([[0], # initial condition for system states
				[0]])
f.F = np.array([[1, timestep], # A matrix
				[0, 1]])
f.H = np.array([[1, 0]]) # C matrix
f.P = np.array([[1, 0], # covariance intial condition
				[0, 1]])
f.R = 0.1**2 # measurement variance
f.Q = common.Q_discrete_white_noise(dim=2, dt=timestep, var=1) # process variance

# Simulate the kalman filter over time
x_kal = np.zeros(n_iters)
v_kal = np.zeros(n_iters)
epss = np.zeros(n_iters)
Q_list = []
count = 0
for k in range(n_iters):
	f.predict()
	f.update(z[k])
	epss[k] = np.dot(f.y.T, np.linalg.inv(f.S)).dot(f.y)
	x_kal[k] = f.x[0]
	v_kal[k] = f.x[1]

	if epss[k] > eps_max:
		f.Q *= Q_scale_factor
		count +=1
	elif count > 0:
		f.Q /= Q_scale_factor
		count -= 1
	Q_list.append(np.max(f.Q))

# The numerical derivative
dz_dt = np.concatenate((np.array([0]), np.diff(z))) / timestep

# Calculate thenoise values
rms_z = np.sqrt(np.mean(z[-300:]**2))
rms_z_kal = np.sqrt(np.mean(x_kal[-300:]**2))
print('RMS measurement: ', rms_z)
print('RMS Kalman: ', rms_z_kal)

# Plot the results
plt.figure()
# plt.subplot(2,1,1)
plt.scatter(time, z, s=2)
plt.step(time, x_kal, c='k')

# plt.subplot(2,1,2)
# plt.scatter(time, dz_dt, s=2)
# plt.plot(time, v_kal, c='k')
# plt.show()


plt.figure()
plt.plot(time, epss)
# plt.show()

plt.show()
