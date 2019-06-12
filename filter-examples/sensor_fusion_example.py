# Adaptive Kalman filter example

import tracking_kalman_filter as tkf
import matplotlib.pyplot as plt
import numpy as np
import tracking_kalman_filter as tkf
from sensor import Sensor
from filterpy.kalman import KalmanFilter
from filterpy import common

# general parameters
sensor1_rate = 100 # Hz
sensor2_rate = 100 # Hz
sensor1_var = 0.1 # variance
sensor2_var = 1 # variance

# Create the true path
dt = 1/sensor1_rate
time = np.linspace(0,10,10000)
path = 10*np.sin(0.1 * 2*np.pi*time)

# Create sensor measurements of the path
sensor1 = Sensor(sensor1_var, sensor1_rate)
sensor2 = Sensor(sensor2_var, sensor2_rate)
sensor1.read(path, time)
sensor1.add_distortion(0.5, 0.7)
sensor2.read(path, time)

# Create the filter
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = np.array([[1, dt],
				 [0, 1]])
kf.H = np.array([[1, 0],
				 [1, 0]])
kf.R = np.array([[sensor1_var, 0],
				 [0, sensor2_var]])
kf.Q = common.Q_discrete_white_noise(dim=2, dt=dt, var=1000)

saver = common.Saver(kf)
r = []
r2 = []
e_limit = 3
sensor1_R_scale_factor = 10
count_R = 0
count_Q = 0
eps_max = 3
Q_scale_factor = 10
for idx in range(len(sensor1.measurements)):
	z = np.array([[sensor1.measurements[idx]],
				  [sensor2.measurements[idx]]])

	kf.predict()
	kf.update(z)
	saver.save()
	r.append(np.dot(kf.y.T, np.linalg.inv(kf.S)).dot(kf.y)[0][0])
	r2.append((z[0]-z[1])**2)
	epss = np.dot(kf.y.T, np.linalg.inv(kf.S)).dot(kf.y)[0][0]

	if r2[-1] > e_limit and count_R < 100:
		kf.R[0,0] *= sensor1_R_scale_factor
		count_R += 1
	elif count_R > 0:
		kf.R[0,0] /= sensor1_R_scale_factor
		count_R -= 1


	if epss > eps_max:
		kf.Q *= Q_scale_factor
		count_Q += 1
	elif count_Q > 0:
		kf.Q /= Q_scale_factor
		count_Q -= 1


	# print(kf.R)


saver.to_array()

plt.figure()
plt.plot(time, path)
plt.scatter(sensor1.time, sensor1.measurements, s=5, c='k')
plt.scatter(sensor2.time, sensor2.measurements, s=5, c='r')
plt.plot(sensor1.time, saver.x[:,0])
plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(sensor1_time, saver.y[:,0,0])
# plt.subplot(2,1,2)
# plt.plot(sensor1_time, saver.y[:,1,0])
# # plt.show()

# plt.figure()
# plt.plot(r2)
# plt.show()