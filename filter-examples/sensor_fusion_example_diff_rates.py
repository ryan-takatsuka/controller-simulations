# Adaptive Kalman filter example

import tracking_kalman_filter as tkf
import matplotlib.pyplot as plt
import numpy as np
import tracking_kalman_filter as tkf
from sensor import Sensor
from filterpy.kalman import KalmanFilter

# general parameters
sensor1_rate = 250 # Hz
sensor2_rate = 60 # Hz
sensor1_var = 0.1 # variance
sensor2_var = 1 # variance

# Create the true path
time = np.linspace(0,1,1000)
path = 10*np.sin(0.1 * 2*np.pi*time)

# Create sensor measurements of the path
sensor1 = Sensor(sensor1_var, sensor1_rate)
sensor2 = Sensor(sensor2_var, sensor2_rate)
sensor1_time, sensor1_z = sensor1.read(path, time)
sensor2_time, sensor2_z = sensor2.read(path, time)

# Create the filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = array()



dt = 1/125
r_var = 0.1**2
q_var= 10

sensor = tkf.mySensor(r_var)
cvfilter = tkf.standard_cvfilter(r_var, q_var, dt)
adaptive_Q_cvfilter = tkf.adaptive_Q_cvfilter(r_var, 1, dt, 5, 10)
zarchan_adaptive_cvfilter = tkf.zarchan_adaptive_cvfilter(r_var, 1, dt, 1, 10000)




plt.figure()
plt.plot(time, path)
plt.scatter(sensor1_time, sensor1_z, s=5, c='k')
plt.scatter(sensor2_time, sensor2_z, s=5, c='r')
plt.show()

