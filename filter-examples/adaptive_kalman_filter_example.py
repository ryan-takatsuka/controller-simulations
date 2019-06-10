# Adaptive Kalman filter example

import tracking_kalman_filter as tkf
import matplotlib.pyplot as plt
import numpy as np

# general parameters
dt = 1/125
r_var = 0.1**2
q_var= 10

p = tkf.myPath(dt)
sensor = tkf.mySensor(r_var)
cvfilter = tkf.standard_cvfilter(r_var, q_var, dt)
adaptive_Q_cvfilter = tkf.adaptive_Q_cvfilter(r_var, 1, dt, 5, 10)
zarchan_adaptive_cvfilter = tkf.zarchan_adaptive_cvfilter(r_var, 1, dt, 1, 10000)

p.addLineSegment(100, 0, 0)
p.addLineSegment(100, 0, 10)
p.addLineSegment(100, 10, 0)
p.addSineSegment(100, 10, 5)
p.addLineSegment(300, 0, 0)
p.smoothData()

# Create noise in the measurements
measurements = sensor.readManyValues(p.path)

# Filter the measurement data
s0 = cvfilter.filter_data(measurements)
s1 = adaptive_Q_cvfilter.filter_data(measurements)
s2 = zarchan_adaptive_cvfilter.filter_data(measurements)

print('----- Standard Constant Volume Filter -----')
tkf.calc_noise(cvfilter, sensor)
print('\n----- Adaptive Process Noise Filter -----')
tkf.calc_noise(adaptive_Q_cvfilter, sensor)
print('\n----- Zarchan Adaptive Filter -----')
tkf.calc_noise(zarchan_adaptive_cvfilter, sensor)

plt.figure()
plt.plot(p.sample, p.path)
plt.step(p.sample, s0.x[:,0], label='standard')
plt.step(p.sample, s1.x[:,0], label='adaptive Q')
plt.step(p.sample, s2.x[:,0], label='Zarchan')
plt.scatter(p.sample, measurements, s=3, c='k')
plt.legend()
# plt.show()

plt.figure()
plt.subplot(3,1,1)
plt.plot(p.sample, np.maximum(s0.Q[:,0,0], s0.Q[:,1,1]))
plt.subplot(3,1,2)
plt.plot(p.sample, np.maximum(s1.Q[:,0,0], s1.Q[:,1,1]))
plt.subplot(3,1,3)
plt.plot(p.sample, np.maximum(s2.Q[:,0,0], s2.Q[:,1,1]))
plt.show()
