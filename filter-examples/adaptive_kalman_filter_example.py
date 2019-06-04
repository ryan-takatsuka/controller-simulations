# Adaptive Kalman filter example

import tracking_kalman_filter as tkf
import matplotlib.pyplot as plt
import numpy as np

dt = 1/125
r_var = 0.1
q_var= 10


p = tkf.myPath(dt)
sensor = tkf.mySensor(np.sqrt(r_var))
cv_filter = tkf.constantVelocityFilter(r_var, q_var, dt)

p.addLineSegment(100, 0, 0)
p.addLineSegment(100, 0, 10)
p.addLineSegment(100, 10, 0)
p.addSineSegment(100, 10, 5)
p.addLineSegment(1000, 0, 0)
# p.addLineSegment(200, 0, 0)
# p.addSineSegment(100, 100, 0.5)
# p.addSineSegment(100, 100, 5)
# p.addLineSegment(100, p.path[-1], p.path[-1]-50)
# p.addLineSegment(100, p.path[-1], 0)
# p.addLineSegment(500, 0, 0)
p.smoothData()

# Create noise in the measurements
measurements = sensor.readManyValues(p.path)

# Filter the measurement data
s0 = cv_filter.standard_filter(measurements)
x1, Q1 = cv_filter.adaptive_Q_filter(measurements, 5, 10)
x2, Q2, std2 = cv_filter.zarchan_adaptive_filter(measurements, 1, 100000)

print(s0.x)


# plt.figure()
# plt.plot(p.sample, p.path)
# plt.plot(p.sample, x0[:,0], label='standard')
# plt.plot(p.sample, x1[:,0], label='adaptive Q')
# plt.plot(p.sample, x2[:,0], label='Zarchan')
# plt.scatter(p.sample, measurements, s=3, c='k')
# plt.legend()
# # plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(p.sample, Q1, label='adaptive Q')
# plt.subplot(2,1,2)
# plt.plot(p.sample, Q2, label='Zarchan')
# plt.legend()
# # plt.show()

# plt.figure()
# plt.plot(p.sample, std2)
# plt.show()