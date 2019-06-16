'''
This script analyzes the bike model with a reduced number of sensors.  
This assumes that not all the state are being measured, and some
need to be estimated.  It also assumes that there is gaussian noise in
the sensor readings that needs to be filtered out.

For this simulation, it is assumed that there is a steer rate and roll
rate sensor (gyroscopes) because those states are cheap and easy to measure.
'''

import numpy as np
from matplotlib import pyplot as plt
import bike_parameters as params
from BikeModel import BikeModel
import controller_design
from sensor import Sensor
import csv
from filterpy.kalman import KalmanFilter
from filterpy import common

filename = "real_data\LSM6DSM_Gyroscope.csv"

time = []
x = []
y = []
z = []
with open(filename) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        time.append(float(row[0]))
        x.append(float(row[1]))
        y.append(float(row[2]))
        z.append(float(row[3]))
        
time = np.asarray(time)
time = (time - time[0]) / 1000
measurements = np.asarray(y)        

#plt.figure()
#plt.plot(time, x)
#plt.figure()
#plt.plot(time, y)
#plt.figure()
#plt.plot(time, z)
#plt.show()



# Set debuggin print options to fit entire matrices
np.set_printoptions(suppress=True, linewidth=200)

# Parameters for the simulation
x0 = np.array([[0, 0, 0, 0]]).T  # initial states
dt = 1/1000  # sampling time for the sensor
velocity = 5  # velocity of the bike
r_var = 1e-2
q_var = 8

# Create the sensors
roll_gyro_sensor = Sensor(r_var)
steer_gyro_sensor = Sensor(r_var)
sensors = [roll_gyro_sensor, steer_gyro_sensor]

# Create the bike model
# C = np.array([[1, 0, 0, 0],
# 			  [0, 1, 0, 0]])
C = np.array([[0, 0, 1, 0]])
D = np.zeros(1)
bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
bike.update_C(C)
sys = bike.discrete_ss(dt)
print(sys.A)
print(sys.B)
print(sys.C)
print(sys.D)

#dt = 1/60
#t_sim, x_sim = bike.continuous_response(time, np.zeros(time.size), 
#                                        x0=np.array([np.deg2rad(6), 0, 0, 0]))
#measurements = x_sim[:,2] + np.random.randn(t_sim.size)*r_var
#print(measurements)

kf = KalmanFilter(dim_x=4, dim_z=1)
kf.F = sys.A
kf.B = sys.B
kf.H = sys.C
kf.R *= r_var
kf.Q = bike.calc_process_noise(q_var, dt)
kf.P *= 1

saver = common.Saver(kf)
for idx in range(time.size):
    kf.predict()
    kf.update(measurements[idx])
    saver.save()
saver.to_array()


plt.figure()
plt.subplot(2,2,1)
plt.plot(time, np.rad2deg(saver.x[:,0,0]))
plt.title('Roll angle')
plt.xlabel('Time [sec]')

plt.subplot(2,2,2)
plt.plot(time, np.rad2deg(saver.x[:,1,0]))
plt.title('Steer angle')
plt.xlabel('Time [sec]')

plt.subplot(2,2,3)
plt.plot(time, np.rad2deg(saver.x[:,2,0]))
plt.scatter(time, np.rad2deg(measurements), s=2, c='k')
plt.title('Roll angular velocity')
plt.xlabel('Time [sec]')

plt.subplot(2,2,4)
plt.plot(time, np.rad2deg(saver.x[:,3,0]))
plt.title('Steer angular velocity')
plt.xlabel('Time [sec]')

plt.figure()
plt.plot(time, np.rad2deg(saver.x[:,2,0]), label='Kalman Filter Estimate')
plt.scatter(time, np.rad2deg(measurements), s=2, c='k', label='Measurements')
plt.title('Roll angular velocity')
plt.xlabel('Time [sec]')
plt.ylabel('Angular Velocity [deg/sec]')
plt.xlim([20, 70])
plt.grid()
plt.legend()
plt.show()



