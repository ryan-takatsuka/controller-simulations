# Implement an unscented kalman filter

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control
import bike_parameters as params
from matplotlib import animation
from filterpy import kalman
from filterpy.stats import plot_covariance_ellipse
from BikeModel import BikeModel


# References:
# http://control.ucsd.edu/mauricio/courses/mae280a/chapter5.pdf
# http://www.cds.caltech.edu/~murray/books/AM05/pdf/obc08-trajgen_24Jan08.pdf

def normalize_angle(x):
	x = x % (2 * np.pi)    # force in range [0, 2 pi)
	if x > np.pi:          # move to [-pi, pi)
		x -= 2 * np.pi
	return x

def residual_h(a, b):
	y = a - b
	# data in format [dist_1, bearing_1, dist_2, bearing_2,...]
	for i in range(0, len(y), 2):
		y[i + 1] = normalize_angle(y[i + 1])
	return y

def residual_x(a, b):
	y = a - b
	y[2] = normalize_angle(y[2])
	return y


# Set some constants
velocity = 1 # [m/s] The velocity of the car
wheelbase = 0.5 # [m] The length of the wheelbase
dt = 1 # [sec] The timestep
N = 200 # The number of points in the simulation


bike = BikeModel(params.M, params.C_1, params.K_0, params.K_2)
bike.set_velocity(velocity)
# sys = bike.discrete_ss(dt)


# Define the state transition function
def move(x0, dt, u, wheelbase):
	x = x0[0]
	y = x0[1]
	heading = x0[2]
	velocity = u[0]
	steering_angle = u[1]
	distance = velocity * dt

	x_dot = velocity * np.cos(heading)
	y_dot = velocity * np.sin(heading)
	heading_dot = velocity / wheelbase * np.tan(steering_angle)

	x = x_dot * dt + x
	y = y_dot * dt + y
	heading = heading_dot * dt + heading

	return np.array([x, y, heading])
# Create the measurement function
def Hx(x, landmarks):
	""" takes a state variable and returns the measurement
	that would correspond to that state. """
	hx = []
	for lmark in landmarks:
		px, py = lmark
		dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
		angle = np.arctan2(py - x[1], px - x[0])
		hx.extend([dist, normalize_angle(angle - x[2])])
	return np.array(hx)

ref_path = [np.array([2, 6, .3])]
for n in range(N):
	ref_path.append(move(ref_path[-1], 
		dt/10, np.array([1.1, .01]), wheelbase))
ref_path = np.array(ref_path)
ref_landmarks = ref_path[0::10,0:2]


landmarks = np.array([[5, 10], [10, 5], [15, 15]])
cmds = [np.array([1.1, 0])] * N
sigma_vel = 0.1
sigma_steer = np.radians(1)
sigma_range = 0.3
sigma_bearing = 0.1
ellipse_step = 1
step = 10

print(landmarks)
print(ref_landmarks)
landmarks = ref_landmarks

points = kalman.MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
									subtract=residual_x)


# ukf = kalman.UnscentedKalmanFilter(dim_x=3, dim_z=2*len(landmarks), 
# 								fx=move, hx=Hx,
# 								dt=dt, points=points, x_mean_fn=state_mean, 
# 								z_mean_fn=z_mean, residual_x=residual_x, 
# 								residual_z=residual_h)
ukf = kalman.UnscentedKalmanFilter(dim_x=3, dim_z=2*len(landmarks), 
								fx=move, hx=Hx,
								dt=dt, points=points, 
								residual_x=residual_x, 
								residual_z=residual_h)


ukf.x = np.array([2, 6, .3])
ukf.P = np.diag([.1, .1, .05])
ukf.R = np.diag([sigma_range**2, 
				 sigma_bearing**2]*len(landmarks))
ukf.Q = np.eye(3)*0.0001

sim_pos = ukf.x.copy()

plt.figure()
# plot landmarks
if len(landmarks) > 0:
	plt.scatter(landmarks[:, 0], landmarks[:, 1], 
				marker='s', s=60)

track = []
for i, u in enumerate(cmds):     
	sim_pos = move(sim_pos, dt/step, u, wheelbase)
	track.append(sim_pos)

	if i % step == 0:
		try:
			ukf.predict(u=u, wheelbase=wheelbase)
		except:  # force positive definite matrix
			ukf.P = np.matmul(ukf.P, ukf.P.T)
			ukf.predict(u=u, wheelbase=wheelbase)		

		if i % ellipse_step == 0:
			plot_covariance_ellipse(
				(ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
				 facecolor='k', alpha=0.3)

		x, y = sim_pos[0], sim_pos[1]
		z = []
		for lmark in landmarks:
			dx, dy = lmark[0] - x, lmark[1] - y
			d = np.sqrt(dx**2 + dy**2) + np.random.randn()*sigma_range
			bearing = np.arctan2(lmark[1] - y, lmark[0] - x)
			a = (normalize_angle(bearing - sim_pos[2] + 
				 np.random.randn()*sigma_bearing))
			z.extend([d, a])            
		ukf.update(z, landmarks=landmarks)

		if i % ellipse_step == 0:
			plot_covariance_ellipse(
				(ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
				 facecolor='g', alpha=0.8)
track = np.array(track)
plt.plot(track[:, 0], track[:,1], color='k', lw=2)
plt.axis('equal')
plt.title("UKF localization")
# plt.show()

plt.figure()
# plt.plot(track[:,0], track[:,1], color='k', lw=2)
plt.plot(ref_path[:,0], ref_path[:,1])
plt.scatter(ref_landmarks[:,0], ref_landmarks[:,1])
plt.show()
