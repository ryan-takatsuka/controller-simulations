# Implement an unscented kalman filter

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control
from CarModel import normalize_angle, state_mean, z_mean
from matplotlib import animation
from filterpy import kalman
from filterpy.stats import plot_covariance_ellipse


# References:
# http://control.ucsd.edu/mauricio/courses/mae280a/chapter5.pdf
# http://www.cds.caltech.edu/~murray/books/AM05/pdf/obc08-trajgen_24Jan08.pdf

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
N = 10 # The number of points in the simulation

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
for n in range(200):
	ref_path.append(move(ref_path[-1], 
		dt/10, np.array([1.1, .01]), wheelbase))
ref_path = np.array(ref_path)
ref_landmarks = ref_path[0::10,0:2]


landmarks = np.array([[5, 10], [10, 5], [15, 15]])
cmds = [np.array([1.1, 0])] * 200
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


ukf = kalman.UnscentedKalmanFilter(dim_x=3, dim_z=2*len(landmarks), 
								fx=move, hx=Hx,
								dt=dt, points=points, x_mean_fn=state_mean, 
								z_mean_fn=z_mean, residual_x=residual_x, 
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
		except:
			ukf.P = np.matmul(ukf.P, ukf.P.T) # force positive definite
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

# print(cmds)


# # Calculate the controllability matrix
# C0 = control.ctrb(A, B)
# print("Controllability matrix rank =", np.linalg.matrix_rank(C0), "(should be 2)")

# # Calculate the LQR parameters
# Q = C0.T*C0
# R = 1

# # Calculate the controller gain, K
# K, S, E = control.lqr(A, B, Q, R)
# print("Controller gain =", K[0])


# # Create reference path
# initial_pose = CarPose(0, 0, 0) # initial Pose
# ref_path = [initial_pose] # Initialize the list of reference paths
# ref_steer_ang = np.linspace(0, 1, N) * 0.1 # Create the steering angle

# # Create the reference path of N points
# for i in range(N):
# 	ref_path.append(Step(ref_path[-1], velocity, ref_steer_ang[i], dt, L))


# # -------- SIMULATION -------------
# # The initial pose
# initial_pose.theta = np.pi/2 # Adjust the starting orientation to 90 off from the reference path

# def Simulate(initial_pose):
#     # Initialize some variables
#     control_steer_angle = np.zeros(N) # The controlled steering angle
#     u_K_ar = np.zeros(N) # The controlled steering angle
#     state_error = np.zeros((N,2)) # The error between the current state and the reference states
#     sim_path = [initial_pose] # Initialize the starting simulation path
	
#     # Iterate through the N points and calculate a controlled steering angle
#     for i in range(N):
#         # Define the current/recent pose
#         current_pose = sim_path[-1]

#         # Define the expected pose from the reference path
#         current_ref_pose = ref_path[i]

#         # Calculate the difference between the current states and the reference states (only y and theta!!)
#         state_diff = current_pose.subtract(current_ref_pose).toStateVector()[1:]

#         # Calculate the controller output (Note: this is only the controlled output for the error system, not the actual system)
#         # The reference steering angle must be added to this to calculate the actual output
#         u_K = np.dot(-K, state_diff)[0,0]
#         control_steer_angle[i] = np.arctan(u_K) + ref_steer_ang[i]

#         # Calculate the pose and add to the simulation list
#         sim_path.append(Step(current_pose, velocity, control_steer_angle[i], dt, L))

#         # Add the state error and controller output to an array for plotting
#         state_error[i,:] = state_diff.T
#        	u_K_ar[i] = u_K

		
#     return sim_path, state_error, control_steer_angle, u_K_ar

# sim_path, state_error, control_steer_angle, u_K_ar = Simulate(initial_pose)


# # The simulation path plotting variables
# sim_position = np.zeros((N+1,4))
# for idx, pose in enumerate(sim_path):
# 	sim_position[idx,0] = pose.x
# 	sim_position[idx,1] = pose.y
# 	sim_position[idx,2], sim_position[idx,3] = linearProjectForward(pose, 1) 

# # The reference path plotting variables
# ref_position = np.zeros((N+1,2))
# for idx, pose in enumerate(ref_path):
# 	ref_position[idx,0] = pose.x
# 	ref_position[idx,1] = pose.y


# # Plot the reference path and simulated path
# plt.figure()
# plt.plot(ref_position[:,0], ref_position[:,1], label="reference")
# plt.plot(sim_position[:,0], sim_position[:,1], label="simulation")
# plt.title("Simulated Car Output")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.legend()

# # Plot the steering angle applied to the car
# plt.figure()
# plt.plot(control_steer_angle, label="Steering Angle")
# plt.plot(u_K_ar, label="Direct Controller Output")
# plt.plot(ref_steer_ang, label="Reference Steering Angle")
# plt.legend()
# plt.title("Control Variables")
# plt.xlabel("Step Number")
# plt.ylabel("Value")

# # Plot the state errors
# plt.figure()
# plt.plot(state_error[:,0], label="Y error")
# plt.plot(state_error[:,1], label="theta error")
# plt.title("State errors")
# plt.xlabel("Step Number")
# plt.ylabel("State error")
# plt.legend()

# # Create the figure and axes
# fig = plt.figure()
# ax = plt.axes(xlim=(min(ref_position[:,0]), max(ref_position[:,0])), ylim=(min(ref_position[:,1]), max(ref_position[:,1])))

# # Initialize plot items
# reference_line, = ax.plot(ref_position[:,0], ref_position[:,1], label='Reference path')
# car_line, = ax.plot([], [], label='Car', linewidth=5, color='k')
# # point, = ax.plot([], [], 'ro', label='Target Point')
# ax.legend()
# plt.title('Simulated Path Following Car Model')

# # Setup the axes to be 20% larger than the reference path
# range_x = 1.2*(max(ref_position[:,0]) - min(ref_position[:,0])) # range of x
# range_y = 1.2*(max(ref_position[:,1]) - min(ref_position[:,1])) # range of y
# mean_x = (max(ref_position[:,0]) + min(ref_position[:,0])) / 2 # mean of x
# mean_y = (max(ref_position[:,1]) + min(ref_position[:,1])) / 2 # mean of y
# ax.set_xlim([mean_x - range_x/2, mean_x + range_x/2])
# ax.set_ylim([mean_y - range_y/2, mean_y + range_y/2])

# # Plot animation function
# def animate(i):
#     # point.set_data(sim_position[:,0][i], sim_position[:,1][i]) # plot target point
#     car_line.set_data([sim_position[i,0], sim_position[i,2]], 
#     	[sim_position[i,1], sim_position[i,3]]) # car

#     return car_line

# # Create animation
# ani = animation.FuncAnimation(fig, animate, N, interval=1)
# plt.show()

