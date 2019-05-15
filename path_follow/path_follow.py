import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants from the dynamic model
L = 2 # [m] wheelbase
velocity = 1 # [m/s] velocity of the car
dt = 0.1 # [sec] Time step
N = 300 # Number of points
PROJ_DISTANCE = 0.5 # [m] projection distance for target point
MIN_DIST_MAX = 0.1 # [m] minimum distance before steering correction kicks in

class CarPose(object):
	''' Create the CarPose class that contains state variables
	for the car at a specific pose
	'''
	def __init__(self, x, y, theta, x_dot=0, y_dot=0, theta_dot=0):
		''' Initialize the CarPose object

		Args:
			x (scalar): [m] The x-position of the car
			y (scalar): [m] The y-positino of the car
			theta (scalar): [rad] The orientation of the car
			x_dot (scalar): [m/s] (0) The x velocity component
			y_dot (scalar): [m/s] (0) The y velocity component
			theta_dot (scalar): [rad/s] (0) The angular velocity

		Returns:
			The CarPose object

		'''
		self.x = x # [m] x-position
		self.y = y # [m] y-position
		self.theta = theta # [rad] orientation angle
		self.x_dot = x_dot # [m/s] x-velocity
		self.y_dot = y_dot # [m/s] y-velocity
		self.theta_dot = theta_dot # [rad/s] angular velocity


def linearProjectForward(pose, dist):
	''' Project a point forward a specific distance (linearly)
	
	Args:
		pose (CarPose): The CarPose for the current pose
		dist (scalar): The distance to project a point forward

	Returns:
		x,y - A linearly projected point (x,y) away from the input pose

	'''

	# Calculate the projected pose
	x = pose.x + dist*np.cos(pose.theta) # project the x component with heading, theta
	y = pose.y + dist*np.sin(pose.theta) # project the y component with heading, theta

	return x, y


def Step(last_pose, steering_angle, velocity, dt):
	''' Calculate the next pose of the car given the specified input steering angle and velocity

	Args:
		last_pose (CarPose): The previous/current pose of the car
		steering_angle (scalar): [rad] The steering angle of the car
		velocity (scalar): [rad] The total velocity of the car
		dt (scalar): [s] The time step
	'''

	# Calculate the derivatives
	x_dot = velocity * np.cos(last_pose.theta) # [m/s] x velocity
	y_dot = velocity * np.sin(last_pose.theta) # [m/s] y velocity
	theta_dot = velocity / L * np.tan(steering_angle) # [rad/s] angular velocity

	# Calculate the next position and orientation using the slope estimation
	x = x_dot * dt + last_pose.x # [m] new x-position
	y = y_dot * dt + last_pose.y # [m] new y-position
	theta = theta_dot * dt + last_pose.theta # [rad] new orientation

	return CarPose(x, y, theta, x_dot, y_dot, theta_dot)


def Simulate(initial_pose, reference_path, velocity, dt, N):
	''' Simulate a controller that causes a car to follow a reference path

	Args:
		initial_pose (CarPose): The initial starting pose of the car
		reference_path (list): The list of CarPose values for the reference path to follow
		velocity (scalar): [m/s] The velocity (assumed to be constant)
		dt (scalar): [s] The time step
		N (scalar): The number of points to iterate across

	Returns:
		The simulated car path and some variables for plotting
	'''

	# Create variables for the pose components in the reference path
	x_ref = np.zeros(len(reference_path))
	y_ref = np.zeros(len(reference_path))
	theta_ref = np.zeros(len(reference_path))
	for idx, pose in enumerate(reference_path):
		x_ref[idx] = pose.x
		y_ref[idx] = pose.y
		theta_ref[idx] = pose.theta

	# Initialize some lists used for plotting the results
	x_predict_list = [] # The projected point (x) from the current pose
	y_predict_list = [] # The projected point (y) from the current pose
	x_target_list = [] # The target point on the reference path (x)
	y_target_list = [] # The target point on the reference path (y)

	# Simulate the path for the car with controller inputs
	sim_path = [initial_pose]
	for i in range(N): # iterate through all the points in the path
		current_pose = sim_path[-1] # Set the current/last pose
		x_predict, y_predict = linearProjectForward(current_pose, PROJ_DISTANCE) # project a point forward

		# Calculate the minimum distance from the projected point to the reference path
		distance_to_path = np.min(np.sqrt((x_ref-x_predict)**2 + (y_ref-y_predict)**2)) # distance
		dist_index = np.argmin(np.sqrt((x_ref-x_predict)**2 + (y_ref-y_predict)**2)) # corresponding index

		# Calculate the target distance.  This is a point on the reference path that is used to
		# adjust the steering angle.  This controller adjusts the steering angle to guide the current
		# pose towards the target pose.  This is set some PROJ_DISTANCE ahead of the current pose to
		# prevent overcorrecting the steering angle.
		target_distance = 0 # initialize the target distance
		target_index = dist_index # initialize the target index with the minimum distance index
		while target_distance<PROJ_DISTANCE: # keep iterating until the target distance is a sufficient distance away
			target_index = target_index + 1
			try: # If this is near the end of the path, the incremented index might not exist
				target_distance = np.sqrt((x_ref[dist_index]-x_ref[target_index])**2 + 
					(y_ref[dist_index]-y_ref[target_index])**2)
			except: # if it doesn't exist, deincrement the index and break
				target_index = target_index-1
				break

		# Calculate the steering angle as the difference between the target orientation
		# and the current orientation
		steering_angle = theta_ref[target_index] - current_pose.theta

		# Calculate a correction to the steering angle if the distance from the path becomes too large
		y0 = y_ref[target_index] - current_pose.y # Calculate y difference
		x0 = x_ref[target_index] - current_pose.x # Calculate x difference
		steer_correct = 0 # initialize the steer correction term
		if distance_to_path>MIN_DIST_MAX: # if the error is too large
			steer_correct = current_pose.theta - np.arctan2(y0, x0) # Calculate the correction
			steering_angle = steering_angle - steer_correct # adjust the steering angle

		# Use the calculated steering angle to add the next pose to the list 
		sim_path.append(Step(current_pose, steering_angle, velocity, dt))

		# Add the target point to the output list
		x_target_list.append(x_ref[target_index])
		y_target_list.append(y_ref[target_index])
		
		# Add the projection point to the output list
		x_predict_list.append(x_predict)
		y_predict_list.append(y_predict)
		
		# Create a list of the plotting variables
		plot_vars = [x_target_list, y_target_list, x_predict_list, y_predict_list]

	return sim_path, plot_vars


# Create the reference path
initial_pose = CarPose(0, 0, np.pi/2) # initial pose
reference_path = [initial_pose]
ref_steer = np.linspace(0, 1, N) # variable steering angle for the reference path
for i in range(N):
	# Create the reference path with constant velocity
	reference_path.append(Step(reference_path[-1], ref_steer[i], velocity, dt))

# Simulate the controller and car path
sim_path, plot_vars = Simulate(initial_pose, reference_path, velocity, dt, N)

# Create variables for plotting
x_ref = []
y_ref = []
x_sim = []
y_sim = []
for pose in reference_path:
	x_ref.append(pose.x)
	y_ref.append(pose.y)
for pose in sim_path:
	x_sim.append(pose.x)
	y_sim.append(pose.y)

# Create the figure and axes
fig = plt.figure()
ax = plt.axes(xlim=(min(x_ref), max(x_ref)), ylim=(min(y_ref), max(y_ref)))

# Initialize plot items
reference_line, = ax.plot(x_ref, y_ref, label='Reference path')
car_line, = ax.plot([], [], label='Car', linewidth=5, color='k')
target_point, = ax.plot([], [], 'ro', label='Target Point')
ax.legend()
plt.title('Simulated Path Following Car Model')

# Setup the axes to be 20% larger than the reference path
range_x = 1.2*(max(x_ref) - min(x_ref)) # range of x
range_y = 1.2*(max(y_ref) - min(y_ref)) # range of y
mean_x = (max(x_ref) + min(x_ref)) / 2 # mean of x
mean_y = (max(y_ref) + min(y_ref)) / 2 # mean of y
ax.set_xlim([mean_x - range_x/2, mean_x + range_x/2])
ax.set_ylim([mean_y - range_y/2, mean_y + range_y/2])

# Plot animation function
def animate(i):
    target_point.set_data(plot_vars[0][i], plot_vars[1][i]) # plot target point
    car_line.set_data([x_sim[i], plot_vars[2][i]], [y_sim[i], plot_vars[3][i]]) # car

    return car_line, target_point

# Create animation
ani = animation.FuncAnimation(fig, animate, N, interval=30)
plt.show()