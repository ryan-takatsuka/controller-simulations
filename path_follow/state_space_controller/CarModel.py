import numpy as np

class CarPose(object):
	''' Create the CarPose class that contains state variables
	for the car at a specific pose
	'''
	def __init__(self, x, y, theta):
		''' Initialize the CarPose object

		Args:
			x (scalar): [m] The x-position of the car
			y (scalar): [m] The y-positino of the car
			theta (scalar): [rad] The orientation of the car

		Returns:
			The CarPose object
		'''

		self.x = x # [m] x-position
		self.y = y # [m] y-position
		self.theta = theta # [rad] orientation angle

	def subtract(self, pose):
		''' subtract another pose from this pose object '''

		x_diff = self.x - pose.x
		y_diff = self.y - pose.y
		theta_diff = self.theta - pose.theta

		return CarPose(x_diff, y_diff, theta_diff)

	def toStateVector(self):
		''' convert to a state vector '''

		state_vec = np.zeros((3,1))
		state_vec[0] = self.x
		state_vec[1] = self.y
		state_vec[2] = self.theta

		return state_vec

	def calcCurvature(self, steering_angle, velocity, L):
		''' calculate the steering angle '''

		curvature = np.tan(steering_angle) / L

		return curvature
		

def Step(last_pose, velocity, phi, dt, L):

	x_dot = velocity * np.cos(last_pose.theta)
	y_dot = velocity * np.sin(last_pose.theta)
	theta_dot = velocity / L * np.tan(phi)

	x = x_dot * dt + last_pose.x
	y = y_dot * dt + last_pose.y
	theta = theta_dot * dt + last_pose.theta

	return CarPose(x, y, theta)


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