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

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x	

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = np.arctan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = np.arctan2(sum_sin, sum_cos)
    return x    