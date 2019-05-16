import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# http://control.ucsd.edu/mauricio/courses/mae280a/chapter5.pdf
# http://www.cds.caltech.edu/~murray/books/AM05/pdf/obc08-trajgen_24Jan08.pdf

class CarPose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta


def Step(last_pose, velocity, phi, dt):

	x_dot = velocity * np.cos(last_pose.theta)
	y_dot = velocity * np.sin(last_pose.theta)
	theta_dot = velocity / L * np.tan(phi)

	x = x_dot * dt + last_pose.x
	y = y_dot * dt + last_pose.y
	theta = theta_dot * dt + last_pose.theta

	return CarPose(x, y, theta)


# Initialize variables
A = np.zeros((3,3))
B = np.zeros((3,1))
C = np.zeros((3,3))
D = np.zeros((3,1))

theta = 0
velocity = 1
L = 2
dt = 0.1
N = 200

# Set values
A[0,2] = -velocity*np.sin(theta)
A[1,2] = velocity*np.cos(theta)

B[2,0] = velocity / L

C = np.eye(3)

sys = signal.StateSpace(A, B, C, D, dt=dt)

print(sys)


t = np.linspace(0, 1, N)
u = np.tan(0) * np.ones(N)

t_out, y_out, x_out = signal.dlsim(sys, u)

print(x_out)

# plt.figure()
# plt.plot(t, y_out[:,0])
# # plt.show()

# Create reference path
initial_pose = CarPose(0, 0, 0)
ref_path = [initial_pose]
u_d = np.ones(N) * 0.2
for i in range(N):
	ref_path.append(Step(ref_path[-1], velocity, u_d[i], dt))


ref_position = np.zeros((N+1,2))
for idx, pose in enumerate(ref_path):
	ref_position[idx,0] = pose.x
	ref_position[idx,1] = pose.y



u = np.zeros(N)
sim_path = [initial_pose]
error = 0.1
for i in range(N):
	u[i] = u_d[i]
	current_pose = sim_path[-1]

	current_pose.x = current_pose.x + error
	sim_path.append(Step(current_pose, velocity, u[i], dt))


sim_position = np.zeros((N+1,2))
for idx, pose in enumerate(sim_path):
	sim_position[idx,0] = pose.x
	sim_position[idx,1] = pose.y



plt.figure()
plt.plot(ref_position[:,0], ref_position[:,1])
plt.plot(sim_position[:,0], sim_position[:,1])
plt.show()