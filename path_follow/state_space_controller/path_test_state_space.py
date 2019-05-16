import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import control
from CarModel import CarPose, Step, linearProjectForward
from matplotlib import animation

# References:
# http://control.ucsd.edu/mauricio/courses/mae280a/chapter5.pdf
# http://www.cds.caltech.edu/~murray/books/AM05/pdf/obc08-trajgen_24Jan08.pdf


# Set some constants
velocity = 1 # [m/s] The velocity of the car
L = 2 # [m] The length of the wheelbase
dt = 0.1 # [sec] The timestep
N = 500 # The number of points in the simulation


# Set up the state space model
# Initialize the state space matrices, A and B
A = np.zeros((2,2))
B = np.zeros((2,1))

# Set values in the matrices
A[0,1] = velocity
B[1,0] = velocity / L


# Calculate the controllability matrix
C0 = control.ctrb(A, B)
print("Controllability matrix rank =", np.linalg.matrix_rank(C0), "(should be 2)")

# Calculate the LQR parameters
Q = C0.T*C0
R = 1

# Calculate the controller gain, K
K, S, E = control.lqr(A, B, Q, R)
print("Controller gain =", K[0])


# Create reference path
initial_pose = CarPose(0, 0, 0) # initial Pose
ref_path = [initial_pose] # Initialize the list of reference paths
ref_steer_ang = np.linspace(0, 1, N) * 0.1 # Create the steering angle

# Create the reference path of N points
for i in range(N):
	ref_path.append(Step(ref_path[-1], velocity, ref_steer_ang[i], dt, L))


# -------- SIMULATION -------------
# The initial pose
initial_pose.theta = np.pi/2 # Adjust the starting orientation to 90 off from the reference path

def Simulate(initial_pose):
    # Initialize some variables
    control_steer_angle = np.zeros(N) # The controlled steering angle
    u_K_ar = np.zeros(N) # The controlled steering angle
    state_error = np.zeros((N,2)) # The error between the current state and the reference states
    sim_path = [initial_pose] # Initialize the starting simulation path
    
    # Iterate through the N points and calculate a controlled steering angle
    for i in range(N):
        # Define the current/recent pose
        current_pose = sim_path[-1]

        # Define the expected pose from the reference path
        current_ref_pose = ref_path[i]

        # Calculate the difference between the current states and the reference states (only y and theta!!)
        state_diff = current_pose.subtract(current_ref_pose).toStateVector()[1:]

        # Calculate the controller output (Note: this is only the controlled output for the error system, not the actual system)
        # The reference steering angle must be added to this to calculate the actual output
        u_K = np.dot(-K, state_diff)[0,0]
        control_steer_angle[i] = np.arctan(u_K) + ref_steer_ang[i]

        # Calculate the pose and add to the simulation list
        sim_path.append(Step(current_pose, velocity, control_steer_angle[i], dt, L))

        # Add the state error and controller output to an array for plotting
        state_error[i,:] = state_diff.T
       	u_K_ar[i] = u_K

        
    return sim_path, state_error, control_steer_angle, u_K_ar

sim_path, state_error, control_steer_angle, u_K_ar = Simulate(initial_pose)


# The simulation path plotting variables
sim_position = np.zeros((N+1,4))
for idx, pose in enumerate(sim_path):
	sim_position[idx,0] = pose.x
	sim_position[idx,1] = pose.y
	sim_position[idx,2], sim_position[idx,3] = linearProjectForward(pose, 1) 

# The reference path plotting variables
ref_position = np.zeros((N+1,2))
for idx, pose in enumerate(ref_path):
	ref_position[idx,0] = pose.x
	ref_position[idx,1] = pose.y


# Plot the reference path and simulated path
plt.figure()
plt.plot(ref_position[:,0], ref_position[:,1], label="reference")
plt.plot(sim_position[:,0], sim_position[:,1], label="simulation")
plt.title("Simulated Car Output")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()

# Plot the steering angle applied to the car
plt.figure()
plt.plot(control_steer_angle, label="Steering Angle")
plt.plot(u_K_ar, label="Direct Controller Output")
plt.plot(ref_steer_ang, label="Reference Steering Angle")
plt.legend()
plt.title("Control Variables")
plt.xlabel("Step Number")
plt.ylabel("Value")

# Plot the state errors
plt.figure()
plt.plot(state_error[:,0], label="Y error")
plt.plot(state_error[:,1], label="theta error")
plt.title("State errors")
plt.xlabel("Step Number")
plt.ylabel("State error")
plt.legend()

# Create the figure and axes
fig = plt.figure()
ax = plt.axes(xlim=(min(ref_position[:,0]), max(ref_position[:,0])), ylim=(min(ref_position[:,1]), max(ref_position[:,1])))

# Initialize plot items
reference_line, = ax.plot(ref_position[:,0], ref_position[:,1], label='Reference path')
car_line, = ax.plot([], [], label='Car', linewidth=5, color='k')
# point, = ax.plot([], [], 'ro', label='Target Point')
ax.legend()
plt.title('Simulated Path Following Car Model')

# Setup the axes to be 20% larger than the reference path
range_x = 1.2*(max(ref_position[:,0]) - min(ref_position[:,0])) # range of x
range_y = 1.2*(max(ref_position[:,1]) - min(ref_position[:,1])) # range of y
mean_x = (max(ref_position[:,0]) + min(ref_position[:,0])) / 2 # mean of x
mean_y = (max(ref_position[:,1]) + min(ref_position[:,1])) / 2 # mean of y
ax.set_xlim([mean_x - range_x/2, mean_x + range_x/2])
ax.set_ylim([mean_y - range_y/2, mean_y + range_y/2])

# Plot animation function
def animate(i):
    # point.set_data(sim_position[:,0][i], sim_position[:,1][i]) # plot target point
    car_line.set_data([sim_position[i,0], sim_position[i,2]], 
    	[sim_position[i,1], sim_position[i,3]]) # car

    return car_line

# Create animation
ani = animation.FuncAnimation(fig, animate, N, interval=1)
plt.show()

