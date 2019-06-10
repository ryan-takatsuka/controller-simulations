# Functions and classes used for the adaptive kalman filter example

# Import some stuff
import numpy as np
from scipy import signal
from filterpy.kalman import KalmanFilter
from filterpy import common

class myPath:
	def __init__(self, timestep):
		''' Initialize the path '''

		self.N = 0
		self.dt = timestep
		self.sample = []
		self.path = []


	def addLineSegment(self, n_seg, start, end):
		''' Add a straight line segment to the total path '''

		x = np.linspace(start, end, n_seg)
		self.path.extend(x)
		self.addSamples(n_seg)


	def addSineSegment(self, n_seg, amp, freq):
		''' Add a sine wave segment '''

		time = np.arange(0, n_seg*self.dt, self.dt)
		x = amp * np.sin(freq*2*np.pi*time)
		self.path.extend(x)
		self.addSamples(n_seg)


	def smoothData(self, b=0, a=0):
		''' Smooth the data using the specified filter parameters.
		If the filter parameters are 0, then a default butterworth
		filter is used.  This function removes any discontinuties
		in the data. '''

		if b==0 and a==0:
			b, a = signal.butter(2, 0.08, 'low') # lowpass filter design

		# Filter the data
		self.path = signal.lfilter(b, a, self.path)


	def addSamples(self, n_seg):
		''' Add to the list of samples '''

		try:
			s = np.arange(n_seg) + self.sample[-1] + 1
		except IndexError:
			s = np.arange(n_seg)

		self.sample.extend(s)


class mySensor:
	def __init__(self, var):
		''' Initialize the sensor '''
		self.var = var

	def readSingleValue(self, val):
		''' Read a single value from the sensor '''
		return val + np.random.randn()*self.var

	def readManyValues(self, vals):
		''' read multiple measurement with the sensor '''

		return vals + np.random.normal(0, np.sqrt(self.var), size=(len(vals),))

	def read_derivative(self, z, dt):
		''' take the numberical derivative of some measurements '''

		dz_dt = np.concatenate((np.array([0]), np.diff(z))) / dt
		return dz_dt


class standard_cvfilter:
	def __init__(self, r_var, q_var, dt):
		''' Initialize the constant velocity model kalman filter '''

		# Create the kalman filter and initialize the variables
		self.dt = dt
		self.filter = KalmanFilter(dim_x=2, dim_z=1)
		self.filter.F = np.array([[1, dt],
								  [0, 1]]) # The A matrix
		self.filter.H = np.array([[1, 0]]) # The C matrix
		self.filter.R *= r_var # The measurement noise covariance

		# Set the process noise
		self.filter.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=q_var)

	def filter_data(self, measurements):
		''' Filter the measurement data using this filter '''

		# Create the saver object to save all internal variables
		saver = common.Saver(self.filter)

		# Iterate through the measurement values
		for z in measurements:
			self.filter.predict()
			self.filter.update(z)
			saver.save()

		saver.to_array() # Convert all keys to arrays
		return saver


class adaptive_Q_cvfilter:
	def __init__(self, r_var, q_var, dt, eps_max, Q_scale_factor):
		''' An adaptive kalman filter that uses an adjustable Q value '''

		# Create the kalman filter and initialize variables
		self.dt = dt
		self.filter = KalmanFilter(dim_x=2, dim_z=1)
		self.filter.F = np.array([[1, dt],
								  [0, 1]]) # The A matrix
		self.filter.H = np.array([[1, 0]]) # The C matrix
		self.filter.R *= r_var # The measurement noise covariance

		# Set the process noise covariance
		self.filter.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=q_var)

		# Set the adaptive filter parameters
		self.eps_max = eps_max
		self.Q_scale_factor = Q_scale_factor

	def filter_data(self, measurements):
		''' Filter the measurement data '''

		# Create the saver object to save all the internal variables
		saver = common.Saver(self.filter)

		# Iterate through the measurement values
		count = 0
		for z in measurements:
			self.filter.predict()
			self.filter.update(z)
			
			# Calculate the normalized residual
			epss = np.dot(self.filter.y.T, 
				np.linalg.inv(self.filter.S)).dot(self.filter.y)

			# If the normalized residual is too large, adjust the process noise
			if epss > self.eps_max:
				self.filter.Q *= self.Q_scale_factor
				count += 1
			elif count > 0:
				self.filter.Q /= self.Q_scale_factor
				count -= 1
			saver.save()

		saver.to_array()
		return saver

class zarchan_adaptive_cvfilter:
	def __init__(self, r_var, q_var, dt, std_scale, Q_scale_factor):
		''' An adaptive kalman filter that uses an adjustable process
		covariance '''

		# Create the kalman fitler and initialize the variables
		self.dt = dt
		self.filter = KalmanFilter(dim_x=2, dim_z=1)
		self.filter.F = np.array([[1, dt],
								  [0, 1]]) # The A matrix
		self.filter.H = np.array([[1, 0]]) # The C matrix
		self.filter.R *= r_var # the measurement noise covariance

		# Initialize the process noise
		self.phi = q_var
		self.update_Q()

		# Set the adaptive filter parameters
		self.std_scale = std_scale
		self.Q_scale_factor = Q_scale_factor

	def update_Q(self):
		''' Update the process noise covariance '''

		self.filter.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=self.phi)

	def filter_data(self, measurements):
		''' Filter the measurement data '''

		# Create the saver object to save internal variables
		saver = common.Saver(self.filter)

		# Iterate through the measurements
		count = 0
		for z in measurements:
			self.filter.predict()
			self.filter.update(z)

			# Calculate the standard deviation of the noise
			std = np.sqrt(self.filter.S[0][0])

			# If the residual, update the process noise
			if np.abs(self.filter.y) > self.std_scale*std:
				self.phi += self.Q_scale_factor
				self.update_Q()
				count += 1
			elif count > 0:
				self.phi -= self.Q_scale_factor
				self.update_Q()
				count -= 1
			saver.save()

		saver.to_array()
		return saver


def saver2array(s_list):
	''' convert a list of 2d arrays to a single 2d array '''

	return np.squeeze(np.stack(s_list))


def calculate_rms(saver, dt):
	''' Calculate the rms values from the saver '''

	z = saver2array(saver.z)
	dz_dt = np.concatenate((np.array([0]), np.diff(z))) / dt


	return z

def calc_noise(filt, sensor):
	''' Calculate the rms noise using the specified sensor '''

	# Length of steady state values for noise calc
	L = 1000

	p = myPath(filt.dt)
	p.addSineSegment(200, 10, 1)
	p.addSineSegment(200, 10, 5)
	p.addLineSegment(1*filt.dt, 0, 0)
	p.addLineSegment(L, 0, 0)
	p.smoothData()
	z = sensor.readManyValues(p.path)
	dz_dt = sensor.read_derivative(z, filt.dt)

	# Filter the data
	saver = filt.filter_data(z)

	# Calculate the noise values
	rms_z = np.sqrt(np.mean(z[-L+100:]**2))
	rms_z_kal = np.sqrt(np.mean(saver.x[-L+100:,0]**2))

	rms_z_vel = np.sqrt(np.mean(dz_dt[-L:]**2))
	rms_z_vel_kal = np.sqrt(np.mean(saver.x[-L:,1]**2))

	print('RMS position measurement: ', rms_z)
	print('RMS position Kalman: ', rms_z_kal)
	print('RMS velocity measurement: ', rms_z_vel)
	print('RMS velocity Kalman: ', rms_z_vel_kal)

	return saver