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
	def __init__(self, std):
		self.std = std

	def readSingleValue(self, val):
		return val + np.random.randn()*self.std

	def readManyValues(self, vals):
		''' vals is a list of real data to read with
		this sensor. '''

		return vals + np.random.normal(0, self.std, size=(len(vals),))


class constantVelocityFilter:
	def __init__(self, r_std, q_std, dt):
		''' Initialize the constant velocity model kalman filter '''

		# Create the kalman filter and initialize variables
		self.dt = dt
		self.f = KalmanFilter(dim_x=2, dim_z=1)
		self.f.F = np.array([[1, dt],
							 [0, 1]])
		self.f.H = np.array([[1, 0]])
		self.f.R *= r_std**2
		self.f.Q = common.Q_discrete_white_noise(dim=2, dt=dt, var=q_std**2)


	def standard_filter(self, measurements):
		''' Use the standard Kalman filter '''

		x = []
		saver = common.Saver(self.f)
		for z in measurements:
			self.f.predict()
			self.f.update(z)
			x.append([self.f.x[0][0], self.f.x[1][0]])
			saver.save()

		return saver


	def adaptive_Q_filter(self, measurements, eps_max, Q_scale_factor, q_var=0.1):
		''' An adaptive kalman filter that uses an adjustable Q value '''

		# Initialize the process noise again
		self.f.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=q_var)

		# Run the filter
		x = []
		Q_list = []
		count = 0
		for z in measurements:
			self.f.predict()
			self.f.update(z)
			epss = np.dot(self.f.y.T, np.linalg.inv(self.f.S)).dot(self.f.y)
			x.append([self.f.x[0][0], self.f.x[1][0]])

			if epss > eps_max:
				self.f.Q *= Q_scale_factor
				count += 1
			elif count > 0:
				self.f.Q /= Q_scale_factor
				count -= 1
			Q_list.append(np.max(self.f.Q))

		return np.asarray(x), Q_list


	def zarchan_adaptive_filter(self, measurements, std_scale, Q_scale_factor, phi0=1):
		''' An adaptive kalman filter that uses an adjustable Q value '''

		# Initialize the process noise again
		phi = phi0
		self.f.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=phi)

		# Run the filter
		x = []
		Q_list = []
		count = 0
		std = []
		saver = common.Saver(self.f)
		for z in measurements:
			self.f.predict()
			self.f.update(z)
			std.append(np.sqrt(self.f.S[0]))
			x.append([self.f.x[0][0], self.f.x[1][0]])

			if np.abs(self.f.y) > std_scale*std[-1]:
				phi += Q_scale_factor
				self.f.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=phi)
				count += 1
			elif count > 0:
				phi -= Q_scale_factor
				self.f.Q = common.Q_discrete_white_noise(dim=2, dt=self.dt, var=phi)
				count -= 1
			Q_list.append(np.max(self.f.Q))

		return np.asarray(x), Q_list, std		