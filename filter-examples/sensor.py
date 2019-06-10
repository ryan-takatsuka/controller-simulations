import numpy as np
from scipy import signal

class Sensor:
	def __init__(self, var, rate):
		''' Initialize a sensor '''
		self.var = var
		self.rate = rate

	def readSingleValue(self, val):
		''' Read a single value from the sensor '''
		return val + np.random.randn()*self.var

	def read(self, vals, time):
		''' Read a batch set of data '''

		new_time = np.arange(0, time[-1], 1/self.rate)
		new_vals = np.zeros(len(new_time))
		self.real_path = np.zeros(len(new_time))
		count = 0
		for idx in range(len(time)):
			try:
				if new_time[count] <= time[idx+1]:
					new_vals[count] = self.readSingleValue(vals[idx])
					self.real_path[count] = vals[idx]
					count += 1
			except IndexError:
				pass

		self.measurements = new_vals
		self.time = new_time
		return new_time, new_vals


	def add_distortion(self, range_start, range_end):
		''' Add distortion to the sensor measurements '''

		num_samples = len(self.time)
		start_index = np.round(range_start * num_samples)
		end_index = np.round(range_end * num_samples)
		self.measurements = self.real_path

		for idx in range(num_samples):
			if idx >= start_index and idx <= end_index:
				self.measurements[idx] *= 2

		self.measurements = self.smooth_data()

		for idx in range(num_samples):
			self.measurements[idx] = self.readSingleValue(self.measurements[idx])

	def smooth_data(self, b=0, a=0):
		''' Filter to prevent discontinuities '''

		if b==0 and a==0:
			b, a = signal.butter(2, 0.08, 'low') # lowpass filter design

		# Filter the data
		self.measurements = signal.lfilter(b, a, self.measurements)


if __name__ == "__main__":
	sensor1 = Sensor(1, 3)

	time = np.linspace(0,1.1,10)
	vals = np.sin(2*np.pi*time)

	sensor1_time, sensor1_z = sensor1.read(vals, time)

