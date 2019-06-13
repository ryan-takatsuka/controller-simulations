import numpy as np

class Sensor:
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