import numpy as np


def simulate(A, B, C, D, K, u, time, x0=None, dt=1):
	''' simulate the closed_loop response '''

	dt0 = np.mean(np.diff(time))

	if x0 is None:
		x0 = np.zeros(A.shape[0])

	xout = [x0]
	tout = [time[0]]
	uout = [np.zeros(D.shape[1])]
	x = x0
	for idx, t in enumerate(time[1:]):
		x_dot = A @ x + B@(-K@xout[-1])
		x = x + x_dot * (t-time[idx])

		if (t-tout[-1])>=dt:
			uout.append(-K@xout[-1])
			xout.append(x)
			tout.append(t)

	xout = np.asarray(xout)
	uout = np.asarray(uout)

	return tout, xout, uout