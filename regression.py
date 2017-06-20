import math
import numpy as np
import matplotlib.pyplot as plt

def regress(X, y, transform=lambda x: x, V=1):
	# define error epsilon as a normal distribution with mu=0, sigma=V
	# The transform is a mapping from one feature space to another to be applied to each x_i 
	N, D = X.shape

	for n in range(N):
		X[n, :] = transform(X[n, :])

	XtX_inv = np.linalg.inv(np.dot(X.T, X))
	XtY = np.dot(X.T, y)
	beta = np.dot(XtX_inv, XtY)
	return beta



def apply(funct, input):
	N, D = input.shape
	result = np.zeros(N)
	for n in range(N):	
		x = funct(input[n, :])

		if D > 1:
			result[n] = np.dot(x, np.ones(D)) 
		else:	
			result[n] = x
	
	return result



if __name__ == "__main__":
	X = np.random.randn(100, 1)	
	f = lambda x: x**4 + 3*(x**2) + 4*x + 1
	feature_map = lambda x: np.array([x, x**2])
	Y = apply(f, X)
	beta = regress(X, Y, f)
	print (beta)

	N, D = X.shape	
	loss = 0.

	for n in range(N):
		estimate = np.dot(beta, X[n, :]) 
		y_n = Y[n]
		err = abs(y_n - estimate)
		loss += err
	
	loss = loss / N
	
	print("Loss: " + str(loss))

	convert = lambda x: np.dot(feature_map(x), beta)	
	results = apply(convert, X)
	
	print(results.shape)

	plt.plot(X, Y, 'o', label='Original data', markersize=10)
	plt.plot(X, results,  'r', label='Fitted line')

	plt.legend()
	plt.show()


