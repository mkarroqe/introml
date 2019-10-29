import numpy as np

# Some random data
nx = 100
ny = 10
p = 2
X = np.random.rand(nx,p)
Y = np.random.rand(ny,p)

# Computing the distances in two lines.  No for loop!
dists = x[:, None, :] - y[None, :, :]
result = np.sum(dists**2, axis = 2)

def loop(X, beta):
	n = X.shape[0]
	yhat = np.zeros(n)
	for i in range(n):
		yhat[i] = beta[0]*X[i,0] + beta[1]*X[i,1] + beta[2]*X[i,1]*X[i,2]

X = np.matrix('1 2 1; 3 4 3')
beta = np.matrix('1 2 3')
# print(loop(X, beta))

n = X.shape[0]
yhat = np.zeros(n)

yhat = beta[0]*X[:,0] + beta[1]*X[:,1] + beta[2]*X[:,1]*X[:,2]

print(yhat)

yhat = np.dot(X, beta)
print(yhat)

A = np.outer(-beta, x)
yhat = np.dot(np.transpose(alpha), np.exp(A))
# :- )


yhat = np.sum(alpha[:, None] * np.exp((-beta[:, None]) * x[None, :]), axis = 0)

