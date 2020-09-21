import numpy as np


X = np.array([4,3])

#L0Norm:
#l0norm=2
l0norm = np.linalg.norm(X, ord=0)
print(l0norm)

#L1Norm: Mahattan distance
l1norm = np.linalg.norm(X, ord=1)
print(l1norm)

#L2Norm: Euclid Distance
l2norm = np.linalg.norm(X, ord=2)
print(l2norm)
