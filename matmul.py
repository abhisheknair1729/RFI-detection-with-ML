import numpy as np
import matplotlib.pyplot as plt

p = np.load("../data/Parray.npy")
q = np.load("../data/Qarray.npy")
ro = np.load("../data/spec.npy")
r = np.matmul(p,q.T)
plt.imshow(r)
plt.show()
plt.imshow(ro)
plt.show()

