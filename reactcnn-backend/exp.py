import numpy as np
l = [np.array([1,2,3]), np.array([4,5,6])]
la = np.asarray(l)
np.savetxt('out.csv', la, fmt='%.5f', delimiter=',')