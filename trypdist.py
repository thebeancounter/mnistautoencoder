from scipy.spatial.distance import pdist
import scipy

import numpy as np
a = np.array([[1],[4],[0],[5]])
#print a
#print pdist(a)

print scipy.spatial.distance.squareform(pdist(a))