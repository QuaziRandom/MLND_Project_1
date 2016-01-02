import numpy as np
import pylab as pl
from sklearn import datasets

a = np.array(((1, 2, 3),(4, 5, 6)), dtype=np.float32)
print "Array =", a
print "Mean =", a.mean()
print "Std =", a.std()
boston = datasets.load_boston()
