import numpy as np
def foo(i):
    x = np.random.permutation(1000000)
    x.sort()
    print 'RUNED JOB! ', i