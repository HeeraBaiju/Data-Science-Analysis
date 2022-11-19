import numpy as np
from scipy import stats
vals = np.random.normal(loc=0,scale=1,size=1000)
stats.kstest(vals,'norm')
print(vals)
