import numpy as np
import nestle

data_x = np.array([1., 2., 3.])
data_y = np.array([1.4, 1.7, 4.1])
data_yerr = np.array([0.2, 0.15, 0.2])

# Define a likelihood function
def loglike(theta):
    y = theta[1] * data_x + theta[0]
    chisq = np.sum(((data_y - y) / data_yerr)**2)
    return -chisq / 2.

# Define a function mapping the unit cube to the prior space.
# This function defines a flat prior in [-5., 5.) in both dimensions.
def prior_transform(x):
    return 10.0 * x - 5.0

# Run nested sampling.
result = nestle.sample(loglike, prior_transform, 2)

result.logz     # log evidence
result.logzerr  # numerical (sampling) error on logz
result.samples  # array of sample parameters
result.weights  # array of weights associated with each sample