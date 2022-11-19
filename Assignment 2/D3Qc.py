from scipy import stats
import numpy as np


np.random.seed(10)
N = 50
L0 = 10
dL = 0.2
t = np.linspace(0, 1, N)
L_obs = np.random.normal(L0, dL, N)
y_vals = [L_obs, L_obs, L_obs, L_obs + 0.5 - t ** 2]
y_errs = [dL, dL * 2, dL / 2, dL]
titles = ['correct errors', 'overestimated errors', 'underestimated errors', 
'incorrect model']
for i in range(4):

    mu = np.mean(y_vals[i])
    z = (y_vals[i] - mu) / y_errs[i]
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (N - 1)

    pvalue = stats.chi2(N-1).sf(chi2dof*chi2)
    print(" The p-value for the chi-square values of {} is: {}" .format(titles[i], pvalue))