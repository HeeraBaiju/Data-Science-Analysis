import numpy as np
from scipy.stats import norm
from astroML.resample import bootstrap
from astroML.stats import median_sigmaG
from matplotlib import pyplot as plt


points = 1000 
bootstraps = 10000 

np.random.seed(10)
gdsamples = norm(0, 1).rvs(points)

median, sigmaG = bootstrap(gdsamples, bootstraps, median_sigmaG, kwargs=dict(axis=1))

x = np.linspace(-2, 2, 1000)
sd = np.sqrt(np.pi/(2*points))
mean = np.mean(median)
pdf = norm(mean, sd).pdf(x)

fig, ax = plt.subplots(figsize=(5, 3.75))
ax.hist(median, bins = 20, density = True, histtype = 'step', color = 'orange', 
label = r'$\sigma\ {\rm (median)}$')
ax.plot(x, pdf, color = 'blue', label = '$Gaussian\ Distribution$')
ax.set_xlim(-0.5, 0.5)
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$p(\sigma|x,I)$')
ax.legend()
plt.show()