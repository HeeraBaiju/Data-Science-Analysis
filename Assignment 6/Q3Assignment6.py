import emcee
import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Heera Baiju\\Desktop\\Q3data.csv")
x = data['x'].values
y = data['y'].values
e = data['sigma_y'].values
xfit = np.linspace(0, 300, 1000)
t = np.linspace(-20, 20)

def squared_loss(theta, x = x, y = y, e = e):
    dy = y - theta[0] - theta[1] * x
    return np.sum(0.5 * (dy / e) ** 2)

theta1 = optimize.fmin(squared_loss, [0, 0], disp = False)

def huber_loss(t, c = 3):
    return ((abs(t) < c) * 0.5 * t ** 2 + (abs(t) >= c) * -c * (0.5 * c -
abs(t)))

def total_huber_loss(theta, x = x, y = y, e = e, c = 3):
    return huber_loss((y - theta[0] - theta[1] * x) / e, c).sum()
theta2 = optimize.fmin(total_huber_loss, [0, 0], disp = False)
plt.errorbar(x, y, e, fmt = '.k', ecolor = 'gray')
plt.plot(xfit, theta1[0] + theta1[1] * xfit, color = 'lightgray')
plt.plot(xfit, theta2[0] + theta2[1] * xfit, color = 'black')
plt.title('Maximum Likelihood fit: Huber loss')
plt.show()

def log_prior(theta):

    if (all(theta[2:] > 0) and all(theta[2:] < 1)):
        return 0
    else:
        return -np.inf 

def log_likelihood(theta, x, y, e, sigma_B):
    dy = y - theta[0] - theta[1] * x
    g = np.clip(theta[2:], 0, 1) 
    logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
    logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy /
sigma_B) ** 2
    return np.sum(np.logaddexp(logL1, logL2))

def log_posterior(theta, x, y, e, sigma_B):
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)
ndim = 2 + len(x)
nwalkers = 50 
nburn = 10000 
nsteps = 15000 
np.random.seed(4)
starting_guesses = np.zeros((nwalkers, ndim))
starting_guesses[:, :2] = np.random.normal(theta2, 1, (nwalkers, 2))
starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x, y, e,
50])
sampler.run_mcmc(starting_guesses, nsteps)
sample = sampler.chain # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
theta3 = np.mean(sample[:, :2], 0)
g = np.mean(sample[:, 2:], 0)
outliers = (g < 0.38)
# Plotting
plt.errorbar(x, y, e, fmt = '.k', ecolor = 'gray')
plt.plot(xfit, theta1[0] + theta1[1] * xfit, color = 'lightgray')
plt.plot(xfit, theta2[0] + theta2[1] * xfit, color = 'lightgray')
plt.plot(xfit, theta3[0] + theta3[1] * xfit, color = 'black')
plt.plot(x[outliers], y[outliers], 'ro', ms=20, mfc = 'none', mec='red')
plt.title('Maximum Likelihood fit: Bayesian Marginalization')
plt.show()