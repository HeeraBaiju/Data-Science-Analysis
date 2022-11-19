import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt

gasdata = pd.read_csv("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignments\\Assignment 7\\SPT.csv")
xd = gasdata['#z'].values
yd = gasdata['fgas'].values
e = gasdata['fgas_error'].values

def compute_sigma_level(trace1, trace2, nbins = 20):

    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    shape = L.shape
    L = L.ravel()

    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)
    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])
    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

def plot_MCMC_trace(ax, trace, scatter = False, **kwargs):

    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels = [0.683, 0.955], **kwargs)
    if scatter:
       ax.plot(trace[0], trace[1], ',k', alpha = 0.1)
    ax.set_xlabel('m')
    ax.set_ylabel('b')

def plot_MCMC_results(trace, colors = 'k'):

    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    plt.title('68% and 95% joint confidence intervals on b and m')
    plot_MCMC_trace(ax, trace, True, colors = colors)

def log_prior(theta):
    beta = theta
    return -1.5 * np.log(1 + beta ** 2) 

def log_likelihood(theta, x, y):
    alpha, beta = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * e ** 2) + (y - y_model) ** 2 / e ** 
2)

def log_post(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)
ndim = 2 
nwalkers = 50 
nburn = 1000 
nsteps = 2000 

np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args = [xd, 
yd])
sampler.run_mcmc(starting_guesses, nsteps)
emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
plot_MCMC_results(emcee_trace)
plt.show()