import matplotlib.pyplot as plt 
from scipy import optimize
import numpy as np
import emcee
plt.ion()

data = np.loadtxt('q3_data.txt')
xdata = data[:,1]
ydata = data[:,2]
e = data[:,3] # error in y 


def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)
    shape = L.shape
    L = L.ravel()
    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)

def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel('m')
    ax.set_ylabel('b')
    
    
# def plot_MCMC_model(ax, xdata, ydata, trace):
#     """Plot the linear model and 2sigma contours"""
#     ax.plot(xdata, ydata, 'ok')

#     alpha, beta = trace[:2]
#     xfit = np.linspace(0, 300, 10)
#     yfit = alpha[:, None] + beta[:, None] * xfit
#     mu = yfit.mean(0)
#     sig = 2 * yfit.std(0)

#     ax.plot(xfit, mu, '-k')
#     ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.title(' 68 and 95 percent joint confidence intervals on b and m')
    plot_MCMC_trace(ax, xdata, ydata, trace, True, colors=colors)
    # plot_MCMC_model(ax[1], xdata, ydata, trace)


def log_prior(theta):
    alpha, beta = theta
    return -1.5 * np.log(1 + beta ** 2) 

def log_likelihood(theta, x, y):
    alpha, beta = theta
    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 * np.pi * e ** 2) + (y - y_model) ** 2 / e ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)


ndim = 2  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 1000  # "burn-in" period to let chains stabilize
nsteps = 2000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata, ydata])
sampler.run_mcmc(starting_guesses, nsteps)

emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
plot_MCMC_results(xdata, ydata, emcee_trace)