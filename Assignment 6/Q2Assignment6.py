import emcee
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('C:\\Users\\Heera Baiju\\Desktop\\Q2data.txt')
xdata = data[:,1]
ydata = data[:,2]
error = data[:,3] 

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
    return -0.5 * np.sum(np.log(2 * np.pi * error ** 2) + (y - y_model) ** 2 / error **
2)


def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)
    
ndim = 2 
nwalkers = 50 
nburn = 1000 
nsteps = 2000 

np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args = [xdata,
ydata])
sampler.run_mcmc(starting_guesses, nsteps)
emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T
plot_MCMC_results(emcee_trace)
plt.show()