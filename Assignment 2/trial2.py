import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize, stats
import csv
global data, x, y, sigma_y, val1, val2, val3 

data = np.array([[ 0.42,  0.72,  0.  ,  0.3 ,  0.15,
                   0.09,  0.19,  0.35,  0.4 ,  0.54,
                   0.42,  0.69,  0.2 ,  0.88,  0.03,
                   0.67,  0.42,  0.56,  0.14,  0.2  ],
                 [ 0.33,  0.41, -0.22,  0.01, -0.05,
                  -0.05, -0.12,  0.26,  0.29,  0.39, 
                   0.31,  0.42, -0.01,  0.58, -0.2 ,
                   0.52,  0.15,  0.32, -0.13, -0.09 ],
                 [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1  ]])
x, y, sigma_y = data


#initialise theta for all the fits to zero
val1 = np.array([0, 0])
val2 = np.array([0, 0, 0])

#Defining the required functions for fitting
def linearFunc(x, val1):
    return val1[1]*x+val1[0]

def quadraticFunc(x, val2):
    return val2[2]*x**2 + val2[1]*x + val2[0]

def logL(theta, n):
    if n==1:
        y_fit = linearFunc(x, theta)
    elif n==2:
        y_fit = quadraticFunc(x, theta)

    return sum(stats.norm.logpdf(*args)
               for args in zip(y, y_fit, sigma_y))

              
               #This function returns the best theta for the fitting
def best_theta(n, theta_val):
    if n==1:
        theta_0 = (n+1)*[0]
        neg_logL = lambda theta: -logL(theta, 1)
        return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)
    if n==2:
        theta_0 = (n+1)*[0]
        neg_logL = lambda theta: -logL(theta, 2)
        return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)

   
#compute chi2 likelihood for frequentist
def compute_chi2(n):
    if n==1:
        theta = best_theta(n, val1)
        resid = ((y - linearFunc(x, theta)) / sigma_y)
    elif n==2:
        theta = best_theta(n, val2)
        resid = ((y - quadraticFunc(x, theta)) / sigma_y)

r1 = best_theta(1, val1)
r2 = best_theta(2, val2)
print("R1", r1)
print("R2", r2)

print("Log L values")
print("linear model:    logL =", logL(best_theta(1, val1), 1))
print("quadratic model: logL =", logL(best_theta(2, val2), 2))

"""Bayesian Analysis"""
#Compute the AICc values as number of data points is considerably small
AIC1 = -2*logL(r1, 1) + (2.0*2*20)/(17.0)
AIC2 = -2*logL(r2, 2) + (2.0*3*20)/(16.0)

#Compute the BIC values
BIC1 = -2*logL(r1, 1) + 2*np.log(x.shape[0])
BIC2 = -2*logL(r2, 2) + 3*np.log(x.shape[0])

print("AICc values")
print("- linear model:    ", AIC1)
print("- quadratic model: ", AIC2)

print("BIC vlaues")
print("- linear model:    ", BIC1)
print("- quadratic model: ", BIC2)

"""Plotting"""
t =  np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(t, linearFunc(t, r1), label='linear_fitting')
plt.plot(t, quadraticFunc(t, r2), label='quadratic_fitting')

plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("Curve fitting using Linear and Quadratic", size=15)
ax.errorbar(x, y, sigma_y, fmt='ok', ecolor = 'gray')
plt.show()

def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""
    return sum(t * x ** n for (n, t) in enumerate(theta))

def compute_chi2(degree, data=data):
    x, y, sigma_y = data
    theta = best_theta(degree, data=data)
    resid = (y - polynomial_fit(theta, x)) / sigma_y
    return np.sum(resid ** 2)

def compute_dof(degree, data=data):
    return data.shape[1] - (degree + 1)

def chi2_likelihood(degree, data=data):
    chi2 = compute_chi2(degree, data)
    dof = compute_dof(degree, data)
    return stats.chi2(dof).pdf(chi2)

print("chi2 likelihood")
print("- linear model:    ", chi2_likelihood(1))
print("- quadratic model: ", chi2_likelihood(2))

fig, ax = plt.subplots()
for degree, color in zip([1, 2], ['blue', 'green']):
    v = np.linspace(0, 40, 1000)
    chi2_dist = stats.chi2(compute_dof(degree)).pdf(v)
    chi2_val = compute_chi2(degree)
    chi2_like = chi2_likelihood(degree)
    ax.fill(v, chi2_dist, alpha=0.3, color=color,
            label='Model {0} (degree = {0})'.format(degree))
    ax.vlines(chi2_val, 0, chi2_like, color=color, alpha=0.6)
    ax.hlines(chi2_like, 0, chi2_val, color=color, alpha=0.6)
    ax.set(ylabel='L(chi-square)')
ax.set_xlabel('chi-square')
ax.legend(fontsize=14);