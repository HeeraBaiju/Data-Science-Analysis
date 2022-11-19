import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
global data, x, y, sigma_y


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
x,y,sigma_y = data

def p_fit(theta, x):
    

    return sum(a * x ** n for (n, a) in enumerate(theta))


def logL(theta): 

    y_fit = p_fit(theta, x)
    return sum(stats.norm.logpdf(*args)
for args in zip(y, y_fit, sigma_y))

def best_theta(deg):
    theta_0 = (deg + 1) * [0]
    neg_logL = lambda theta: -logL(theta)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)
theta1 = best_theta(1)
theta2 = best_theta(2)

AIC1 = -2*logL(theta1) + (2.0*2*20)/(17.0)
AIC2 = -2*logL(theta2) + (2.0*3*20)/(16.0)

BIC1 = -2*logL(theta1) + 2*np.log(x.shape[0])
BIC2 = -2*logL(theta2) + 3*np.log(x.shape[0])

print("AIC values")
print(" Linear : ", AIC1)
print(" Quadratic : ", AIC2)

print("BIC values")
print(" Linear : ", BIC1)
print(" Quadratic : ", BIC2)

def compute_chi2(deg, data = data):
    x, y, sigma_y = data
    theta = best_theta(deg)
    resid = (y - p_fit(theta, x)) / sigma_y
    return np.sum(resid ** 2)

def compute_dof(deg, data = data):
    return data.shape[1] - (deg + 1)

def chi2_likelihood(deg, data = data):
    chi2 = compute_chi2(deg, data)
    dof = compute_dof(deg, data)
    return stats.chi2(dof).pdf(chi2)
# Print the chi2 likelihood
print("chi2 likelihood")
print(" Linear model: ", chi2_likelihood(1))
print(" Quadratic model: ", chi2_likelihood(2))
# Computing delta AIC
AIC_min = min(AIC1, AIC2)
print("Delta AIC values")
print(" Linear model: ", AIC1-AIC_min)
print(" Quadratic model: ", AIC2-AIC_min)
# Computing delta BIC
BIC_min = min(BIC1, BIC2)
print("Delta BIC values")
print(" Linear model: ", BIC1-BIC_min)
print(" Quadratic model: ", BIC2-BIC_min)
# Plotting 
fig, ax = plt.subplots()
for deg, color in zip([1, 2], ['blue', 'red']):
    v = np.linspace(0, 40, 1000)
    chi2_dist = stats.chi2(compute_dof(deg)).pdf(v)
    chi2_val = compute_chi2(deg)
    chi2_like = chi2_likelihood(deg)
    ax.fill(v, chi2_dist, alpha=0.3, color = color, label = 'Model {0} (deg = {0})'.format(deg))
    ax.vlines(chi2_val, 0, chi2_like, color = color, alpha = 0.6)
    ax.hlines(chi2_like, 0, chi2_val, color = color, alpha = 0.6)
    ax.set(ylabel='L(chi-square)')
    ax.set_xlabel('chi-square')
ax.legend(fontsize=14)
plt.show()