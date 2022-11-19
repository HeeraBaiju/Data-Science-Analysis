#------------------------------------------------------------
# Importing all libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
global data, x, y, sigma_y
#------------------------------------------------------------
import csv

#read the excel file
global data, x, y, sigma_y, val1, val2, val3  #defined as global so all functions can access these values
data = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignment 4\\D4data.xlsx")
x = data['x']
y = data['y']
sigma_y = data['sigma_y']
#------------------------------------------------------------
# Defining the required functions for polynomial fitting
def polynomial_fit(theta, x):
# Polynomial model of degree (len(theta) - 1)
return sum(t * x ** n for (n, t) in enumerate(theta))
#------------------------------------------------------------
# Defining the required logL function
def logL(theta):
# Gaussian log-likelihood of the model at theta
y_fit = polynomial_fit(theta, x)
return sum(stats.norm.logpdf(*args)
for args in zip(y, y_fit, sigma_y))
#------------------------------------------------------------
# Defining the function that returns the best theta for the fitting
def best_theta(degree):
theta_0 = (degree + 1) * [0]
neg_logL = lambda theta: -logL(theta)
return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)
#------------------------------------------------------------
# Defining the function that computes chi2
def compute_chi2(degree, data = data):
x, y, sigma_y = data
theta = best_theta(degree)
resid = (y - polynomial_fit(theta, x)) / sigma_y
return np.sum(resid ** 2)
#------------------------------------------------------------
# Defining the function that computes DOF
def compute_dof(degree, data = data):
return data.shape[1] - (degree + 1)
#------------------------------------------------------------
# Defining the function that computes chi2 likelihood
def chi2_likelihood(degree, data = data):
chi2 = compute_chi2(degree, data)
dof = compute_dof(degree, data)
return stats.chi2(dof).pdf(chi2)
# Compute the p value for the fit using linear model as the null hypothesis
def p_val(n):
return 1-stats.chi2(n-1).cdf(compute_chi2(1) - compute_chi2(n))
theta1 = best_theta(1)
theta2 = best_theta(2)
theta3 = best_theta(3)
# Print the Log L values
print("Log L values")
print(" Linear model: logL = ", logL(best_theta(1)))
print(" Quadratic model: logL = ", logL(best_theta(2)))
print(" Cubic model: logL = ", logL(best_theta(3)))
print(" Chi2 likelihood")
# Print the chi2 likelihood
print("Chi2 likelihood")
print(" Linear model: ", chi2_likelihood(1))
print(" Quadratic model: ", chi2_likelihood(2))
print(" Cubic model: ", chi2_likelihood(3))
# Print the p values
print("p_values") # The p value for null hypothesis will not be defined as the
delta chi square value is zero
print(" Quadratic model: ", p_val(2))
print(" Cubic model: ", p_val(3))
#------------------------------------------------------------
# Bayesian Analysis
# Compute the AIC values
AIC1 = -2*logL(theta1) + (2.0*2*20)/(17.0)
AIC2 = -2*logL(theta2) + (2.0*3*20)/(16.0)
AIC3 = -2*logL(theta3) + (2.0*4*20)/(15.0)
# Compute the BIC values
BIC1 = -2*logL(theta1) + 2*np.log(x.shape[0])
BIC2 = -2*logL(theta2) + 3*np.log(x.shape[0])
BIC3 = -2*logL(theta3) + 4*np.log(x.shape[0])
# Print the AIC values
print("AIC values")
print(" Linear model: ", AIC1)
print(" Quadratic model: ", AIC2)
print(" Cubic model: ", AIC3)
# Computing delta AIC
AIC_min = min(AIC1, AIC2, AIC3)
print("Delta AIC values")
print(" Linear model: ", AIC1-AIC_min)
print(" Quadratic model: ", AIC2-AIC_min)
print(" Cubic model: ", AIC3-AIC_min)
# Print the BIC values
print("BIC vlaues")
print(" Linear model: ", BIC1)
print(" Quadratic model: ", BIC2)
print(" Cubic model: ", BIC3)
# Computing delta BIC
BIC_min = min(BIC1, BIC2, BIC3)
print("Delta BIC values")
print(" Linear model: ", BIC1-BIC_min)
print(" Quadratic model: ", BIC2-BIC_min)
print(" Cubic model: ", BIC3-BIC_min)
# Plotting
t = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(t, polynomial_fit(theta1, t), label = r'$Linear\ fitting$')
plt.plot(t, polynomial_fit(theta2, t), label = r'$Quadratic\ fitting$')
plt.plot(t, polynomial_fit(theta3, t), label = r'$Cubic\ fitting$')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("Curve fitting using Linear, Quadratic and Cubic Models", size=15)
ax.errorbar(x, y, sigma_y, fmt='ok', ecolor = 'gray')
plt.show()