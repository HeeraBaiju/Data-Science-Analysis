#------------------------------------------------------------
# Importing all libraries 
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
#------------------------------------------------------------
# Importing dataset
data = pd.read_excel("E:\Data Science\Assignment-3\Data.xlsx")
x_values = data ['x']
y_values = data ['y']
sigmaY = data ['sigmaY']
#------------------------------------------------------------
# Objective function
def function(x, a, b):
return a * x + b
#------------------------------------------------------------
# Getting the optimal values for the parameters and the estimated covariance of 
parameters
val = np.array([0, 0]) # Assuming the values as (0,0)
param, param_cov = curve_fit(function, x_values, y_values, val, sigmaY)
print("Line function coefficients: {}" .format(param) )
print("Covariance of coefficients: {}" .format(param_cov) )
#------------------------------------------------------------
# Getting the y-intercept and slope given by curve-fit() function 
t = np.linspace(0, 300, 1000)
c = param[1] # The y-intercept given by curve-fit() function
m = param[0] # The slope given by curve-fit() function
print("The value of y-intercept is {}" .format(c) )
print("The value of slope is {}" .format(m) )
#------------------------------------------------------------
# Plot the results
plt.errorbar(x_values, y_values, sigmaY , fmt='.k', ecolor='gray', label='Plot of 
y Vs x')
plt.plot(t, m*t+c, '--', color ='blue', label ="Optimized data")
plt.xlim(0, 300, 50)
plt.ylim(0, 700, 100)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
