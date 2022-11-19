import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


q2data = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Q2Data.xlsx")
xq2data = q2data ['x']
yq2data = q2data ['y']
sigmaq2data = q2data ['sigmaY']

def function(x, m, c):
    return m * x + c

arr = np.array([0, 0]) 
param, param_cov = curve_fit(function, xq2data, yq2data, arr, sigmaq2data)
print("Line function coefficients: {}" .format(param) )
print("Covariance of coefficients: {}" .format(param_cov) )
 
x1 = np.linspace(0, 300, 1000)
c1 = param[1] 
m1 = param[0] 
print("Y-intercept = {}" .format(c1) )
print("Slope = {}" .format(m1) )

plt.errorbar(xq2data, yq2data, sigmaq2data , fmt='.k', ecolor='red', label='y vs x')
plt.plot(x1, m1*x1+c1, '--', color ='green', label ="Optimized data")
plt.xlim(0, 300, 50)
plt.ylim(0, 700, 100)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
