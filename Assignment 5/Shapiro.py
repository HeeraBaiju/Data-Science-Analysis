#------------------------------------------------------------
# Importing all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#------------------------------------------------------------
# Reading the data using pandas
data = pd.read_excel('E:\Data Science Analysis\Assignment-5\data_Q1.xlsx')
x = data['Dens']
#------------------------------------------------------------
# Shapiro-wilk test for density values
result1 = stats.shapiro(x)
#------------------------------------------------------------
# Shapiro-wilk
result2 = stats.shapiro(np.log(x))
print('The p value for shapiro-wilk test of density values is :', result1[1])
print('The p value for shapiro-wilk test of log(density) values is :', 
result2[1])
print('-> It is clear that p value of log(density) is higher so it cannot reject 
the null hypothesis that the data is drawn from normal distribution.')
#------------------------------------------------------------
# Curve fitting
mu_x1, std_x1 = stats.norm.fit(x)
mu_x2, std_x2 = stats.norm.fit(np.log(x))
t=np.linspace(-7, 7, 1000)
norm1 = stats.norm.pdf(t, mu_x1, std_x1)
norm2 = stats.norm.pdf(t, mu_x2, std_x2)
#------------------------------------------------------------
# Plotting the normal fits and the respective histograms
fig, ax = plt.subplots(2, 1, figsize = (6,6))
plt.xlim(-1.5, 3.5)
ax[0].plot(t, norm1, 'k-', label = 'norm 1 pdf')
ax[0].hist(x, density = True, histtype = 'stepfilled', alpha = 0.8, 
label='density', bins = 'fd')
ax[0].legend(loc = 'upper right')
ax[0].set_xlim([-3, 6])
ax[0].set_title('Histogram and Gaussian fitting of Density values')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[1].plot(t, norm2, 'k-', label='norm 2 pdf')
ax[1].hist(np.log(x), density = True, histtype='stepfilled', alpha = 0.8, 
label='log(density)', bins='fd')
ax[1].legend(loc = 'upper right')
ax[1].set_xlim([-2, 3])
ax[1].set_title('Histogram and Gaussian fitting of log(Density) values')
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$y$')
plt.tight_layout()
plt.show()
