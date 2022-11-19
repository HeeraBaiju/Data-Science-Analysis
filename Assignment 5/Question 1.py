#Question-1
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignments\\Assignment 5\\Question 1 data.xlsx")
x = data['Dens']


a = stats.shapiro(x)
b = stats.shapiro(np.log(x))

print('Shapiro-wilk test of density, P values = :', a[1])
print('Shapiro-wilk test of log(density), P values = :', b[1])

mn1, sd1 = stats.norm.fit(x)
mn2, sd2 = stats.norm.fit(np.log(x))
t=np.linspace(-7, 7, 1000)
normal_distribution_1 = stats.norm.pdf(t, mn1, sd1)
normal_distribution_2 = stats.norm.pdf(t, mn2, sd2)

fig, ax = plt.subplots(2, 1, figsize = (6,6))
plt.xlim(-1.5, 3.5)
ax[0].plot(t, normal_distribution_1 , 'k-', label = 'Normal distribution 1')
ax[0].hist(x, density = True, histtype = 'stepfilled', alpha = 0.8, 
label='density', bins = 'fd')
ax[0].legend(loc = 'upper right')
ax[0].set_xlim([-3, 6])
ax[0].set_title('Histogram and Gaussian fitting of Density values')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$y$')
ax[1].plot(t, normal_distribution_2, 'k-', label='Normal distribution 2')
ax[1].hist(np.log(x), density = True, histtype='stepfilled', alpha = 0.8, 
label='log(density)', bins='fd')
ax[1].legend(loc = 'upper right')
ax[1].set_xlim([-2, 3])
ax[1].set_title('Histogram and Gaussian fitting of log(Density) values')
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$y$')
plt.tight_layout()
plt.show()
