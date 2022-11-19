import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

datContent = [i.strip().split() for i in open("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignments\\Assignment 7\\SDSS_quasar.dat").readlines()]

with open("./SDSS_quasar.csv", "w") as f:
     writer = csv.writer(f)
     writer.writerows(datContent)

d = pd.read_csv('SDSS_quasar.csv', usecols=['z'])
data = d.values
t = np.linspace(-0.5, 5.5, 100)

kde1 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
kde1 = kde1.score_samples(t.reshape(-1,1))
kde2 = KernelDensity(kernel='exponential', bandwidth=0.2).fit(data)
kde2 = kde2.score_samples(t.reshape(-1,1))
dist = norm(np.mean(data), np.std(data)).pdf(t.reshape(-1,1))
plt.plot(t, np.exp(kde1), label='gaussian kernel')
plt.plot(t, np.exp(kde2), label='exponential kernel')
plt.fill(t.reshape(-1,1), dist, fc='black', alpha=0.2,label='input distribution')
plt.title('Kernel Density estimation')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()