import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

a = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignment 2\\Swiss Wind Power data.xlsx")
b = a["frequency"]
c = a["class"]
x = np.linspace(0, 20, 1000)
k = 2
lam = 6
mu = np.mean(x)

def weib(x, k, lam):
    return (k / lam) * (x / lam)**(k - 1) * np.exp(-(x / lam)**k)


fig, ax = plt.subplots(figsize=(5, 3.75))
plt.plot(x, 100*weib(x, k, lam), ls = '-', c='red', label = r'$k=%.1f,\
\lambda=%.1f$' % (k, lam))
plt.step(c, b, ls = '-', c='blue', label = r'$Data$')
plt.xlabel('$Wind\ speed\ (m/s)$')
plt.ylabel('$Frequency$')
plt.title('Weibull Distribution')
plt.xlim(0, 20)
plt.ylim(0.0, 16)
plt.legend()
plt.show()