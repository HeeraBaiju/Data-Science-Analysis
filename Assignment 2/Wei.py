
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

a = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignment 2\\Swiss Wind Power data.xlsx")
  
freq = a["frequency"]
windsp = a["class"]
x = np.linspace(0, 20, 1000)

k = 2
al = 6
v = np.mean(x)

def weib(x, k, al):
    return (k / al) * (x / al)**(k - 1) * np.exp(-(x / al)**k)


fig, ax = plt.subplots(figsize=(5, 3.75))
plt.plot(x, 100*weib(x, k, al), ls = '-', c='g', label = r'k=%.1f and al=%.1f' % (k, al))
plt.step(windsp, freq , ls = '-', c='black', label = r'$Data$')
plt.xlabel('$Wind\ speed\ (m/s)$')
plt.ylabel('$Frequency$')
plt.title('Weibull Distribution')
plt.xlim(0, 20)
plt.ylim(0.0, 16)
plt.legend()
plt.show()





