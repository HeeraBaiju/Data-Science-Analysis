import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pearsonr


a = pd.read_excel("C:\\Users\Heera Baiju\\Downloads\\Data Science\\Assignment 2\\LuminosityHeera.xlsx")
Luminosity = np.log(a["Lx"])
Redshift = np.log(a["z"])
# Resizing the figure
plt.figure(figsize=[7, 5])

# Plotting the scatter plot
plt.scatter(Luminosity, Redshift, c='g', alpha=0.6)

plt.title('Luminosity vs redshift', fontsize=15)
plt.xlabel('Luminosity', fontsize=13)
plt.ylabel('Redshift', fontsize=13)
plt.show()

Pearson_Correlation_Coefficient, p_value_p = pearsonr(Luminosity, Redshift)
Kendalltau_Correlation_Coefficient, p_value_k = kendalltau(Luminosity, Redshift)
Spearman_Correlation_Coefficient, p_value_s = spearmanr(Luminosity, Redshift)
print('Pearsons correlation: {}' .format(Pearson_Correlation_Coefficient))
print('p value of Pearson Correlation Coefficient: {}' .format(p_value_p))
print('Kendalltau correlation: {}' .format(Kendalltau_Correlation_Coefficient))
print('p value of Kendalltau Correlation Coefficient: {}' .format(p_value_k))
print('Spearman correlation: {}' .format(Spearman_Correlation_Coefficient))
print('p value of Spearman Correlation Coefficient: {}' .format(p_value_s))

