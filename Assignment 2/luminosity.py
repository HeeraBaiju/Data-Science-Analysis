import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr
#------------------------------------------------------------
# Importing dataset
a = pd.read_excel("C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignment 2\\Luminosity Vs Redshift.xlsx")
x_points = np.log(a["Lx"])
y_points = np.log(a["z"])
#------------------------------------------------------------
# Creating histogram
fig, ax = plt.subplots(figsize=(5, 3.75))
plt.scatter(x_points, y_points, color = 'red')
#------------------------------------------------------------
# Show plot
ax.set_xlabel('Luminosity')
ax.set_ylabel('Redshift')
plt.title('$Luminosity\ Vs\ Redshift$')
plt.show()
#------------------------------------------------------------
# Print Spearman, Pearson and Kendall-tau correlation coefficients and the p value for the null hypothesis
Pearson_Correlation_Coefficient, p_value_p = pearsonr(x_points, y_points)
Kendalltau_Correlation_Coefficient, p_value_k = kendalltau(x_points, y_points)
Spearman_Correlation_Coefficient, p_value_s = spearmanr(x_points, y_points)
print('Pearsons correlation: {}' .format(Pearson_Correlation_Coefficient))
print('p value of Pearson Correlation Coefficient: {}' .format(p_value_p))
print('Kendalltau correlation: {}' .format(Kendalltau_Correlation_Coefficient))
print('p value of Kendalltau Correlation Coefficient: {}' .format(p_value_k))
print('Spearman correlation: {}' .format(Spearman_Correlation_Coefficient))
print('p value of Spearman Correlation Coefficient: {}' .format(p_value_s))
