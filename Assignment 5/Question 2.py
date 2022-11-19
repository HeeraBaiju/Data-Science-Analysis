import numpy as np
import csv
from scipy import stats
import pandas as pd


datContent = [i.strip().split() for i in open('C:\\Users\\Heera Baiju\\Downloads\\Data Science\\Assignments\\Assignment 5\\Question 2 data dat.txt').readlines()]
with open("./HIP_star.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(datContent)

data = pd.read_csv('HIP_star.csv')

hyades = data[data['RA']>50]
hyades = hyades[hyades['RA']<100]
hyades = hyades[hyades['DE']>0]
hyades = hyades[hyades['DE']<25]
hyades = hyades[hyades['pmRA']>90]
hyades = hyades[hyades['pmRA']<130]
hyades = hyades[hyades['pmDE']>-60]
hyades = hyades[hyades['pmDE']<-10]
hyades = hyades[hyades['e_Plx']<5]
hyades = hyades[hyades['B-V']<0.2]

df = pd.concat([data, hyades])
non_hyades = df.drop_duplicates(keep = False)

d1 = hyades['B-V'].values
d2 = non_hyades['B-V'].values
d2 = d2[~np.isnan(d2)]

a = np.var(d1)
b = np.var(d2)
print("Hyades color array variance is :", a)
print("Non-hyades color array variance is :", b)


Tstat, pvalue = stats.ttest_ind(d1, d2, equal_var = False)
print("T-statistic value = ",Tstat)
print("2 sample t-test p value = : ",pvalue)
