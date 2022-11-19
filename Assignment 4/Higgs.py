import scipy 
from scipy import stats
significance_Higgs =scipy.stats.norm.isf(1.7e-9)

#Higgs boson
print('The significance in terms of number of sigmas of the Higgs boson discovery claim from the p value given in the abstract of the ATLAS discovery paper, ',significance_Higgs)

#Ligo
significance_Ligo =scipy.stats.norm.isf(2e-7)
print('Significance in terms of number of sigmas LIGO',significance_Ligo)

#Goodness of fit
p_value=1-scipy.stats.chi2(67).cdf(65.2)
print('GOF using the best-fit',p_value)