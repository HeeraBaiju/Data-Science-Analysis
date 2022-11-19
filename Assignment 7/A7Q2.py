import numpy as np
import nestle
from scipy import stats
global data, x, y, sigma_y

data = np.array([[ 0.417022004703, 0.720324493442, 0.000114374817345, 
0.302332572632, 
0.146755890817, 0.0923385947688, 0.186260211378, 
0.345560727043,
0.396767474231, 0.538816734003, 0.419194514403, 
0.685219500397, 
0.204452249732, 0.878117436391, 0.0273875931979, 
0.670467510178, 
0.417304802367, 0.558689828446, 0.140386938595, 0.198101489085
],
[ 0.121328306045, 0.849527236006, -1.01701405804, -
0.391715712054, 
-0.680729552205, -0.748514873007, -0.702848628623, -
0.0749939588554, 
0.041118449128, 0.418206374739, 0.104198664639, 0.7715919786, 
-0.561583800669, 1.43374816145, -0.971263541306, 
0.843497249235, 
-0.0604131723596, 0.389838628615, -0.768234900293, -
0.649073386002 ],
[ 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
0.1 , 0.1 , 0.1 , 0.1 , 0.1 ,
0.1 , 0.1 , 0.1 , 0.1 , 0.1 ]])
x, y, sigma_y = data

def polynomial_fit(theta, x):

    return sum(t * x ** n for (n, t) in enumerate(theta))

def logL(theta): 

    y_fit = polynomial_fit(theta, x)
    return sum(stats.norm.logpdf(*args) for args in zip(y, y_fit, sigma_y))

def prior_transform(x):
    return 10.0 * x - 5.0

Lin = nestle.sample(logL, prior_transform, 2)
Quad = nestle.sample(logL, prior_transform, 2)

print(" Linear model's Bayesian evidence : ", Lin.logz)
print(" Quadratic model's Bayesian evidence : ", Quad.logz)