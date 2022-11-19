# Importing necessary libraries
import numpy as np
from matplotlib import pyplot as plt

# Preparing the data for the plot
#x = np.arange(1, 100, 5)
#y = 32 * x

Luminosity = array('i', [345.2,
66.3,
684.0, 
209.0,
16.0, 
91.0, 
16.3, 
19.4, 
310.5, 
124.9, 
137.9, 
93.1, 
53.6, 
122.1, 
25.4, 
26.3,
196.9, 
68.8,
82.0, 
8.4, 
319.3, 
86.8, 
8.7, 
37.7, 
3.4, 
166.0,
104.2, 
45.0, 
14.5, 
38.1, 
17.8, 
31.1, 
89.0, 
66.3, 
17.7, 
4.5, 
69.1, 
1.1, 
6.2, 
47.2, 
23.4, 
160.8, 
6.4,
5.1, 
134.5, 4.1])

Redshift = array ('i', [0.1,
0.04,
 0.07,
 0.05,
 0.02,
 0.05,
 0.05,
 0.04,
 0.02,
 0.07,
0.0010,
0.12,
0.05,
0.04,
0.02,
0.0010,
0.09,
0.06,
0.02,
0.0010,
0.17,
0.02,
0.0010,
0.0010,
0.0010,
0.06,
0.03,
0.09,
0.0010,
0.03,
0.03,
0.04,
0.04,
0.06,
0.05,
0.0010,
0.05,
0.0010,
0.0010,
0.05,
0.0010,
0.06,
0.02,
0.02,
 0.03,
0.0010])
# Resizing the figure
plt.figure(figsize=[7, 5])

# Plotting the scatter plot
plt.scatter(x, y, c='g', alpha=0.6)
plt.title('Luminosity vs redshift', fontsize=15)
plt.xlabel('Redshift', fontsize=13)
plt.ylabel('Luminosity', fontsize=13)

# Changing to the log ticks at x and y axis using loglog
plt.loglog(basex=10, basey=2)

plt.show()