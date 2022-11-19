import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t as student_t
from scipy.stats import norm
from scipy.stats import pearsonr


mu = 0
sigma = 1
gau_x = np.linspace(-10, 10, 1000)
k = 1e10
stu_x = np.linspace(-10, 10, 1000)


fig, ax = plt.subplots(figsize=(5, 3.75))

studist = student_t(k)
plt.plot(stu_x, studist.pdf(stu_x), ls = '-', c = 'green', label = r' Student t Distribution')

dist_g = norm(mu, sigma)
plt.plot(gau_x, dist_g.pdf(gau_x), ls = '--', c = 'yellow', label = r'mean=%.1f,sigma=%.1f Gaussian Distribution' % (mu, sigma))

plt.xlim(-5, 5)
plt.ylim(0.0, 0.45)
plt.xlabel('$x$')
plt.ylabel(r'$p(x|k)\ p(x|\mu,\sigma)$')
plt.title("Student's $t$ Distribution and Gaussian Distribution")
plt.legend()
plt.show()
corr_s, p_value_s = pearsonr(stu_x, studist.pdf(stu_x))
corr_g, p_value_g = pearsonr(gau_x, dist_g.pdf(gau_x))


print('p value of Student\'s t Distribution: {}' .format(p_value_s))
print('p value of Gaussian Distribution: {}' .format(p_value_g))