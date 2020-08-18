import numpy as np
from scipy.integrate import odeint
import os

import time
from matplotlib import pyplot as plt
import scipy.stats as stats
import pickle
myseed = np.abs(int((np.random.normal()) * 1000000))
np.random.seed(myseed)
print(myseed)
today = time.strftime("%d%b%Y", time.localtime())
numSimulations = 1000

#distribution for R0
lower, upper = 2.5,  3.5
mu, sigma = 3, 0.2
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
R0vals = X.rvs(numSimulations)


#distribution asymptomatic people
lower, upper = 0.15,  0.55
mu, sigma = 0.35, 0.2
Y = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
frac_asymVals = Y.rvs(numSimulations)



#distribution of duration of latent period
durEvals = np.random.gamma((3 / 0.1), 0.1, numSimulations) #distribution for gammaE

#distribution of duration of infectiousness after developing symptoms
lower, upper = 2, 5
mu, sigma = 3, 0.2
durIvals = np.random.gamma((3 / 0.2), 0.2, numSimulations) #distribution for gammaE


lower, upper = 1.1, 1.4
mu, sigma = 1.3, 0.05
Z = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
redPvals = Z.rvs(numSimulations)



durPvals = np.random.gamma((2 / 0.2), 0.2, numSimulations) #
#distribution for reduction/increase of infectiousness of asymptomatic infections:
redAdist = [0.2, 0.5, 1]
# #sample with replacement
redAvals = np.random.choice(redAdist, numSimulations)
# redAvals = np.random.uniform(low=0.2, high=1, size=numSimulations)




fig, ax = plt.subplots(6)
ax[0].hist(X.rvs(1000), density=True)
ax[1].hist(durIvals, density=True)
ax[2].hist(durPvals, density=True)
ax[3].hist(durEvals, density=True)
ax[4].hist(frac_asymVals, density=True)
ax[5].hist(redPvals, density=True)
plt.show()


params = np.vstack((durEvals, durIvals, durPvals, frac_asymVals, redAvals, redPvals, R0vals)).transpose()
# print((params))
print(np.median(params,0))
myfilename = 'randomMatrices/randomParamsForModelSensitivityAnalysis' + today + '.pickle'
print(myfilename)
myfile = open(myfilename, 'wb')
pickle.dump([params, myseed], myfile)
myfile.close()

# plt.show()