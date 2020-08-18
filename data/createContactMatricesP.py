import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import time
from timeit import default_timer as timer
# from simCoronavirusSeattle2 import corEqs1, findBeta2
import pickle
mycolors = sns.color_palette("hls", 5)


#load the contact matrices from Prem et al for the US:
matAll = pd.read_csv('../../contactMatrices/contact_matrices_USA_Prem2017/usaAll.csv', sep=',',header=None).to_numpy()
# print(matAll)

matSchool = pd.read_csv('../../contactMatrices/contact_matrices_USA_Prem2017/usaSchool.csv', sep=',',header=None).to_numpy()

matWork = pd.read_csv('../../contactMatrices/contact_matrices_USA_Prem2017/usaWork.csv', sep=',',header=None).to_numpy()

matOL = pd.read_csv('../../contactMatrices/contact_matrices_USA_Prem2017/usaOtherLocations.csv', sep=',',header=None).to_numpy()

matHome = pd.read_csv('../../contactMatrices/contact_matrices_USA_Prem2017/usaHome.csv', sep=',',header=None).to_numpy()

populationUS = pd.read_csv('populationPyramidUS.csv')  #this was downloaded from https://www.populationpyramid.net/united-states-of-america/2020/


populationUS['total'] = populationUS['M'] + populationUS['F']
print(populationUS)



populationUSarray = populationUS['total'].to_numpy()
totalUSPop = np.sum(populationUSarray)

populationUSPercentages = (1/totalUSPop)*populationUSarray
# print(populationUSPercentages)


def createConsistentMatrix(myMat, totalPop):
    #based on this website: https://cran.r-project.org/web/packages/socialmixr/vignettes/introduction.html
    #which assumes that the rows are participants and the columns are contacts
    [n, m] = np.shape(myMat)
    newMat = np.zeros((n, n))

    for ivals in range(n):
        for jvals in range(n):
            newMat[ivals, jvals] = (1/(2*totalPop[ivals]))*(myMat[ivals, jvals]*totalPop[ivals] + myMat[jvals, ivals]*totalPop[jvals])

    return newMat


def createAlmostSymmetricMatrix(myMat, totalPop):
    pop = np.sum(totalPop)
    [n, m] = np.shape(myMat)
    newMat = np.zeros((n,n))
    for ivals in range(n):
        for jvals in range(n):
            newMat[ivals, jvals] = myMat[ivals, jvals]*pop/totalPop[ivals]
    return newMat

def computePercentages(myMat, refMat):
    [n, m] = np.shape(myMat)
    newMat = np.zeros((n, n))
    for ivals in range(n):
        for jvals in range(n):
            newMat[ivals, jvals] = myMat[ivals, jvals]/refMat[ivals, jvals]
    return newMat


populationUS16groups = np.concatenate([populationUSarray[0:15], [np.sum(populationUSarray[15:])]])
print((populationUS16groups))
pop75_80 = populationUSarray[15]
print(pop75_80)
pop80plus = np.sum(populationUSarray[16:])
print(pop80plus)



populationUS16groupsPercentages = (1/totalUSPop)*populationUS16groups
print(populationUS16groupsPercentages)


myNewMatAll2 = createConsistentMatrix(matAll.transpose(), populationUS16groups)

myNewMatHome2 = createConsistentMatrix(matHome.transpose(), populationUS16groups)

myNewMatSchool2 = createConsistentMatrix(matSchool.transpose(), populationUS16groups)

myNewMatWork2 = createConsistentMatrix(matWork.transpose(), populationUS16groups)

myNewMatOL2 = createConsistentMatrix(matOL.transpose(), populationUS16groups)




mymats2 = {"all": myNewMatAll2, 'home': myNewMatHome2, 'school': myNewMatSchool2, 'work': myNewMatWork2, 'otherloc': myNewMatOL2}
# #####save the results in a pickle file
today = time.strftime("%d%b%Y", time.localtime())
myfilename = 'consistentMatricesUS_polymodMethod' + today + '.pickle'
myfile = open(myfilename, 'wb')
pickle.dump(mymats2, myfile)
myfile.close()


