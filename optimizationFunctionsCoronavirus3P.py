import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
# import seaborn as sns
# from matplotlib import pyplot as plt
import pickle
# from matplotlib.colors import ListedColormap
# # %matplotlib qt
# from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import sys
import time
import pyswarms as ps
import pandas as pd
sys.path.insert(1, '../')




from coronavirusMainFunctions import findBetaNewModel_eqs2_withHosp4
from coronavirusMainFunctions import coronavirusEqs2_withHospitalizationsAndICU_withVaccine4
# print(sys.path)



def findFractionSus(y, numAgeGroups):
    """
    finds the relative fraction of  the  susceptibles, among susceptibles, exposed, asymptomatic infected,  pre-symptomatic
    infected, and recovered asymptomatic in each age group. This is valid ONLY for equations coronavirusEqs2_withHospitalizationsAndICU_withVaccineBis, coronavirusEqs2_withHospitalizationsAndICU_withVaccine3, coronavirusEqs2_withHospitalizationsAndICU_withVaccine4
    :param y:
    :return:
    """
    temp = np.reshape(y, (23, numAgeGroups))

    relativeTotal = (temp[0, :] + #susceptibles
                           temp[1, :] + #exposed
                           temp[2, :]+ #asymptomatic infected
                           temp[3, :]+ # pre-symptomatic infected
                           temp[8, :]) #recovered asymptomatic

    fractionSus = np.divide(temp[0, :], relativeTotal)
    return fractionSus


def splitVaccineAmongAgeGroups(vacVector, fracPerAgeGroup, totalPop):
    """
    This function will split vaccine across age groups for a single vaccination group
    :param vacVector: a vector of size 1*5 with the number of vaccines for each vaccination group
    fracPerAgeGroup: a list with 5 entries, each entry has the number of age groups in that vaccination group.
    :return: a vector of size 1X16 with the number of vaccines to be given to each age group
    """
    #group 1: 0-20:  this includes 4 age groups: 0-5, 5-10, 10-15, 15-20
    #group 2: 20-50: this includes 6 age groups: 20-25, 25-30, 30-35, 35-40, 40-45, 45-50
    #group 3: 50-65: this includes 3 age groups: 50-55, 55-60, 60-65
    #group 4: 65-75  this includes 2 age groups: 65-70, 70-75
    #group 5: 75+    this includes 1 age group:  75+
    mylist = np.zeros(16) 

    mylist[0: 4] = vacVector[0]*fracPerAgeGroup[0]
    mylist[4: 10] = vacVector[1]*fracPerAgeGroup[1]
    mylist[10:13] = vacVector[2]*fracPerAgeGroup[2]
    mylist[13:15] = vacVector[3]*fracPerAgeGroup[3]
    mylist[15] = vacVector[4]*fracPerAgeGroup[4]

    mylist2 = np.minimum(np.floor(mylist), np.floor(totalPop))

    return mylist2



    #########################################
    ## extraParams is a list of lists
    ## type(extraParams)
    ##    
    ## extraParams[0] is  mortality_rate_16  is a numpy.ndarray of length 16 
    ## extraParams[1] is  groupFracs  is a list of length 5
    ## extraParams[2] is  y0   is a numpy.ndarray shape (368,)
    ## extraParams[3] is  numAgeGroups  is an int
    ## extraParams[4] is  numVaccinesAvailable  is an int
    ## extraParams[5] is  numVaccineGroups  is an int
    ## extraParams[6] is  paramsODE  is a list
    ## extraParams[7] is  totalPop16  is a numpy.ndarray of length numAgeGroups
    ## extraParams[8] is  totalPop5   is a numpy.ndarray of length numVaccineGroups
    ## extraParams[9] is  tspan  is a numpy.ndarray of length 200

###########################################################################
###########################################################################
###  NEW!! OBJECTIVE FUNCTION WITH REALLOCATION ADDED 6/29/20  #####
###########################################################################
###########################################################################

def objectiveFunctionICU_HospPeak_with_reallocation(fracVacs, extraParams):
    '''
    This function computes all the objectives we want for a given decision variable fracVacs.
    It will reallocate whatever vaccine is not used starting with the oldest age group and continuing in
    that order.
    '''
    # fracVacs is a vector (array) of fraction of VACCINE accross vaccine groups
    # numAgeGroups = number of age groups
    # numVaccineGroups = number of vaccine groups (fewer than age groups)
    # totalPop = vector with the number of people in each age group

    # extraParams will have everything extra
    [deathRate, groupFracs, initCond, numAgeGroups, numVaccinesAvailable, numVaccineGroups, paramsODE, totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams

    numVaccinesByGroup = np.multiply(fracVacs,numVaccinesAvailable)
    isExcessVaccineByGroup = np.zeros(numVaccineGroups)
    excessVaccine = 0

    #check that the number of vaccines given to each vaccination group is no more than the pop in that group:
    for i in range(0, numVaccineGroups):
        if numVaccinesByGroup[i] > totalPopByVaccineGroup[i]:
            isExcessVaccineByGroup[i] = 1
    
    ## skims off excess vaccine each group.
    if sum(isExcessVaccineByGroup) > 0:
        excessVaccine = np.dot(isExcessVaccineByGroup,(numVaccinesByGroup - totalPopByVaccineGroup))
        newNumVaccinesByGroup = numVaccinesByGroup - np.multiply(isExcessVaccineByGroup, (numVaccinesByGroup - totalPopByVaccineGroup))
        numVaccinesByGroup = newNumVaccinesByGroup

    ###############################################################
    ### with a re-allocation strategy implemented, below.
    ### unused in main file for now.
    ###############################################################
    print(excessVaccine)
    print(numVaccinesByGroup)
    if excessVaccine > 0:

        vaccineSaturationByGroup = np.zeros(numVaccineGroups)

        for i in range(0, numVaccineGroups):
            if numVaccinesByGroup[i] ==  totalPopByVaccineGroup[i]:
                vaccineSaturationByGroup[i] = 1

        numUnvaccinatedByGroup = totalPopByVaccineGroup - numVaccinesByGroup
     
        ###################################################################
        ## 1st do oldest age group, 2nd oldest, and on down to youngest. 
        reallocationOrder =  list(range(numVaccineGroups-1,-1,-1))

        for i in reallocationOrder: 
            print(i)
            if vaccineSaturationByGroup[i] == 0:
                if excessVaccine <= numUnvaccinatedByGroup[i]:
                    numVaccinesByGroup[i] = numVaccinesByGroup[i] + excessVaccine
                    excessVaccine = 0
                    if excessVaccine == numUnvaccinatedByGroup[i]:
                        vaccineSaturationByGroup[i] = 1
                else:
                    excessVaccine = excessVaccine - numUnvaccinatedByGroup[i]
                    numVaccinesByGroup[i] = totalPopByVaccineGroup[i]
                    vaccineSaturationByGroup[i] = 1
    print(excessVaccine)
    print(numVaccinesByGroup)
    ### end new addition
    ###
    
    # 2. Feed the groupFracs to the model
    #    this distributes the vaccine from the vaccine group into the age groups 
    #    uniformly by % of age group in the vaccine group.
    numVaccinesAgeGroup = splitVaccineAmongAgeGroups(numVaccinesByGroup, groupFracs, totalPopByAgeGroup)

    # run the model.
    [totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU] = \
        runVaccination(deathRate,  initCond, numAgeGroups, numVaccinesAgeGroup, paramsODE, tspan)

    # This is the actual fractions of available vaccine
    # distributed among the vaccine groups.
    actualFracs = (1/numVaccinesAvailable)*numVaccinesByGroup


    # Return output with all what we want to know.
    # Bundle as a list of lists, then flatten to a list of numbers.
    mytempList = [fracVacs, actualFracs.tolist(), [excessVaccine, totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU]]
    flat_list = [item for sublist in mytempList for item in sublist]
    return flat_list
    ## Index tracking
    ## 0, 1, 2, 3, 4:  fracVacs (proportions over 5 vaccination groups)
    ## 5, 6, 7, 8, 9: actualFracs (after skimming off excess vaccine)
    ## 10: excessVaccine (scalar)
    ## 11: totalInfections
    ## 12: totalSymptomaticInfections
    ## 13: totalHospitalizations
    ## 14: totalICU
    ## 15: totalDeaths
    ## 16: maxHosp
    ## 17: maxICU
    
    #########################################









###########################################################################
###########################################################################

def objectiveFunctionICU_HospPeak(fracVacs, extraParams):
    '''
        This function computes all the objectives we want for a given decision variable fracVacs.
    '''
    # fracVacs is a vector (array) of fraction of VACCINE accross vaccine groups
    # leave everything as a variable (see below for extraParams)
    # numAgeGroups = number of age groups
    # numVaccineGroups = number of vaccine groups (fewer than age groups)
    # totalPop = vector with the number of people in each age group

    # extraParams will have everything extra
    [deathRate, groupFracs, initCond, numAgeGroups, numVaccinesAvailable, numVaccineGroups, paramsODE, totalPopByAgeGroup, totalPopByVaccineGroup, tspan] = extraParams

    numVaccinesByGroup = np.array(fracVacs) * numVaccinesAvailable
    isExcessVaccineByGroup = np.zeros(numVaccineGroups)
    excessVaccine = 0
    # print(numVaccinesByGroup)
    # print(totalPopByVaccineGroup)
    for i in range(0, numVaccineGroups):
        if numVaccinesByGroup[i] > totalPopByVaccineGroup[i]:
            isExcessVaccineByGroup[i] = 1

    if sum(isExcessVaccineByGroup) > 0:
        excessVaccine = np.dot(isExcessVaccineByGroup,(numVaccinesByGroup - totalPopByVaccineGroup))
        newNumVaccinesByGroup = numVaccinesByGroup - np.multiply(isExcessVaccineByGroup, (numVaccinesByGroup - totalPopByVaccineGroup))
        numVaccinesByGroup = newNumVaccinesByGroup
    # print(excessVaccine)

    # 2. Feed the groupFracs to the model
    #    this distributes the vaccine from the vaccine group into the age groups 
    #    uniformly by % of age group in the vaccine group.
    numVaccinesAgeGroup = splitVaccineAmongAgeGroups(numVaccinesByGroup, groupFracs, totalPopByAgeGroup)

    # run the model.
    [totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU] = \
        runVaccination(deathRate,  initCond, numAgeGroups, numVaccinesAgeGroup, paramsODE, tspan)

    # This is the actual fractions of available vaccine
    # distributed among the vaccine groups.
    # It does _not_ tell us the proportion of each age group being vaccinated.
    actualFracs = (1/numVaccinesAvailable)*numVaccinesByGroup


    # Return output with all what we want to know.
    # Bundle as a list of lists, then flatten to a list of numbers.
    mytempList = [fracVacs, actualFracs.tolist(), [excessVaccine, totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU]]
    flat_list = [item for sublist in mytempList for item in sublist]
    return flat_list
    ## Index tracking
    ## 0, 1, 2, 3, 4:  fracVacs (proportions over 5 vaccination groups)
    ## 5, 6, 7, 8, 9: actualFracs (after skimming off excess vaccine)
    ## 10: excessVaccine (scalar)
    ## 11: totalInfections
    ## 12: totalSymptomaticInfections
    ## 13: totalHospitalizations
    ## 14: totalICU
    ## 15: totalDeaths
    ## 16: maxHosp
    ## 17: maxICU
    



##################################################################
##################################################################
def constraintCheckFunction(vector):
    ## We have the following cases:
    ##   1. the nonnegativity constraints are satisfied and the sum 
    ##      of fracVacs does not exceed 1. No repair needed.
    ##   2. the nonnegativity constraints are satisfied but the sum
    ##      of the fracVacs exceeds 1. Repair to the sum is needed.
    ##   3. the nonnegativity constraints are not satisfied
    ##      --> step i, reset negative entries to zero.
    ##      --> step ii, check if sum constraint is satisfied. If not, repair.
    exitCode = 0
    test = np.array(vector)
    ## If any entry is negative, exitCode is 3.
    if np.any(test < 0):
        exitCode = 3
    ## Otherwise, if nonnegative, and sum exceeds 1, then exitCode is 2.
    else:
        if sum(test) > 1:
            exitCode = 2
        ## Otherwise exitCode is 1 (fracVacs is fine).
        else: 
            exitCode = 1
    return exitCode 


def repairVector(vector, exitCode):
    '''
    input: a numpy array & an integer exit code
    output: a (repaired) numpy array of same length
    If exitCode = 1, no change is made to  vector. 
    If exitCode = 2, the nonnegativity constraints are satisfied but the sum
       of the fracVacs exceeds 1. Repair to the sum is needed. 
       So vector is replaced with itself divided by its sum.
    If exitCode = 3, 
          --> first reset negative entries to zero.
          --> then  check if sum constraint is satisfied. If not, repair.
    This function will repair a vaccine vector that is outside the limits by 
    keeping the relative fractions in each entry
    but forcing the sum to be equal to 1
    :param vector, exitCode:
    :return:
    '''
    newVector = vector

    if exitCode == 3:
        ## Returns boolean vector (True & False) entry-wise.
        testBoolean = vector > 0
        ## Converts True to  1  and  False to 0, so entry is
        ## a 1 if the entry >= 0 and a 0 if entry < 0.
        intTestBoolean = testBoolean*1
        ## This multiplication converts every negative entry to zero
        ## and leaves nonnegative entries unchanged.
        newVector = np.multiply(intTestBoolean,vector)
        
    ## Re-test constraint sum violation for the new nonnegative vector.        
    vectorSum = np.sum(newVector)

    ## Below repairs the following cases:
    ##    --> case 3, after nonnegativity is repaired, if vectorSum > 1, AND
    ##    --> case 2, whic is that vectorSum > 1
    if vectorSum > 1:
        temp = np.array([newVector[i]/vectorSum for i in range(len(vector))])
        newVector = temp

    return newVector



###############################################################################
def objectiveFunction_NM(fracVacs, extraParamsNM):
    '''
    This is the objective function that will be passed to Nelder-Mead. It evaluates the function objectiveFunctionICU_HospPeak
    for the decision variable fracVacs given. Depending on the variable myIndex, it returns the appropriate objective.
    Because NM will try decision variables that will NOT satisfy the constraints a priori, we need to first check that
    this particular fracVacs satisfies being between 0 and 1 and "repair" it if it is not.
    '''
    [extraParams, myIndex] = extraParamsNM
    ## REPAIRVECTOR
    constraintCheck = constraintCheckFunction(fracVacs)
    newFracVacs = repairVector(fracVacs, constraintCheck)
    ## If no modification needed for fracVacs, then
    if constraintCheck == 1:
        modelOutput = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
        return float(modelOutput[myIndex])
    else:
        fracVacs = np.copy(newFracVacs)
        modelOutput = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
        return float(modelOutput[myIndex])
        

        
############################################################################################################
############################################################################################################
############################################################################################################

def runVaccination(deathRate,  initCond, numAgeGroups, numVaccinesAgeGroup, paramsODE, tspan):
    """
    runs the ODEs with vaccination. Vaccination is assumed to occur all at once. 
    :param deathRate: 1x16 vector with the death rate AMONG ALL INFECTIONS 
    :param initCond: initial conditions just before vaccination
    :param numAgeGroups: number of age groups in the model
    :param numVaccinesAgeGroup: number of vaccines per age group
    :param paramsODE: all the parameters needed to run the ODE
    :param tspan: time horizon for the optimization
    :return: 
    """
    # Compute the fraction of people susceptible in each age group relative 
    # to other groups that could potentially receive vaccination.
    fractionSus = findFractionSus(initCond, numAgeGroups)
    # print(fractionSus)
    initCond = initCond.reshape((23, numAgeGroups))
    # print(initCond)
    initCond2 = np.copy(initCond)
    # print((initCond2))
    # Move people out of the susceptible groups.
    peopleVac = np.minimum(initCond[0,:], numVaccinesAgeGroup*fractionSus)
    initCond2[0, :] = initCond[0,:] - peopleVac  #we only use the fractionSus in the susceptible
    # population, see notes

    # Move people into the vaccinated susceptible groups.
    initCond2[11, :] = initCond2[11, :] + peopleVac
    # print(np.sum((initCond2)))
    # Flatten the initial conditions to pass to ODE:
    initCond2 = initCond2.reshape(23 * numAgeGroups)

    out = odeint(coronavirusEqs2_withHospitalizationsAndICU_withVaccine4, initCond2, tspan, args=(paramsODE,))
    # Compute the number of cases, hospitalizations and deaths.
    # print(np.sum(out[-1, :22*numAgeGroups]))
    # Extract the information we want from the output, ie. total number of infections, 
    # total number of symptomatic infections,
    # total nubmer of hospitalizations, total number of people requiring ICU, 
    # total number of deaths,
    # number of hospitalizations at peak, number of ICU at peak.
    out2 = out[-1,:].reshape(23, numAgeGroups)
    # print(np.sum(out2[0:22, :]))
    # infections = out2[7,:] + out2[8, :] + out2[9,:] + out2[10,:] + \
    # out2[18 ,:] + out2[19 ,:] + out2[20 ,:] + out2[21 ,:] #total recovered vaccinated

    infections = out2[22, :]

    totalInfections = np.sum(infections)
    # print(totalInfections)

    #this reads the recovered symptomatic groups. We need to substract from here the recovered that were already there before vaccination
    totalSymptomaticRecoveredPreVaccination = np.sum(initCond[7,:] + initCond[9,:] + initCond[10, :])
    totalSymptomaticInfections = np.sum(out[-1, (7 * numAgeGroups): (8 * numAgeGroups)]) + \
                                 np.sum(out[-1, (9 * numAgeGroups): (11 * numAgeGroups)]) + \
                                 np.sum(out[-1, (18 * numAgeGroups):(19 * numAgeGroups)]) + \
                                 np.sum(out[-1, (20 * numAgeGroups):(22 * numAgeGroups)]) - \
                                 totalSymptomaticRecoveredPreVaccination

    #this reads the recovered hospitalized. We need to substract from here the recovered that were already there before vaccination
    totalHopsPreVaccination = np.sum(initCond[9,:])
    totalHospitalizations = (np.sum(out[-1, (9 * numAgeGroups):(10 * numAgeGroups)]) +  #hospitalized unvaccinated
                            np.sum(out[-1, (20 * numAgeGroups): (21 * numAgeGroups)])) - \
                            totalHopsPreVaccination

    #this reads the recovered from the ICU, we need to substract those who were in the ICU before vaccination
    totalICUPreVaccination = np.sum(initCond[10, :])
    totalICU = (np.sum(out[-1, (10 * numAgeGroups): (11 * numAgeGroups)]) + #ICU unvaccinated
                np.sum(out[-1, (21 * numAgeGroups): (22 * numAgeGroups)])) - \
                totalICUPreVaccination

    totalDeaths = np.dot(infections, deathRate)

    hosp = np.sum(out[:,(5*numAgeGroups):(6*numAgeGroups) ],1) + np.sum(out[:,(16*numAgeGroups): (17*numAgeGroups) ], 1)

    icu = np.sum(out[:,(6*numAgeGroups):(7*numAgeGroups) ],1) + np.sum(out[:,(17*numAgeGroups): (18*numAgeGroups) ], 1)
    maxHosp = np.max(hosp)
    maxICU = np.max(icu)
    # return out
    return [np.rint(totalInfections), np.rint(totalSymptomaticInfections), np.rint(totalHospitalizations), np.rint(totalICU),
            np.rint(totalDeaths), np.rint(maxHosp), np.rint(maxICU)]



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################




######################################################################
######################################################################
def coarseGlobalSearch(readFileName, extraParams, lowerIndex=0, upperIndex = 10626):
    """
    This function takes in a file (which is a partition of 100% over
    vectors of length 5, incremented by either 5% or 10%, depending on 
    the filename.
    
    It also takes the row indices to be read as well as the additional parameters
    that will be fed into the objective function.

    For each row (a feasible decision vector) it evaluates the function objectiveFunctionICU_HospPeak and returns the
    values of all of the objectives for that particular feasible solution.

    The output is a huge matrix that consists of...the objective function
    evaluated at the desired indices.

    """

    outputMatrix = np.zeros((upperIndex - lowerIndex,19))

    f = open(readFileName, 'rb')
    simplexMatrix = pickle.load(f)
    f.close()
 
    for i in range(lowerIndex,upperIndex):
        fracVacs = simplexMatrix[i,:]
        # storing the index in the last column, -1 in python is the last element of the vector
        # fracVacs + actualFracs.tolist() + [excessVaccine, totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU] = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
        outputMatrix[i, -1] = i

        outputMatrix[i, 0:-1] = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
            # # List is returned:
            # #    fracVacs +   (dimension is 5)  FLOAT
            # #    actualFracs.tolist() +  (dimension is 5) FLOAT
            # #    [excessVaccine,   INT
            # #    totalInfections,  INT
            # #    totalSymptomaticInfections,  INT
            # #    totalHospitalizations,  INT
            # #    totalICU,  INT
            # #    totalDeaths,  INT
            # #    maxHosp,  INT
            # #    maxICU] INT
            # #    index   INT
            # # Total number of columns for this is: 19
    # results = [outputMatrix, extraParams]
    return outputMatrix


def coarseGlobalSearch2(simplexMatrix, extraParams, lowerIndex=0, upperIndex=10626):
    """
    This function takes in a matrix (which is a partition of 100% over
    vectors of length 5, incremented by either 5% or 10%, depending on
    the filename. These partitions were pre-computed using one part SageMath (CoCalc)
    and one part python, in the file GENERATE_CELL_PARITIONS.py))

    It also takes the row indices to be read as well as the additional parameters
    that will be fed into the objective function.

    The output is a huge matrix that consists of...the objective function
    evaluated at the desird indices.

    AFTER this is computed, the matrix will be sorted by a column (probably
    the desired objective function, such as total deaths). The top 100 values
    (meaning the 100 smallest values) will be fed in as starting points to PS.
    """
    # lowerIndex = 0
    # upperIndex = 50
    # THE 100% SPLIT BY 5% (100 DIVIDED BY 20) BY 5 VACCINE GROUPS FILE HAS 10626 ENTIRES.

    outputMatrix = np.zeros((upperIndex - lowerIndex, 19))
    for i in range(lowerIndex, upperIndex):
        fracVacs = simplexMatrix[i, :]
        # storing the index in the last column, -1 in python is the last element of the vector
        # fracVacs + actualFracs.tolist() + [excessVaccine, totalInfections, totalSymptomaticInfections, totalHospitalizations, totalICU, totalDeaths, maxHosp, maxICU] = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
        outputMatrix[i, -1] = i

        outputMatrix[i, 0:-1] = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
        # # List is returned:
        # #    fracVacs +   (dimension is 5)  FLOAT
        # #    actualFracs.tolist() +  (dimension is 5) FLOAT
        # #    [excessVaccine,   INT
        # #    totalInfections,  INT
        # #    totalSymptomaticInfections,  INT
        # #    totalHospitalizations,  INT
        # #    totalICU,  INT
        # #    totalDeaths,  INT
        # #    maxHosp,  INT
        # #    maxICU] INT
        # #    index   INT
        # # Total number of columns for this is: 19
    # results = [outputMatrix, extraParams]
    return outputMatrix


################################################################################
def randomSampleSimplex(dim, numSamples):
    """
    This function in the simplex dimension+1 and number of samples
    to be generated.
    
    FOR NOW, this is a random sample on the simplex, but it is NOT
    a uniform sample on the simplex (place-holder is below)

    Once I get the rest of the pieces to work I'll implement the uniform sample.
    (It's an understood problem, so I should be able to find an implementation
    somewhere.)
    """
    randomPoints = np.random.uniform(low=0.0, high=1.0, size=(numSamples,dim,))
    print(randomPoints.shape)
    ## For each row, sum up elements. 
    randomSum = np.sum(randomPoints, axis=1)
    ## Divide each row of randomPoints by the randomSum entry
    ## so that each row lies on the unit simplex.
    onSimplex = np.divide(randomPoints.T,randomSum).T
    print(onSimplex)
    ## Verify on the simplex
    np.sum(onSimplex, axis=1) == np.ones(numSamples)
    ###########    
    return onSimplex


def uniformSampleSimplex(dim, numSamples):
    """
    This functions samples from the N-1 unit simplex (embedded in N-dimensional space)
    i.e. the polytope whose entries are nonnegative and sum to one.
    Currently no errors are thrown if the sample space or ambient space
    dimension are insufficient. But, sample size should be at least 1 and 
    ambient space dimension should be at least 2.
    """
    ambient_space_dimension = dim
    sample_size = numSamples
    random_sample = np.zeros((sample_size, ambient_space_dimension))

    for ROW in range(0,sample_size):
        # print(ROW)
        U = np.random.uniform(low=0.0, high=1.0, size=ambient_space_dimension)
        # print(U)
        E = -np.log(U)
        # print(E)
        S = np.sum(E)
        # print(S)
        X = E/S  
        # print(X)
        random_sample[ROW,] = X
  
    return random_sample




##############################################################################
def particleSwarmsObjectiveFunction_General(fracs, extraParamsPS):
    '''
    general function that will be used in the Particle Swarms algorithm. For each particle in the swarm, it first "repairs
     the particle and then evaluates the function objectiveFunctionICU_HospPeak. Depending on the given myIndex parameter
     will output the corresponding objective function.
    '''
    # print(fracs)
    [extraParams, myIndex] = extraParamsPS
    d = fracs.shape[0]
    myOutput = np.zeros(d)
    for ivals in range(d):
        fracVacs = fracs[ivals]
        constraintCheck = constraintCheckFunction(fracVacs)
        ## returns 1 if no change needed, 2 if nonnegative but high sum
        ## and 3 if not nonnegative.
        ## If no change is needed...then run model as-is with no change
        ## to fracs.
        if constraintCheck == 1:
            modelOutput = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
            myOutput[ivals] = (modelOutput[myIndex])
        else:                    
            fracVacs = repairVector(fracs[ivals], constraintCheck)
            fracs[ivals] = np.copy(fracVacs)
            modelOutput = objectiveFunctionICU_HospPeak(fracVacs, extraParams)
            myOutput[ivals] = modelOutput[myIndex]
    return myOutput





def coarseToPS2(myMat, numOfBestSols, numIterationsPS, numRandomPoints, proRataVac, extraParamsPS, numCores=1):
    """
    This function links the global search with the Particle Swarms search. It will select a number of bestSolutions
    (given by the variable numOfBestSols from the GS and use those solutions as initial particles for the swarm. In
    additionl it will generate a numRandomPoints feasible, random solutions that will also be used as the initial swarm.
    This function is different from coarseToPS because we will add the pro-rata as one of the particles for the swarms
    :param mymat: a matrix with all the combinations and outputs

    :return:
    WARNING: this is hard-coded to have dim=5 cells for VaccineGroups!!
    TO FIX: FIGURE OUT WHICH INDEX OF EXTRAPARAMS IT CORRESPONDS TO AND USE THAT.
    """

    #for reference:
    # extraParams = [mortality_rate_16, groupFracs, y0, numAgeGroups, numVaccinesAvailable, numVaccineGroups, paramsODE,
    #  totalPop16, totalPop5, tspan]


    [extraParams, myIndex] = extraParamsPS

    # create a list to store the results:
    results = []

    ## NUMBER OF ROWS IN OUTPUT.
    numRows = np.shape(myMat)[0]
    if numRows < numOfBestSols:
        print('numBestSols larger than input matrix')
        sys.exit()

    nm_matrix = np.zeros((numOfBestSols, 6))

    # sort the matrix by the myIndex column.
    ## CHECK IT IS THE CORRECT COLUMN!!! THIS WAS MADE
    ## IN THE COARSE FUNCTION
    ## BUT THE OTHER INDICES CORRESPOND TO THE RUNVACCINE FUNCTION!!!
    sortedMat = myMat[myMat[:, myIndex].argsort()]

    # select the numOfBestSols best based on the objective that we are looking at...
    # myBestSets is output of coarseGlobalSearch, the first five columns
    # are the decision variables fracVacs. So I need to select just
    # the first five columns of it.
    # myBestSets = sortedMat[:numOfBestSols, ]

    ## This prints the best values up to numOfBestSols
    # print(np.hstack((myBestSets[:, 0:11], myBestSets[:, columnIndex].reshape((numOfBestSols,1)))))

    ####################################################
    ####################################################
    ## HEREHERE
    ########### INPUT PS optimization here!  ###########
    ####################################################
    ####################################################

    lb = [0] * extraParams[5]
    ub = [1] * extraParams[5]
    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    ## 15 is the total deaths index.
    ## extraParamsPS = [extraParams, 15]
    ## kwargs allows us to pass additional parameters into the function.
    kwargs = {"extraParamsPS": extraParamsPS}

    myBestDecisionVariables = np.vstack((sortedMat[:numOfBestSols, 0:5], proRataVac.reshape(np.size(sortedMat[0, 0:5]))))
    # print(myBestDecisionVariables.shape)

    #set up the initial swarm:
    randomSample = uniformSampleSimplex(dim=extraParams[5], numSamples=numRandomPoints)
    flat1 = myBestDecisionVariables.flatten(order='C')
    flat2 = randomSample.flatten(order='C')
    flat3 = np.append(flat1, flat2)
    numCol = myBestDecisionVariables.shape[1]
    numRow = myBestDecisionVariables.shape[0] + randomSample.shape[0]
    initialSwarm = flat3.reshape(numRow, numCol)

    ## run the PS algorithm:
    optimizer = ps.single.GlobalBestPSO(n_particles=numOfBestSols + 1 + numRandomPoints,
                                        dimensions=extraParams[5],
                                        options=options,
                                        bounds=bounds,
                                        init_pos=initialSwarm,

                                        )

    output = optimizer.optimize(particleSwarmsObjectiveFunction_General,
                                iters=numIterationsPS, n_processes=numCores, **kwargs)
    return output


def coarseToNM2(myMat, numOfBestSols, numRandomPoints, proRataVac, extraParamsNM):
    """
    This function links the global search with the NM search. This funciton is different from coarseToPS because we will add the pro-rata as one of the particles for the swarms
    :param mymat: a matrix with all the combinations and outputs
    :param numOfInitialPoints: number of initial points we want to use for Nelder-Mead
    :return:
    WARNING: this is hard-coded to have dim=5 cells for VaccineGroups!!
    TO FIX: FIGURE OUT WHICH INDEX OF EXTRAPARAMS IT CORRESPONDS TO AND USE THAT.
    """
    [extraParams, myIndex] = extraParamsNM

    # create a list to store the results:
    results = []

    ## NUMBER OF ROWS IN OUTPUT.
    numRows = np.shape(myMat)[0]
    if numRows < numOfBestSols:
        print('numBestSols larger than input matrix')
        sys.exit()

    nm_matrix = np.zeros((numOfBestSols, 6))

    # sort the matrix by the myIndex column.
    sortedMat = myMat[myMat[:, myIndex].argsort()]

    # select the numOfBestSols best based on the objective that we are looking at...
    # myBestSets is output of coarseGlobalSearch, the first five columns
    # are the decision variables fracVacs. So I need to select just
    # the first five columns of it.
    myBestSets = sortedMat[:numOfBestSols, ]

    #select the first numOfBestSols solutions from the GS to be inputs as initial conditions for NM and add the pro-rata sol
    myBestDecisionVariables = np.vstack((sortedMat[:numOfBestSols, 0:5], proRataVac.reshape(np.size(sortedMat[0, 0:5]))))

    #select an additional numRandomPoints feasible solutions at random for initial conditions for NM
    randomSample = uniformSampleSimplex(dim=extraParams[5], numSamples=numRandomPoints)
    flat1 = myBestDecisionVariables.flatten(order='C')
    flat2 = randomSample.flatten(order='C')
    flat3 = np.append(flat1, flat2)
    numCol = myBestDecisionVariables.shape[1]
    numRow = myBestDecisionVariables.shape[0] + randomSample.shape[0]
    x0Mat = flat3.reshape(numRow, numCol)

    results = np.zeros(((numRandomPoints+numOfBestSols+1), 11))
    for ivals in range((numRandomPoints+numOfBestSols+1)):
        x0 = x0Mat[ivals]
        res = minimize(objectiveFunction_NM, x0, args=(extraParamsNM,), method='nelder-mead')
        results[ivals, 0:5] = res.x
        results[ivals, 5:10] = repairVector(res.x, constraintCheckFunction(res.x))
        results[ivals, 10] = res.fun

    sortedResults= results[results[:, 10].argsort()]
    return sortedResults[0,:]




##############################################################################
############################################################################################################################################################
############################################################################################################################################################
##############################################################################



def fromFracVacsToFracPop(fracVacs, numVaccineAvailable, totalPop5):
    '''converts the fraction of vaccine used in the vaccine groups to the fraction of the population that this represents in
    each of those vaccine groups'''
    dosesPerVaccineGroup = numVaccineAvailable * fracVacs
    fracPopVaccinated = np.array([dosesPerVaccineGroup[i]/totalPop5[i] for i in range(5)])
    return fracPopVaccinated



def pickAndStoreBestSols(myMat, numOfBestSols):

    myobjectives = ['totalInfections', 'totalSymptomaticInfections,', 'totalHospitalizations,', 'totalICU', 'deaths',  'hosp_peak', 'ICU_peak']


    func_dict_columnVals = {'totalInfections': 11,
                            'totalSymptomaticInfections': 12,
                            'totalHospitalizations': 13,
                            'totalICU': 14,
                            'deaths': 15,
                            'hosp_peak': 16,
                            'ICU_peak': 17}
    # create a list to store the results:
    results = []
    n1 = np.shape(myMat)[0]
    if n1 < numOfBestSols:
        print('numBestSols larger than input matrix')
        sys.exit()
    for keyvals in func_dict_columnVals:
        print(keyvals)
        nm_matrix = np.zeros((numOfBestSols, 6))
        columnIndex = func_dict_columnVals[keyvals]

        #sort the matrix by that column:
        sortedMat = myMat[myMat[:, columnIndex].argsort()]

        #select the N best based on the objective that we are looking at:
        mybestSets = sortedMat[:numOfBestSols, :]
        # temp = np.around(np.hstack((mybestSets[:, 0:11], mybestSets[:, columnIndex].reshape((numOfBestSols,1)))), decimals=2)
        temp = np.around(np.hstack((mybestSets[0, 0:11], mybestSets[0, columnIndex])),
                         decimals=2)
        print(temp)



def pickBestSolForEachObjective(myMat):
    """
    picks the optimal solution among all of the sols of myMat for each 
    of the objectives we care about and returns them
    in a matrix
    :param myMat:
    :param numOfBestSols:
    :return:
    """

    myobjectives = ['totalInfections', 'totalSymptomaticInfections', 'totalHospitalizations', 'totalICU', 'deaths',  'hosp_peak', 'ICU_peak']

    func_dict_columnVals = {'totalInfections': 11,
                            'totalSymptomaticInfections': 12,
                            'totalHospitalizations': 13,
                            'totalICU': 14,
                            'deaths': 15,
                            'hosp_peak': 16,
                            'ICU_peak': 17}
    #create a list to store the results:
    results = np.zeros((7, 12))
    for ivals in range(7):
        keyvals = myobjectives[ivals]
        # print(keyvals)
        columnIndex = func_dict_columnVals[keyvals]

        #sort the matrix by that column:
        sortedMat = myMat[myMat[:, columnIndex].argsort()]
        temp = np.around(np.hstack((sortedMat[0, 0:11], sortedMat[0, columnIndex])),
                         decimals=2)
        results[ivals, :] = temp
    return results

################################################################################
def convertToArraySimplex(readFileName, numVaccineGroups, lowerIndex, upperIndex, partition):
    '''
    Converts the text files with the arrays containing the partitions to pickle files with numpy arrays in them.
    '''
    myFracMat = np.zeros((upperIndex, numVaccineGroups))

    ## Note that "with open" automatically closes the file!
    with open(readFileName) as readPointFile:
        lines = readPointFile.readlines()
        for i in range(lowerIndex, upperIndex):
            fracVacsString = lines[i]
            test_str = fracVacsString.rstrip()
            myFracMat[i, :] = [float(idx) for idx in test_str.split(' ')]
       
    print(myFracMat.shape)
    # store the matrix as a pickle file in the outputfile
    myfilename = 'fracVacsDir/' + 'matrix_100_divided_by_' + str(partition) + '_with_' + str(numVaccineGroups) + '_vaccine_groups.pickle'
    print(myfilename)

    myfile = open(myfilename, 'wb')
    pickle.dump(myFracMat, myfile)
    myfile.close()
    
    return myFracMat

def skimOffExcessVaccine(fracVacs, numVaccinesAvailable,numVaccineGroups, totalPopByVaccineGroup):
    totalPopByVaccineGroup = np.floor(totalPopByVaccineGroup)
    numVaccinesByGroup = np.floor(np.multiply(fracVacs, numVaccinesAvailable))
    isExcessVaccineByGroup = np.zeros(numVaccineGroups)
    excessVaccine = 0

    for i in range(0, numVaccineGroups):
        if numVaccinesByGroup[i] > totalPopByVaccineGroup[i]:
            isExcessVaccineByGroup[i] = 1
    ## 1 = excess vaccine in that group
    ## 0 = no excess vaccine in that group

    # print(isExcessVaccineByGroup)

    ## skims off excess vaccine each group.
    if sum(isExcessVaccineByGroup) > 0:

        ## This finds the total number of excess vaccines.
        excessVaccine = np.dot(isExcessVaccineByGroup, (numVaccinesByGroup - totalPopByVaccineGroup))

        newNumVaccinesByGroup = numVaccinesByGroup - np.multiply(isExcessVaccineByGroup,
                                                                 (numVaccinesByGroup - totalPopByVaccineGroup))
        numVaccinesByGroup = newNumVaccinesByGroup
    return [excessVaccine, numVaccinesByGroup]



    

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
############################### START OF MAIN ##########################################################################
############################### START OF MAIN ##########################################################################
############################### START OF MAIN ##########################################################################
############################### START OF MAIN ##########################################################################
############################### START OF MAIN ##########################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == '__main__':
    np.set_printoptions(precision=3)


    ################### demographics and disease severity data :    ###################
    N = 7.615 * 10 ** (6)  # Washington state population

    # load contact matrices
    myfilename = '../data/consistentMatricesUS_polymodMethod01Jun2020.pickle'
    f = open(myfilename, 'rb')
    mymats = pickle.load(f)
    f.close()
    mymatAll = mymats['all']

    # load fractions in each age and vaccine group:
    myfilename = '../data/populationUS16ageGroups03Jun2020.pickle'
    f = open(myfilename, 'rb')
    popUS16 = pickle.load(f)
    f.close()
    popUS16fracs = popUS16[1]

    myfilename = '../data/populationUSGroupsForOptimization03Jun2020.pickle'
    f = open(myfilename, 'rb')
    groupInfo = pickle.load(f)
    f.close()
    groupFracs = groupInfo['groupsFracs']
    fracOfTotalPopulationPerVaccineGroup = groupInfo['fracOfTotalPopulationPerVaccineGroup']
    [relativeFrac75_80, relativeFrac80andAbove] = groupInfo['split75andAbove']

    # Split the population in 16 age groups:
    totalPop16 = N * popUS16fracs
    # Split the population in 5 vaccine groups:
    totalPop5 = N * fracOfTotalPopulationPerVaccineGroup

    numAgeGroups = 16
    numVaccineGroups = 5

    # load disease severity parameters:
    myfilename = '../data/disease_severity_parametersFerguson.pickle'
    f = open(myfilename, 'rb')
    diseaseParams = pickle.load(f)
    hosp_rate_16 = diseaseParams['hosp_rate_16']
    icu_rate_16 = diseaseParams['icu_rate_16']
    mortality_rate_16 = diseaseParams['mortality_rate_16']

    # this is just 1 - ICU rate useful to compute it in advance and pass it to the ODE
    oneMinusICUrate = np.ones(numAgeGroups) - icu_rate_16
    # this is just 1 - Hosp rate useful to compute it in advance and pass it to the ODE
    oneMinusHospRate = np.ones(numAgeGroups) - hosp_rate_16

    ###########################################################################################################
    ###########################################################################################################
    ############################ Parameters of the model ######################################################
    ###########################################################################################################
    ###########################################################################################################
    # Model parameters
    frac_sym = (1 - 0.35)*np.ones(16)  # fraction of infected that are symptomatic
    # frac_sym[0:4] = 0.2
    durI = 3  # duration of infectiousness after developing symptoms
    durP = 2  # duration of infectiousness before developing symptoms
    durA = durI + durP  # the duration of asymptomatic infections is equal to that of symptomatic infections
    gammaA = 1 / durA  # recovery rate for asymptomatic
    gammaH = 1 / 5  # recovery rate for hospitalizations (not ICU)
    gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)
    gammaICU = 1 / 10  # recovery rate for ICU hospitalizations
    gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
    gammaE = 1 / 3  # transition rate from exposed to infectious
    redA = 1  # reduction of infectiousness for asymptomatic infections
    redH = 0.  # reduction of infectiousness for hospitalized infections
    redP = 1.3  # this makes the fraction of infections attributed to pre-sym cases roughly equal to 40% at the peak for 2< R0<3

    oneMinusSymRate = np.ones(16) - frac_sym
    # print(oneMinusSymRate)
    # Disease severity
    R0 = 3

    #reduction in susceptibility: taken from Zhang, Science 2020
    red_sus = np.ones(16)
    red_sus[0:3] = 0.34
    red_sus[3:13] = 1
    red_sus[13:16] = 1.47


    # Disease severity
    R0 = 3

    # compute beta based on these parameters:
    beta = findBetaNewModel_eqs2_withHosp4(mymatAll, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaP,
                                           hosp_rate_16, redA, redH, redP, red_sus,  R0, totalPop16)

    # Vaccine efficacy:
    VE = 0.1

    paramsODE = [beta, mymatAll, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16,
                icu_rate_16, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus,
                 totalPop16, VE]








