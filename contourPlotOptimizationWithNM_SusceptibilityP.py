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
import multiprocessing as mp
import os
import pandas as pd
sys.path.insert(1, '../')


from optimizationFunctionsCoronavirus3 import coarseToNM2, coarseGlobalSearch2, pickBestSolForEachObjective
from optimizationFunctionsCoronavirus3 import constraintCheckFunction, repairVector, fromFracVacsToFracPop
from coronavirusMainFunctions import findBetaNewModel_eqs2_withHosp4





if __name__ == "__main__":
    mp.set_start_method('spawn')  # to make the pyswarms work in parallel
    start = timer()
    today = time.strftime("%d%b%Y", time.localtime())

    mytime = time.localtime()
    myseed = np.abs(int((np.random.normal())*1000000))
    np.random.seed(myseed)

    index = os.environ['SLURM_ARRAY_TASK_ID']

    VE = 0.1*int(index)


    #load the simplex matrix:
    readFileName = "fracVacsDir/matrix_100_divided_by_20_with_5_vaccine_groups.pickle"
    f = open(readFileName, 'rb')
    simplexMatrix = pickle.load(f)
    f.close()
    ncol = np.shape(simplexMatrix)[0]

    ################### Common parameters that will not vary from run to run:    ###################
    N = 7.615 * 10**(6) #Washington state population

    #load contact matrices
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


    #Split the population in 16 age groups:
    totalPop16 = N * popUS16fracs
    #Split the population in 5 vaccine groups:
    totalPop5 = N*fracOfTotalPopulationPerVaccineGroup

    numAgeGroups = 16
    numVaccineGroups = 5

    #load disease severity parameters:
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

    #time horizon for the intervention:
    tspan = np.linspace(0, 365, 365 * 2)




    ##################################################################################################################

    ######################## Parameters that will change for sensitivity analysis ####################################
    ########################################################################################################################
    ######################## Parameters that will change for sensitivity analysis ####################################
    # Model parameters

    #fraction of symptomatic people
    frac_asymptomatic = 0.35
    frac_sym = (1 - frac_asymptomatic) * np.ones(16)  # fraction of infected that are symptomatic
    #fraction of symptomatic children
    # frac_sym[0:4] = 0.2
    oneMinusSymRate = np.ones(16) - frac_sym
    # print(oneMinusSymRate)

    #transition rates:
    durI = 3  # duration of infectiousness after developing symptoms
    durP = 2  # duration of infectiousness before developing symptoms
    durA = durI + durP  # the duration of asymptomatic infections is equal to that of symptomatic infections
    gammaA = 1 / durA  # recovery rate for asymptomatic
    gammaH = 1 / 5  # recovery rate for hospitalizations (not ICU)
    gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)
    gammaICU = 1 / 10  # recovery rate for ICU hospitalizations
    gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
    gammaE = 1 / 3  # transition rate from exposed to infectious

    #reduction/increase of infectiousness
    redA = 1  # reduction of infectiousness for asymptomatic infections
    redH = 0.  # reduction of infectiousness for hospitalized infections
    redP = 1.3  # this makes the fraction of infections attributed to pre-sym cases roughly equal to 40% at the peak for 2< R0<3


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

    paramsODE = [beta, mymatAll, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16,
                icu_rate_16, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus,
                 totalPop16, VE]

    #fraction of the population assumed to be immune at the beginning of vaccination
    frac_rec = 0.2

    #number of current infections:
    currentInfections = 1000*popUS16fracs
    ##################################################################################################################


    for ivals in range(1,11):
        fracCov = 0.1*ivals  #vaccination coverage of the total population
        print(fracCov)
        numVaccinesAvailable = round(fracCov*N) #number of total vaccines available assuming that coverage

        # compute initial conditions
        S0 = (1 - frac_rec) * (totalPop16 - currentInfections)

        I0 = frac_sym * currentInfections
        E0 = np.zeros(numAgeGroups)
        A0 = (1 - frac_sym) * currentInfections
        P0 = np.zeros(numAgeGroups)
        H0 = np.zeros(numAgeGroups)
        ICU0 = np.zeros(numAgeGroups)
        Rec0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections), oneMinusHospRate)
        RecA0 = (1 - frac_sym) * frac_rec * (totalPop16 - currentInfections)
        RecH0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
                            np.multiply(hosp_rate_16, oneMinusICUrate))
        RecICU0 = np.multiply(frac_sym * frac_rec * (totalPop16 - currentInfections),
                              np.multiply(hosp_rate_16, icu_rate_16))

        # Vaccinated initial conditions
        V0 = np.zeros(numAgeGroups)
        E_V0, A_V0, P_V0, I_V0, H_V0, ICU_V0, RecV_0, RecAV_0, RecHV_0, RecICUV_0 = np.zeros(numAgeGroups), \
                                                                                    np.zeros(numAgeGroups), np.zeros(
            numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
                                                                                    np.zeros(numAgeGroups), np.zeros(
            numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)

        Cases0 = np.copy(I0) + np.copy(A0)
        # print(Cases0)
        # print(Cases0)

        y0 = np.array([S0, E0, A0, P0, I0, H0, ICU0, Rec0, RecA0, RecH0, RecICU0,
                       V0, E_V0, A_V0, P_V0, I_V0, H_V0, ICU_V0, RecV_0, RecAV_0, RecHV_0, RecICUV_0,
                       Cases0]).reshape((23 * numAgeGroups))

        extraParams = [mortality_rate_16, groupFracs, y0, numAgeGroups, numVaccinesAvailable, numVaccineGroups,
                       paramsODE,
                       totalPop16, totalPop5, tspan]

        #####################################          GRID SEARCH        ###########################################

        myMat = coarseGlobalSearch2(simplexMatrix, extraParams, 0, ncol)
        results = pickBestSolForEachObjective(myMat)
        print('coarse search done')


        #####################################   NM SEARCH   ############################################################
        # numIterationsPS = 50
        numRandomPoints = 25
        numOfBestSols = 25
        proRataVac = fracOfTotalPopulationPerVaccineGroup

        resultsPS = np.zeros((7, 16))

        for ivals in range(1,7):
            print(ivals)
            myIndex = 11 + ivals

            extraParamsPS = [extraParams, myIndex]
            output = coarseToNM2(myMat, numOfBestSols, numRandomPoints, proRataVac, extraParamsPS)

            resultsPS[ivals, 0:5] = output[0:5]
            resultsPS[ivals, 5:10] = output[5:10]
            resultsPS[ivals, 10] = output[10]
            resultsPS[ivals, 11:] = fromFracVacsToFracPop(output[5:10],
                                                          numVaccinesAvailable, totalPop5)


        myoutput = [results, resultsPS, R0, extraParams]
        # print(resultsPS)
    #     ##store the results
        myfilename = 'resultsOptimizationGSandPS/diff_susceptibility/NM/vaccine_efficacy_' + str(int(VE*100))  + \
                     '_frac_coverage_' + str(int(fracCov*100)) + '_meanParameters_' + '_fraction_already_recovered_' +\
                     str(int(frac_rec*100)) + '_childrenSusceptibility_' + str(int(red_sus[0]*100)) + '_50iterations_' + today + 'GSandNM.pickle'

        myfile = open(myfilename, 'wb')
        print(myfilename)
        pickle.dump(myoutput, myfile)
        myfile.close()

    myend = timer()
    print(myend - start)

        # print(results)


