import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../coronavirus_optimization')  # /coronavirusMainFunctions')
# import coronavirus_optimization/coronavirusMainFunctions')
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
import time

from coronavirusMainFunctionsP import  findBetaNewModel_eqs2_withHosp4
from optimizationFunctionsCoronavirus3P import runVaccination, splitVaccineAmongAgeGroups
# print(sys.path)

def createContourMatsProRata(all_other_params, disease_severity_params, init_cond_params, pop_params):
    #set initial conditions:
    [hosp_rate_16, icu_rate_16, mortality_rate_16, oneMinusHospRate, oneMinusICUrate] = disease_severity_params
    [currentInfections, frac_rec] = init_cond_params
    [fracOfTotalPopulationPerVaccineGroup, groupFracs, N, numAgeGroups, numVaccineGroups, totalPopByAgeGroup, totalPopByVaccineGroup] = pop_params
    [beta, contactMatrix, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, oneMinusSymRate, redA, redH, redP,red_sus, tspan] = all_other_params

    resultsTotalInfections, resultsTotalSymInfections = np.zeros((11, 11)), np.zeros((11, 11))
    resultsTotalHospitalizations, resultsTotalICU = np.zeros((11, 11)), np.zeros((11, 11))
    resultsDeaths, resultsHosp_peak, resultsICU_peak = np.zeros((11, 11)), np.zeros((11, 11)), np.zeros((11, 11))
    results = [resultsTotalInfections, resultsTotalSymInfections, resultsTotalHospitalizations, resultsTotalICU,
               resultsDeaths, resultsHosp_peak, resultsICU_peak]
    for vevals in range(11):
        VE = 0.1*vevals
        # print('VE', VE)
        for kvals in range(11):
            fracCov = 0.1 * kvals  # vaccination coverage of the total population
            print(fracCov)
            numVaccinesAvailable = round(fracCov * N)  # number of total vaccines available assuming that coverage
            # print(numVaccinesAvailable)
            # compute initial conditions
            S0 = (1 - frac_rec) * (totalPopByAgeGroup - currentInfections)

            I0 = frac_sym * currentInfections
            E0 = np.zeros(numAgeGroups)
            A0 = (1 - frac_sym) * currentInfections
            P0 = np.zeros(numAgeGroups)
            H0 = np.zeros(numAgeGroups)
            ICU0 = np.zeros(numAgeGroups)
            Rec0 = np.multiply(frac_sym * frac_rec * (totalPopByAgeGroup - currentInfections), oneMinusHospRate)
            RecA0 = (1 - frac_sym) * frac_rec * (totalPopByAgeGroup - currentInfections)
            RecH0 = np.multiply(frac_sym * frac_rec * (totalPopByAgeGroup - currentInfections),
                                np.multiply(hosp_rate_16, oneMinusICUrate))
            RecICU0 = np.multiply(frac_sym * frac_rec * (totalPopByAgeGroup - currentInfections),
                                  np.multiply(hosp_rate_16, icu_rate_16))

            # Vaccinated initial conditions
            V0 = np.zeros(numAgeGroups)
            E_V0, A_V0, P_V0, I_V0, H_V0, ICU_V0, RecV_0, RecAV_0, RecHV_0, RecICUV_0 = np.zeros(numAgeGroups), \
                                                                                        np.zeros(
                                                                                            numAgeGroups), np.zeros(
                numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups), \
                                                                                        np.zeros(
                                                                                            numAgeGroups), np.zeros(
                numAgeGroups), np.zeros(numAgeGroups), np.zeros(numAgeGroups)

            Cases0 = np.copy(I0) + np.copy(A0)
            # print(Cases0)
            # print(Cases0)
            # print(S0)

            y0 = np.array([S0, E0, A0, P0, I0, H0, ICU0, Rec0, RecA0, RecH0, RecICU0,
                           V0, E_V0, A_V0, P_V0, I_V0, H_V0, ICU_V0, RecV_0, RecAV_0, RecHV_0, RecICUV_0,
                           Cases0]).reshape((23 * numAgeGroups))

            paramsODE = [beta, contactMatrix, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hosp_rate_16,
                 icu_rate_16, numAgeGroups, oneMinusHospRate, oneMinusICUrate, oneMinusSymRate, redA, redH, redP, red_sus,
                 totalPopByAgeGroup, VE]

            extraParams = [mortality_rate_16, groupFracs, y0, numAgeGroups, numVaccinesAvailable, numVaccineGroups,
                           paramsODE,
                           totalPopByAgeGroup, totalPopByVaccineGroup, tspan]

            numVaccinesByGroup = fracOfTotalPopulationPerVaccineGroup * numVaccinesAvailable
            # print(numVaccinesByGroup)
            numVaccinesAgeGroup = splitVaccineAmongAgeGroups(numVaccinesByGroup, groupFracs, totalPopByAgeGroup)
            temp = runVaccination(mortality_rate_16, y0, numAgeGroups, numVaccinesAgeGroup, paramsODE, tspan)
            # print(temp)
            # tempprint(temp[1]/temp[0])
            for ivals in range(7):
                results[ivals][vevals, kvals] = temp[ivals]
    return results



################### Common parameters that will not vary from run to run:    ###################
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
pop_params = [fracOfTotalPopulationPerVaccineGroup, groupFracs, N, numAgeGroups, numVaccineGroups, totalPop16,
              totalPop5]
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

# time horizon for the intervention:
tspan = np.linspace(0, 365, 365 * 2)

##################################################################################################################

######################## Parameters that will change for sensitivity analysis ####################################
########################################################################################################################
######################## Parameters that will change for sensitivity analysis ####################################
# GENERATE THE PRO-RATA MATRICES AND SAVE THEM SO WE DON'T HAVE TO RUN IT EVERY TIME

# fraction of symptomatic people
frac_asymptomatic = 0.35
frac_sym = (1 - frac_asymptomatic) * np.ones(16)  # fraction of infected that are symptomatic
# fraction of symptomatic children
# frac_sym[0:4] = 0.2
oneMinusSymRate = np.ones(16) - frac_sym
print(oneMinusSymRate)

# transition rates:
durI = 3  # duration of infectiousness after developing symptoms
durP = 2  # duration of infectiousness before developing symptoms
durA = durI + durP  # the duration of asymptomatic infections is equal to that of symptomatic infections
gammaA = 1 / durA  # recovery rate for asymptomatic
gammaH = 1 / 5  # recovery rate for hospitalizations (not ICU)
gammaI = 1 / durI  # recovery rate for symptomatic infections (not hospitalized)
gammaICU = 1 / 10  # recovery rate for ICU hospitalizations
gammaP = 1 / durP  # transition rate fromm pre-symptomatic to symptomatic
gammaE = 1 / 3  # transition rate from exposed to infectious

# reduction/increase of infectiousness
redA = 1  # reduction of infectiousness for asymptomatic infections
redH = 0.  # reduction of infectiousness for hospitalized infections
redP = 1.3  # this makes the fraction of infections attributed to pre-sym cases roughly equal to 40% at the peak for 2< R0<3

# Disease severity
R0 = 3
# reduction in susceptibility
red_sus = np.ones(16)
red_sus[0:3] = 0.34
red_sus[3:13] = 1
red_sus[13:16] = 1.47
# compute beta based on these parameters:
beta = findBetaNewModel_eqs2_withHosp4(mymatAll, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaP, hosp_rate_16,
                                       redA, redH, redP, red_sus, R0, totalPop16)

disease_severity_params = [hosp_rate_16, icu_rate_16, mortality_rate_16, oneMinusHospRate, oneMinusICUrate]

# fraction of the population assumed to be immune at the beginning of vaccination
frac_rec = 0.2
childrenSus = red_sus[0]*100

# number of current infections:
currentInfections = 1000 * popUS16fracs

init_cond_params = [currentInfections, frac_rec]

all_other_params = [beta, mymatAll, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, oneMinusSymRate,
                    redA, redH, redP, red_sus, tspan]
proRataMats = createContourMatsProRata(all_other_params, disease_severity_params, init_cond_params, pop_params)
proRataResults = [proRataMats, all_other_params, disease_severity_params, init_cond_params, pop_params]

print(proRataMats)

# #save this so I don't have to re-run it every time:
# today = '02Jul2020'
# myfilename = 'results/prorataMats' + \
#         '_meanParameters_frac_recovered_' + str(int(100*frac_rec)) + '_childrenSusceptibility_' + str(childrenSus)  +\
#                  '__numIterationsPS50_' + today + '.pickle'
# myfile = open(myfilename, 'wb')
# pickle.dump(proRataResults, myfile)
# myfile.close()