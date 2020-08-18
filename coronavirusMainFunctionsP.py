import numpy as np
from scipy.integrate import odeint
# import seaborn as sns
# from matplotlib import pyplot as plt
import pickle
# from matplotlib.colors import ListedColormap
# %matplotlib qt
# from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
# mycolors = sns.color_palette("hls", 20)
# twoColorPalette = sns.color_palette("RdBu_r", 2)
#
# fourColorPalette = sns.color_palette("Paired", 8)
# fiveColorPalette = sns.color_palette("husl", 5)
# twoColorPalette1 = [fourColorPalette[1], fourColorPalette[3]]#sns.color_palette("Paired", 2)
#
# contourPalette = sns.color_palette(sns.color_palette("coolwarm", 25))
# contourPalette1 = sns.color_palette("BrBG", 25)





def findBetaNewModel_eqs2_withHosp4(C, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaP, hosp_rate, redA, redH, redP, red_sus,  R0, totalPop):
    #compute the value of beta for model described in coronavirusEqs2_withHospitalizationsAndICU_withVaccine4
    #here, frac_sym is a vector not a scalar and there is a different beta for each age group representing a different
    #susceptibility
    # compute the eignevalues of F*V^(-1) assuming the infected states are 5, namely: E, A, P, I, D
    [n1, n2] = np.shape(C)

    # create F
    N = np.sum(totalPop)
    Z = np.zeros((n1, n1))
    C1 = np.zeros((n1, n1))
    for ivals in range(n1):
        for jvals in range(n1):
            C1[ivals, jvals] = red_sus[ivals] * C[ivals, jvals] * totalPop[ivals]/totalPop[jvals]


    #create F by concatenating different matrices:
    F1 = np.concatenate((Z, redA*C1, redP*C1, C1, redH*C1), 1)
    F2 = np.zeros((4*n1, 5*n1))

    F = np.concatenate((F1, F2), 0)

    #create V
    VgammaE = np.diag(gammaE * np.ones(n1))
    VgammaA = np.diag(gammaA * np.ones(n1))
    VgammaP = np.diag(gammaP * np.ones(n1))
    VgammaI = np.diag(gammaI * np.ones(n1))
    VgammaH = np.diag(gammaH * np.ones(n1))

    Vsub1 = np.diag(-(np.ones(n1)-frac_sym) * gammaE)
    Vsub2 = np.diag(-(frac_sym) * gammaE)

    Vsub3 = np.diag(-(np.ones(n1) - hosp_rate) * gammaP)
    Vsub4 = np.diag(-(hosp_rate) * gammaP)
    # print(V)

    V1 = np.concatenate((VgammaE, Z, Z, Z, Z), 1)
    V2 = np.concatenate((Vsub1, VgammaA, Z, Z, Z), 1)
    V3 = np.concatenate((Vsub2, Z, VgammaP, Z, Z), 1)
    V4 = np.concatenate((Z, Z, Vsub3, VgammaI, Z), 1)
    V5 = np.concatenate((Z, Z, Vsub4, Z, VgammaI), 1)

    V = np.concatenate((V1, V2, V3, V4, V5), 0)
    # print(np.linalg.inv(V))

    myProd = np.dot(F, np.linalg.inv(V))
    # print(myProd)
    myEig = np.linalg.eig(myProd)
    # print(myEig)
    largestEig = np.max(myEig[0])
    if largestEig.imag == 0.0:

        beta = R0 / largestEig.real
        # print('beta', beta)
        return beta
    else:
        print(largestEig)
        raise Exception('largest eigenvalue is not real')





def coronavirusEqs2_withHospitalizationsAndICU_withVaccine4(y, t, params):
    """
    different from coronavirusEqs2_withHospitalizationsAndICU_withVaccine
    because frac_sym: fraction symptomatic that will be different for each age group, hence it will be a vector
    and I will include a vector of different susceptibilities for each age group
    coronavirus equations with asymptomatics, pre-symptomatic infected, infected symptomatics hospitalized, infected hospitalized in the ICU and infected
    symptomatic not-hospitalized
    infected in the ICU are assumed not to infect anyone
    USING A MATRIX OF CONTACTS THAT IS NOT SYMMETRIC and vaccination
    :param y: vector of the current state of the system
    :param t: time
    :param params: all the params to run the ODE, defined below
    beta: rate of infection given contact
    C: contact matrix across age groups

    gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP: transition rates out of the asymptomatic, exposed,
    infectioussymptomatic hospitalized, infectioussymptomatic non-hospitalized, infectioussymptomatic hospitalized in the ICU,
    infectiouspre-symptomatic
     presymptomatic classes
     hospRate, ICUrate: rates of hospitalization in each age group
    numGroups: number of groups in the simualation
    redA,  redP: reduction in the infectiousness for asymptomatic and pre-symptomatic
    totalPop: a vector of size 1x numGroups with the population in each of the age groups.
    VE: vaccine efficacy
    :return:
    """
    [beta, C, frac_sym, gammaA, gammaE, gammaH, gammaI, gammaICU, gammaP, hospRate, ICUrate,  numGroups,
     oneMinusHospRate, oneMinusICUrate, oneMinusSympRate, redA, redH, redP, red_sus, totalPop, VE] = params

    #beta: infection rate
    #C: matrix of contact rates, c_ij represents the contact rate between group i and group j
    temp = np.reshape(y, (23, numGroups))


    S = temp[0, :]  #susceptibles
    E = temp[1, :]  #Exposed
    A = temp[2, :]  #Asymptomatic infected
    P = temp[3, :]  # Pre-symptomatic infected
    I = temp[4, :]  #Symptomatic infe11,12cted
    H = temp[5, :]  #Hospitalized non ICU
    ICU = temp[6, :]  #Hospitalized ICU
    R = temp[7, :]  #Recovered symptomatic
    RA = temp[8, :] #Recovered Asymptomatic
    RH = temp[9, :] #recovered hospitalized
    RC = temp[10, :] #recovered hospitalized ICU

    S_V = temp[11, :]  # susceptibles vaccinated
    E_V = temp[12, :]  # Exposed vaccinated
    A_V = temp[13, :]  # Asymptomatic infected vaccinated
    P_V = temp[14, :]  # Pre-symptomatic infected vaccinated
    I_V = temp[15, :]  # Symptomatic infected vaccinated
    H_V = temp[16, :]  # Hospitalizes symptomatic infected vaccinated
    ICU_V = temp[17, :]  # Hospitalized ICU vaccinated
    R_V = temp[18, :]  # Recovered symptomatic vaccinated
    RA_V = temp[19, :]  # Recovered Asymptomatic vaccinated
    RH_V =temp[20,:] #Recovered Hospitalized vaccinated
    RC_V = temp[21, :]  # recovered hospitalized ICU

    totalInf = temp[22, :]   #total cummulative infections

    Cnew = np.multiply(C, red_sus[:, np.newaxis])  #
    mylambda = np.dot(Cnew, beta * np.divide((redA * (A + A_V) +
                                           redP * (P + P_V) +
                                           redH * (H + H_V) +
                                           (I + I_V)), totalPop))  # force of infection

    dS = - np.multiply(mylambda, S)
    dE = np.multiply(mylambda, S) - gammaE * E
    dA = gammaE * np.multiply(E, oneMinusSympRate) - gammaA * A
    dP = gammaE * np.multiply(E, frac_sym) - gammaP * P
    dI = gammaP * np.multiply(P, oneMinusHospRate) - gammaI * I
    dH = gammaP * np.multiply(P, np.multiply(hospRate, oneMinusICUrate)) - gammaH * H
    dICU = gammaP * np.multiply(P, np.multiply(hospRate, ICUrate)) - gammaICU * ICU
    dR = gammaI * (I)
    dRA = gammaA * A
    dRH = gammaH * H
    dRICU = gammaICU * ICU


    # vaccinated equations
    dS_V = - (1 - VE) * np.multiply(mylambda, S_V)
    dE_V = (1 - VE) * np.multiply(mylambda, S_V) - gammaE * E_V
    dA_V = (1 - frac_sym) * gammaE * E_V - gammaA * A_V
    dP_V = frac_sym * gammaE * E_V - gammaP * P_V
    dI_V = gammaP * np.multiply(P_V, oneMinusHospRate) - gammaI * I_V
    dH_V = gammaP * np.multiply(P_V, np.multiply(hospRate, oneMinusICUrate)) - gammaH * H_V
    dICU_V = gammaP * np.multiply(P_V, np.multiply(hospRate, ICUrate)) - gammaICU * ICU_V
    dR_V = gammaI * I_V
    dRA_V = gammaA * A_V
    dRH_V = gammaH * H_V
    dRICU_V = gammaICU * ICU_V

    dtotalInf = np.multiply(mylambda, S) + (1 - VE) * np.multiply(mylambda, S_V)

    dydt = np.array([dS, dE, dA, dP, dI, dH, dICU, dR, dRA, dRH, dRICU,
                     dS_V, dE_V, dA_V, dP_V, dI_V, dH_V, dICU_V, dR_V, dRA_V, dRH_V, dRICU_V, dtotalInf]).reshape((numGroups * 23))
    return dydt




