'''
 *--------------------------------------------------------------------------
 *--------------------------------------------------------------------------
 *
 * Copyright (C) 2017 Kareem Abdelfatah - krabea@email.sc.edu
 *
 * The main applications of the StackedGP framework are to integrate different datasets through model composition, 
 * enhance predictions of quantities of interest through a cascade of intermediate predictions, and
 * to propagate uncertainties through emulated dynamical systems driven by uncertain forcing variables. 
 * By using analytical first and second-order moments of a Gaussian process as presented in the 
 * following paper:
 * 
 * Kareem Abdelfatah, Junshu Bao, Gabriel Terejanu (2017). 
 Environmental Modeling Framework using Stacked Gaussian Processes. arXiv:1612.02897v2 . 18 Jun 2017
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the author.
 *
 *--------------------------------------------------------------------------
 *
 * demo_sgp_1_l3f3s1t1_ica_cv.py
 * 
 *--------------------------------------------------------------------------
 *-------------------------------------------------------------------------- 
 '''
# import
import sys
sys.path.append('../')
sys.path.append('../../stackedgp_src/')

import numpy as np
from stackedGPNetwork import StackedGPNetwork
# from analyticalStackedGP_v0_1 import AnalyticalStackedGP as asgp
from sklearn.cross_validation import KFold
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import GPy

def loadData(fname='../data/forestfires.csv', ncols=14):
    weekdic = {'mon': '1','tue': '2','wed': '3','thu': '4','fri': '5','sat': '6','sun': '7'}
    monthdic = {'mar': '3','oct': '10','aug': '8','sep': '9','apr': '4','jun': '6','jul': '7','feb': '2','jan': '1','may': '5','nov': '11','dec': '12'}
    with open(fname) as f:
        lines = f.readlines()
        data = np.zeros((len(lines),ncols))
        indx = 0
        for line in lines:
            if line.startswith('#'):
                continue
            v = line.strip().split(',')
#             print v[2], v[3]
            v[2] = monthdic[v[2]]
            v[3] = weekdic[v[3]]
            data[indx, :-1] = map(float,v)
            data[indx, -1] = np.log(data[indx, -2]+1)
            indx +=1
        f.close()
    return data

def loadData2(fname='../data/candaninIndexes.csv', ncols=12):
    with open(fname) as f:
        lines = f.readlines()
        data = np.zeros((len(lines),ncols))
        indx = 0
        for line in lines:
            if line.startswith('#'):
                continue
            v = line.strip().split(',')
            data[indx, :] = map(float,v)
            indx +=1
        f.close()
    return data

data = loadData()
# load data for Canadian Indexes
data2 = loadData2()
kf = KFold(n=data.shape[0], n_folds=10, shuffle=True,random_state=10)
X = data[:,:-2]
Y = data[:,-2:-1]
logY = np.log(Y+1)
# scaler = StandardScaler()
# logY = scaler.fit_transform(logY)
# logY = data[:,-1:]
kfindex = 0
# errors =np.array([0.,0.,0.,0.,0.,0.])
errors =np.array([0.,0.])
totalTestedPoints = 0.
outInterval = 0.
totalTestedPoints2 = 0.
outInterval2 = 0.
errors2 =np.array([0.,0.])
varmax = -1000
varmin = 1000
varmax2 = -1000
varmin2 = 1000

for train_index, test_index in kf:
    print 'Creating Stacked Network...'
    #Date	Temp	RH	WIND	RAIN	FFMC	DMC	DC	ISI	BUI	FWI	DSR
    # X[:,4:8] = np.log(X[:,4:8]+1)
    trainX,testX = X[train_index], X[test_index]
    # data2[:,5:9] = np.log(data2[:,5:9]+1)
    xdata = np.concatenate((data2[:,5:9],trainX[:,4:8]),axis=0)
    ica = FastICA()
    xdata = ica.fit_transform(xdata)
    testX[:,4:8] = ica.transform(testX[:,4:8])
    data2[:,5:9] = xdata[:data2.shape[0],:]
    trainX[:,4:8] = xdata[data2.shape[0]:,:]
    stackedNetwork = StackedGPNetwork(3)
    # add FFMC node
    stackedNetwork.createNewNode(0,data2[:,1:5],data2[:,5:6], normalize=True, useGPU=False)
    # add DMC node
    stackedNetwork.createNewNode(0,data2[:,[1,2,4]],data2[:,6:7], normalize=True, useGPU=False)
    # add DC node
    stackedNetwork.createNewNode(0,data2[:,[1,4]],data2[:,7:8], normalize=True, useGPU=False)
    # add ISI node
    stackedNetwork.createNewNode(1,data2[:,[3,5]],data2[:,8:9], normalize=True, useGPU=False)
    # add BUI node
    # stackedNetwork.createNewNode(1,data2[:,[6,7]],data2[:,9:10], normalize=True, useGPU=False)
    # # add FWI node
    # stackedNetwork.createNewNode(2,data2[:,[8,9]],data2[:,10:11], normalize=True, useGPU=False)
    # create dumy node in the layer 2
    stackedNetwork.createNewNode(2,data2[:,8:9],data2[:,9:10], normalize=True, useGPU=False)
    print 'Optimizing Model...'
    stackedNetwork.optimize(numoptimizationtrails=1)
    # try:
    print '==========================================================================='
    print '==========================================================================='
    print '===============================================>>KF index = ',kfindex
    ##X	Y	month	day	FFMC	DMC	DC	ISI	temp	RH	wind	rain	arear	ln(area+1)
    ntrain = data.shape[0]

    trainY,testY = logY[train_index], Y[test_index]
    # create the second layer node with three inputs
    inputSecondGP = trainX[:,4:8]
    kernel1 = GPy.kern.RBF(input_dim=inputSecondGP.shape[1], ARD=True,useGPU=False)
    gp = GPy.models.GPRegression(inputSecondGP, trainY,kernel=kernel1, normalizer=True)
    gp.optimize_restarts(1)
    stackedNetwork.setNode(2,0,gp)

    #precition of the first layer
    print 'Predicting from the first layer...'
    fdata = np.concatenate((testX[:,8:12],testX[:,[8,9,11]],testX[:,[8,11]]),axis=1)
    fmean, fvar = stackedNetwork.predictLayer(0,fdata,None,include_likelihood=True)
    sdata = np.concatenate((testX[:,[3]],fmean[:,[0]]),axis=1)
    stv = np.concatenate((np.zeros(testX[:,[3]].shape),fvar[:,[0]]),axis=1)
    mean, var = stackedNetwork.predictLayer(1,sdata,stv, jitter=1e-1, covoption=1, include_likelihood=True,include_covnoise=True)

    tdata = np.concatenate((fmean,mean),axis=1)
    ttv = np.concatenate((fvar,var),axis=1)
    # fdata = np.concatenate((fdata,testX[:,[8,11]]),axis=1)
    pmean, pvar = stackedNetwork.predictLayer(2,tdata,ttv, jitter=1e-1, covoption=1, include_likelihood=True,include_covnoise=True)
    # mean = pmean[-1]
    # # mean = scaler.inverse_transform(mean)
    # var = pvar[-1]
    #=========================================================================
    #=========================================================================
    kfindex += 1
    print 'Applying log-normal distribution inverse...'
    # E_af = mean
    # varun = var
    E_af = np.exp(mean+var/2)-1
    varun = (np.exp(var) -1)*np.exp(2*mean + var)
    maxInter95 = E_af + 2*np.sqrt(varun)
    minInter95 = E_af - 2*np.sqrt(varun)
    for t,mi,mx in zip(testY,minInter95,maxInter95):
        totalTestedPoints += 1
        if t < mi or t > mx:
            outInterval += 1
    rmse = np.sqrt(np.sum(np.square(E_af-testY))/testY.shape[0])
    mad = np.sum(abs(E_af-testY))/testY.shape[0]
    print 'RMSE: ', rmse , 'MAD: ', mad
    pstd = np.sqrt(varun)
    print 'Predicted STD: ', pstd.min(), pstd.max()
    if varmin > pstd.min():
        varmin = pstd.min()
    if varmax < pstd.max():
        varmax = pstd.max()
    errors += np.array([rmse,mad])
    print '---------------->avgerages([RMSE,MAE]): \n',errors / float(kfindex)
    print '------------Total Tested points: ', totalTestedPoints, '------Out: ', outInterval, '---%', outInterval/totalTestedPoints
    print 'varmin / varmax: ',varmin, varmax
    #=========================================================================
    #=========================================================================
    print '======================================================'
    print '**********Single GP:**********************************'
    m,v = gp.predict(testX[:,4:8])
    # m = scaler.inverse_transform(m)
    print 'Applying log-normal distribution inverse...'
    E_af = np.exp(m+v/2)-1
    varun = (np.exp(v) -1)*np.exp(2*m + v)
    # E_af = m
    # varun = v
    maxInter95 = E_af + 2*np.sqrt(varun)
    minInter95 = E_af - 2*np.sqrt(varun)
    for t,mi,mx in zip(testY,minInter95,maxInter95):
        totalTestedPoints2 += 1
        # print mi,t,mx
        if t < mi or t > mx:
            outInterval2 += 1
    # print 'Mean (min/max): ',E_af.min(), E_af.max()
    rmse = np.sqrt(np.sum(np.square(E_af-testY))/testY.shape[0])
    mad = np.sum(abs(E_af-testY))/testY.shape[0]
    print 'RMSE: ', rmse, 'MAD: ', mad
    pstd = np.sqrt(varun)
    print 'Predicted STD: ', pstd.min(), pstd.max()
    if varmin2 > pstd.min():
        varmin2 = pstd.min()
    if varmax2 < pstd.max():
        varmax2 = pstd.max()
    errors2 += np.array([rmse,mad])
    print '---------------->avgerages([RMSE,MAE]): \n',errors2 / float(kfindex)
    print '******************************************************'
    print '======================================================'