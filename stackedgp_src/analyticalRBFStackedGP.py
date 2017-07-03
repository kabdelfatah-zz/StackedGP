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
 * analyticalRBFStackedGP.py
 * 
 *--------------------------------------------------------------------------
 *-------------------------------------------------------------------------- 
 '''
import numpy as np
from util import Util

class AnalyticalRBF():

    @staticmethod
    def getSqrtRoot(lengthscaleList, predictedVar):
        """
        :param lengthscaleList: lengthscale list from the secondGP with length M where M is the number of inputs to second GP
        :param predictedVar: Predicted variance from the first layer Nx1 where N is the number of predicted points
        :return: np.array of NXM. Each element is represent sqrt in the mean equation
                 np.array of NXM. Each element is represent sqrt in the variance equation
        """
        theta = 1./(2.*np.square(lengthscaleList))
        theta2_inv = (1./(2.*theta)).reshape(1,-1)
        theta4_inv = (1./(4.*theta)).reshape(1,-1)
        theta2_inv_matx = np.repeat(theta2_inv,predictedVar.shape[0], axis=0)
        theta4_inv_matx = np.repeat(theta4_inv,predictedVar.shape[0], axis=0)
        roottheta2 = np.sqrt(theta2_inv_matx / (predictedVar + theta2_inv_matx))
        roottheta4 = np.sqrt(theta4_inv_matx / (predictedVar + theta4_inv_matx))
        return roottheta2, roottheta4

    @staticmethod
    def getExpFactors(lengthscaleList, trainingData,predictedMean, predictedVar):
        """
        :param lengthscaleList:lengthscale list from the secondGP with length M where M is the number of inputs to second GP
        :param trainingData: Training data for the secondlayer GP. It should be FxM where F is the number of training points
        :param predictedMean: Predicted Mean from the first layer NxM where N is the number of predicted points
        :param predictedVar:Predicted variance from the first layer NxM where N is the number of predicted points
        :return:
        """
        N = predictedVar.shape[0]
        M = trainingData.shape[1]
        F = trainingData.shape[0]
        Util.debug('num Testing points='+str(N))
        Util.debug('num Training points='+str(F))
        Util.debug('num dimensions='+str(M))
        # Create demneator NxM
        theta = 1./(2.*np.square(lengthscaleList))
        theta2_inv = (1./(2.*theta)).reshape(1,-1)
        theta4_inv = (1./(4.*theta)).reshape(1,-1)
        theta2_inv_matx = np.repeat(theta2_inv,predictedVar.shape[0], axis=0)
        theta4_inv_matx = np.repeat(theta4_inv,predictedVar.shape[0], axis=0)
        demenator = 2.*(predictedVar + theta2_inv_matx)
        demenator2 = 2.*(predictedVar + theta4_inv_matx)

        # Create array with NxFxM
        num = np.zeros((N,F,M))
        fact2 = np.zeros((N,F,F,M))
        fact3 = np.zeros((N,F,F,M))
        for i in range(M):
            rtrain = np.repeat(trainingData[:,i].reshape(1,-1), F, axis=0)
            # ctrain = np.repeat(trainingData[:,i], F, axis=1)
            # trainSubFact is FxF
            trainSubFact = np.square(rtrain - rtrain.T)/ (np.square(lengthscaleList[i])*4.)
            # trainSumFact is FxF
            trainSumFact = (rtrain +rtrain.T) / 2.
            for j in range(N):
                num[j,:,i] = np.square(trainingData[:,i].reshape(1,-1) - np.ones((1,F))* predictedMean[j,i]) / demenator[j,i]
                # fact2[j,:,:,i] is FxF
                fact2[j,:,:,i] = (np.square(trainSumFact - np.ones((F,F))* predictedMean[j,i]) / demenator2[j,i]) \
                                 + trainSubFact
                rnum = np.repeat(num[j,:,i].reshape(1,-1), F, axis=0)
                fact3[j,:,:,i] = rnum + rnum.T
        # sum over M and give you NxF
        expfact = np.exp(-np.sum(num, axis=2))    # NxF
        # sum over M and give you NxFxF
        expfact2 = np.exp(-np.sum(fact2, axis=3)) # NxFxF
        # sum over M and give you NxFxF
        expfact3 = np.exp(-np.sum(fact3, axis=3)) # NxFxF
        return expfact, expfact2, expfact3

    @staticmethod
    def getCov(gp, jitter=0.0, include_covnoise=True):
        if include_covnoise:
            return gp.kern.K(gp.X) + np.eye(gp.X.shape[0]) * gp.Gaussian_noise[0] + np.eye(gp.X.shape[0]) * jitter
        else:
            return gp.kern.K(gp.X) + np.eye(gp.X.shape[0]) * jitter


    @staticmethod
    def predict(targetNode,tm,tv, jitter=0, covoption=1,include_likelihood=True, include_covnoise=True):
        # number of training points for this node
        F = targetNode.X.shape[0]
        # N is the number of testing points
        N = tm.shape[0]
        # Get roots
        Util.debug('Getting Root factors...')
        ll = targetNode.kern.lengthscale
        if len(ll) != targetNode.X.shape[1]:
            ll = np.repeat([ll],targetNode.X.shape[1],axis=0).reshape(-1,1)
        roottheta2, roottheta4 = AnalyticalRBF.getSqrtRoot(ll, tv )
        #Get exp factors
        Util.debug('Getting Exp factors...')
        expfact1, expfact2, expfact3 = AnalyticalRBF.getExpFactors(ll,targetNode.X, tm, tv )

        cov_inv = Util.getInv(AnalyticalRBF.getCov(targetNode, jitter=jitter,include_covnoise=include_covnoise), option=covoption)
        # cov_inv2 = asgp.getInv(asgp.getCov(targetNode, jitter=jitter), option=1)

        ### analytical mean#####
        Util.debug('Predicting mean...')
        mm = np.dot(expfact1, cov_inv.T)
        mmm = np.dot(targetNode.Y.T, mm.T)
        root2_prod = np.prod(roottheta2, axis=1).reshape(-1,1)
        mean = (targetNode.kern.variance * root2_prod*mmm.T).reshape(-1,1)
        # print 'Mean (min/max): ',mean.min(), mean.max()

        # ======= Analytical variance form ==========================
        Util.debug('Predicting variance...')
        # roottheta4_prod is Nx1
        roottheta4_prod = np.prod(roottheta4, axis=1).reshape(-1,1)
        roottheta4_prod_NxF = np.repeat(roottheta4_prod, F, axis=1)
        roottheta4_prod_NFxF = np.repeat(roottheta4_prod_NxF.reshape(-1,1), F, axis=1)
        roottheta4_prod_NxFxF = roottheta4_prod_NFxF.reshape((N,F,F))
        ####==================================================================
        ####==================================================================
        roottheta2_prod = np.prod(np.square(roottheta2), axis=1).reshape(-1,1)
        roottheta2_prod_NxF = np.repeat(roottheta2_prod, F, axis=1)
        roottheta2_prod_NFxF = np.repeat(roottheta2_prod_NxF.reshape(-1,1), F, axis=1)
        roottheta2_prod_NxFxF = roottheta2_prod_NFxF.reshape((N,F,F))

        #=======
        sigma = np.square(targetNode.kern.variance[0]) *(roottheta4_prod_NxFxF * expfact2 - roottheta2_prod_NxFxF * expfact3)
        vfact = np.zeros((1,N))
        vfact2 = np.zeros((1,N))
        for i in range(N):
            vfact[0,i] = np.dot(np.dot(np.dot(np.dot(targetNode.Y.T, cov_inv) , sigma[i,:,:] ), cov_inv) , targetNode.Y)
            vfact2[0,i] = np.sum(cov_inv * expfact2[i,:,:]) * roottheta4_prod[i,0]* np.square(targetNode.kern.variance[0])
        if include_likelihood:
            var = (targetNode.kern.variance[0] + targetNode.Gaussian_noise[0] + vfact - vfact2).reshape(-1,1)
        else:
            var = (targetNode.kern.variance[0] + vfact - vfact2).reshape(-1,1)
        # var = (ffsgp.layer2GP.kern.variance[0]  + vfact - vfact2).reshape(-1,1)
        return mean,var