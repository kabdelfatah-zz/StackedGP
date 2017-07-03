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
 * stackedGPNetwork.py
 * 
 *--------------------------------------------------------------------------
 *-------------------------------------------------------------------------- 
 '''
from util import Util
import GPy
from analyticalRBFStackedGP import AnalyticalRBF as aRBF
import numpy as np
import sys

class StackedGPNetwork():
    def __init__(self, nlayers):
        self.layers = []
        for i in range(nlayers):
            self.layers.append([])

    def createNewNode(self,layerIndx, inputdata, outputdata, kernel=None,normalize=False,useGPU=False,ARD=False):
        """

        :param layerIndx: The index is one based to be more convenient.
        :param inputdata:
        :param outputdata:
        :param kernel: kernel for gp. it can be a mix of kernels. If kernel is None, RBF kernel will be used as default.
        :param normalize:
        :param useGPU:
        :return:
        """
        if kernel is None:
            kernel = GPy.kern.RBF(input_dim=inputdata.shape[1], ARD=ARD,useGPU=useGPU)
        gp = GPy.models.GPRegression(inputdata, outputdata,kernel=kernel, normalizer=normalize)
        self.layers[layerIndx].append(gp)

    def _optimizeGP(self,gp, numoptimizationtrails=1,optimizer=None):
        """

        :param gp:
        :param numoptimizationtrails:
        :param optimizer:  Valid optimizers are:
          - 'scg': scaled conjugate gradient method, recommended for stability.
                   See also GPy.inference.optimization.scg
          - 'fmin_tnc': truncated Newton method (see scipy.optimize.fmin_tnc)
          - 'simplex': the Nelder-Mead simplex method (see scipy.optimize.fmin),
          - 'lbfgsb': the l-bfgs-b method (see scipy.optimize.fmin_l_bfgs_b),
          - 'lbfgs': the bfgs method (see scipy.optimize.fmin_bfgs),
          - 'sgd': stochastic gradient decsent (see scipy.optimize.sgd). For experts only!
          - 'tnc'
        :return:
        """
        if optimizer is None:
            optimizer = 'lbfgs'
        if numoptimizationtrails < 2:
            gp.optimize(optimizer=optimizer)
        else:
            gp.optimize_restarts(num_restarts=numoptimizationtrails,messages=False,optimizer=optimizer)

    def optimize(self, numoptimizationtrails=1,optimizer=None):
        for i in range(len(self.layers)):
            Util.debug('Layer......'+str(i+1),debuglevel=2)
            for j in range(len(self.layers[i])):
                Util.debug('Node......'+str(j+1),debuglevel=2)
                self._optimizeGP(self.layers[i][j], numoptimizationtrails,optimizer=optimizer)
    def getLayer(self, layerIndex):
        """

        :param layerIndex: The index is zero based to be more convenient
        :return:
        """
        return self.layers[layerIndex]
    def getNode(self, layerIndex, nodeIndex):
        """

        :param layerIndex: The index is zero based to be more convenient
        :param nodeIndex: The index is zero based to be more convenient
        :return:
        """
        return self.layers[layerIndex][nodeIndex]

    def setNode(self, layerIndex, nodeIndex, gp):
        """

        :param layerIndex: The index is zero based to be more convenient
        :param nodeIndex: The index is zero based to be more convenient
        :param gp: GP that will be set to this index
        :return:
        """
        self.layers[layerIndex][nodeIndex] = gp

    def setLayer(self, layerIndex, layer):
        """

        :param layerIndex: The index is zero based to be more convenient
        :param layer: layer that will be set to this index
        :return:
        """
        self.layers[layerIndex] = layer

    def predict(self, testDataForFistLayer, jitter=0, covoption=1,include_likelihood=True,include_uncertainty=True, include_covnoise=True,gpindices=None):
        """
        :param testDataForFistLayer:
        :return: predictedMean: is a list with nlayers elements.
                Each element in this list is Nx(number of nodes in that layer) matrix. N  is the number of testing points
                predictedVar: has the same structure like predictedMean
        """
        Util.debug('Test Data for the first layer with shape: '+str(testDataForFistLayer.shape),debuglevel=2)
        predictedMean = []
        predictedVar = []
        # prediction for the first layer
        tm1, tv1 = self.predictLayer(0,testDataForFistLayer,None, include_likelihood=include_likelihood, gpindices=gpindices)
        predictedMean.append(tm1)
        if not include_uncertainty:
            tv1 = np.zeros(tv1.shape)
        predictedVar.append(tv1)
        # predict starting from the second layer
        for lindex in range(1,len(self.layers)):
            tmean = predictedMean[lindex-1]
            tvar = predictedVar[lindex-1]
            tm2,tv2 = self.predictLayer(lindex,tmean,tvar, jitter=jitter, covoption=covoption, include_likelihood=include_likelihood, include_covnoise=include_covnoise, gpindices=gpindices)
            predictedMean.append(tm2)
            predictedVar.append(tv2)
        return predictedMean, predictedVar


    def predictLayer(self, layerIndex, tm,tv, jitter=0, covoption=1,include_likelihood=True, include_covnoise=True,tsndepend=None,tsnahead=10,gpindices=None):
        # for layerIndex = 0, you can use tv = None or zeros.
        if tsndepend is not None:
            return self.predictLayerTS(layerIndex,tm,tv,jitter,covoption,include_likelihood,include_covnoise,tsndepend,tsnahead)
        if layerIndex == 0:
            return Util.predict(self.getLayer(0), None,tm, include_likelihood=include_likelihood,gpindices=gpindices)
        targetLayer = self.getLayer(layerIndex)
        nnodes = len(targetLayer)
        # N is the number of testing points
        N = tm.shape[0]
        pMean = np.zeros((N,nnodes))
        pVar = np.zeros((N,nnodes))

        if gpindices is None:
            gpindices = range(nnodes)
        for nodeIndx in gpindices:
            targetNode = targetLayer[nodeIndx]
            # for now we use only RBF kernel for all nodes.
            mean,var = aRBF.predict(targetNode,tm,tv,jitter, covoption=covoption, include_likelihood=include_likelihood, include_covnoise=include_covnoise)
            pMean[:,nodeIndx] = mean.reshape(-1)
            pVar[:,nodeIndx] = var.reshape(-1)
            # print 'Varinace (min/max): ',var.min(), var.max()
        return pMean, pVar
    def predictLayerTS(self, layerIndex, tm,tv, jitter=0, covoption=1,include_likelihood=True, include_covnoise=True,tsndepend=1,tsnahead=10):
        targetLayer = self.getLayer(layerIndex)
        nnodes = len(targetLayer)
        # N is the number of testing points
        N = tsnahead
        pMean = np.zeros((N,nnodes))
        pVar = np.zeros((N,nnodes))
        tmm = np.copy(tm)
        tvv = np.copy(tv)
        seriesindx = 0
        for nindx in range(N):
            Util.debug('*************Time Series: predicting point at index '+str(nindx)+'*******************',debuglevel=2)
            mean = np.zeros((1,nnodes))
            var = np.zeros((1,nnodes))
            for nodeIndx in range(nnodes):
                Util.debug('Time Series: predicting value for node at index '+str(nodeIndx),debuglevel=2)
                targetNode = targetLayer[nodeIndx]
                # for now we use only RBF kernel for all nodes.
                xinput = tmm[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                xvar = tvv[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                mean[0,nodeIndx:nodeIndx+1],var[0,nodeIndx:nodeIndx+1] = aRBF.predict(targetNode,xinput,xvar,jitter, covoption=covoption, include_likelihood=include_likelihood, include_covnoise=include_covnoise)
            seriesindx += 1
            # Util.debug('---->Update the testing set by adding the predicted time t'+str(nindx))
            tmm = np.vstack((tmm,mean))
            tvv = np.vstack((tvv,var))
            pMean[nindx:nindx+1,:] = mean.reshape(1,-1)
            pVar[nindx:nindx+1,:] = var.reshape(1,-1)
        Util.debug('****************************************************************',debuglevel=1)
        Util.debug('****************************************************************',debuglevel=1)
        Util.debug('Time Series Summary: ',debuglevel=1)
        Util.debug('----------Predicted: '+str(N)+' points',debuglevel=1)
        Util.debug('----------True input used: '+str(tm.shape[0])+' true point',debuglevel=1)
        estimatedpoints = N + tsndepend -1 - tm.shape[0]
        # if estimatedpoints < 0:
        #     estimatedpoints = 0
        Util.debug('----------Estimated input used: '+str(estimatedpoints)+' estimated point',debuglevel=1)
        return pMean, pVar