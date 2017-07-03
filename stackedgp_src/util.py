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
 * util.py
 * 
 *--------------------------------------------------------------------------
 *-------------------------------------------------------------------------- 
 '''
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy, GPy

class Util:
    # ------------------------------------------------------------------------------------------
    # Static Variables
    # ------------------------------------------------------------------------------------------
    # debuglevel helps to show which degree of debugging.
    # From 0 to 3, 0 means no debugging, larger values more debug informations
    debuglevel = 0
    # ------------------------------------------------------------------------------------------
    # unbounding transformation - forwards
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def unbound_trans(X, minval,maxval):
        tmp = (X - minval) / (maxval - minval)
        v = np.log(tmp / (1.0 - tmp))
        return v
    # ------------------------------------------------------------------------------------------
    # unbounding transformation - backwards
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def unbound_inv(Y,minval,maxval):
        return minval + (maxval - minval) / (1.0 + np.exp(-Y))

    @staticmethod
    def normalize(inputdata):
        scaler = StandardScaler()
        norminput = scaler.fit_transform(inputdata)
        return scaler, norminput


    @staticmethod
    def getInv(A, option):
        if option == 1:
            # print 'Inverse used GPy.util.linalg.pdinv...'
            covInv, _, _, _ = GPy.util.linalg.pdinv(A)
        elif option == 2:
            # print 'Inverse used np.linalg.pinv...'
            covInv = np.linalg.pinv(A)
        elif option == 3:
            # print 'Inverse used np.linalg.inv...'
            covInv = np.linalg.inv(A)
        elif option == 4:
            # print 'Inverse used scipy.linalg.pinv...'
            covInv = scipy.linalg.pinv(A)
        elif option == 5:
            # print 'Inverse used scipy.linalg.inv...'
            covInv = scipy.linalg.inv(A)
        else:
            covInv = None
        return covInv

    @staticmethod
    def predict(gpList,normalizerList, data,include_likelihood=True,gpindices=None):
        """
        :param gpList: arraylist of M GPs
        :param data: input data for prediction. The data is Nx(Mxk) np.array. N is number of points. M is number of GPs.
         k is the input dimension for each GP. it should differ from GP to other.
         :param gpindices: is used to index gps from gpList for prediction.
        :return: predicted_mean (NXM), predicted_variance(NxM)
        """
        N = data.shape[0]
        if gpindices is None:
            gpindices = range(0,len(gpList))
        M = len(gpindices)
        indx = 0
        p_mean = np.zeros((N,M))
        p_var = np.zeros((N,M))
        m = 0
        # print '----------------->:',gpindices
        for gp, normalizIndx in zip([gpList[u] for u in gpindices], range(M)):
            k = gp.X.shape[1]
            d = data[:,indx:k+indx]
            indx += k
            if normalizerList is not None:
                d = normalizerList[normalizIndx].transform(d)
            p_mean[:,m:m+1],p_var[:,m:m+1] = gp.predict(d,include_likelihood=include_likelihood)
            m += 1
        return p_mean, p_var

    @staticmethod
    def debug(value, debuglevel=3):
        if Util.debuglevel <= 0:
            return
        elif Util.debuglevel == 1:
            if debuglevel <= 1:
                print value
        elif Util.debuglevel == 2:
            if debuglevel <= 2:
                print value
        elif Util.debuglevel == 3:
            if debuglevel <= 3:
                print value

