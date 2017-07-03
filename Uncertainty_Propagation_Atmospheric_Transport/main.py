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
 * main.py
 * 
 *--------------------------------------------------------------------------
 *-------------------------------------------------------------------------- 
 '''
import numpy as np
import GPy
from sklearn.preprocessing import StandardScaler
import sys, GPy, pickle
sys.path.append('../stackedgp_src')
from stackedGPNetwork import StackedGPNetwork
from analyticalRBFStackedGP import AnalyticalRBF as aRBF
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
#==============================================
#==============================================
N_WIND_STATIONS = 16
N_TRAIN_PUFFS = 15
N_TEST_PUFFS = 1
DELTA_T = 90
TIME_PERIOD = 1800
PUFF_Y_LOC = 6
PUFF_X_LOC = 6

WIND_U_MEAN = 0.004
WIND_U_STD = 0.001
WIND_V_MEAN= 0.004
WIND_V_STD = 0.001

fileId = 'Interactive'
#==============================================
#==============================================
class PuffState:
    def __init__(self, x=0, y=0, down_wind=0,x_var=0,y_var=0,down_w_var=0):
        self.x = x
        self.x_var = x_var
        self.y = y
        self.y_var = y_var
        self.down_wind = down_wind
        self.down_wind_var = down_w_var
    @staticmethod
    def getMontecarloSample(m,v,nsample=1):
        
        if v <= 0:
            sam =  m
        else:
            try:
                sam = np.random.normal(loc=m, scale=v, size=nsample)
            except:
                print 'Sampling Error: m=',m,' v=', v
                return m
        return sam
        
    @staticmethod
    def get_next_state(state, wind, delta_t):
        x_1 = state.x + wind.u * delta_t
        y_1 = state.y + wind.v * delta_t
        down_wind_1 = state.down_wind + np.sqrt(np.square(wind.u) + np.square(wind.v)) * delta_t
        s_1 = PuffState(x_1, y_1, down_wind=down_wind_1)
        return s_1

    def get_next_states(self, delta_t, num_time_samples, WIND_U_MEAN, WIND_U_STD, WIND_V_MEAN, WIND_V_STD):
        winds =  np.ndarray((1, num_time_samples), dtype=np.object)
        states = np.ndarray((1, num_time_samples), dtype=np.object)
        states[0, 0] = self
        winds[0, 0] = None
        wu = np.random.normal(loc=WIND_U_MEAN, scale=WIND_U_STD, size=1)
        wv = np.random.normal(loc=WIND_V_MEAN, scale=WIND_V_STD, size=1)
        
        # wi = scaler.transform(np.array([wu,wv]).reshape(1,2))[0]
        for i in range(1, num_time_samples):
            wind = Wind(wu,wv)
            states[0, i] = PuffState.get_next_state(states[0, i - 1],wind , delta_t)
            winds [0, i] = wind
        return states, winds

    @staticmethod
    def simulate_puffs(init_states, TIME_PERIOD, DELTA_T, WIND_U_MEAN, WIND_U_STD, WIND_V_MEAN, WIND_V_STD):
        '''
        :param init_states: vector (column wise) of initial states for different puffs
        :return:
        '''
        num_init_states = init_states.shape[0]
        num_time_samples = int(np.ceil(TIME_PERIOD / DELTA_T)+1)
        # states is a 2D matrix (num_init_statesX num_time_samples)
        states = np.ndarray((num_init_states, num_time_samples), dtype=np.object)
        winds = np.ndarray((num_init_states, num_time_samples), dtype=np.object)
        stat_indx = 0
        for state in init_states:
            states[stat_indx, :], winds[stat_indx, :] = state[0].get_next_states(DELTA_T, num_time_samples, WIND_U_MEAN, WIND_U_STD, WIND_V_MEAN, WIND_V_STD)
            stat_indx += 1
        return states, winds

    @staticmethod
    def reshapeDate(states, winds,thead=1):
        '''
        thead = 1
        # data should be shaped as rows = states.shape[0]*(states.shape[1]-thead)
        # and number of columns for inputs = featuresize (for state feature at time t)
        # and number of columns for outputs = featuresize (for state feature at time t+1)
        # note the features for state may be (x, y, down_wind)
        # get the first puff and retrieve its features
        '''
        print 'states.shape = ',states.shape

        featuresize = 3
        windcomponentlen = 2
        data = np.zeros((states.shape[0]*(states.shape[1]-thead),2*featuresize+windcomponentlen))
        states_thead = np.ndarray((states.shape[0]*(states.shape[1]-thead),thead+1), dtype=np.object)

        d_indx = 0
        for i in range(states.shape[0]):
            for j in range(states.shape[1]-thead):
                # features at time t
                d = states[i,j].getpuffFeatures()
                # output features at time t+1
                d1 = states[i,j+1].getpuffFeatures()
                # append wind compenents
                d.append(winds[i,j+1].u)
                d.append(winds[i,j+1].v)
                for f in d1:
                    d.append(f)
                data[d_indx] = np.array(d).reshape((1,-1))
                states_thead[d_indx,0] = states[i,j]
                for t in range(thead):
                    states_thead[d_indx,t+1] = states[i,j+t+1]
                d_indx += 1

        return data, states_thead

    @staticmethod
    def getaRBFPrediction(targetNode,xinput,xvar, label):
        vq = 10
#         print 'xinput: ',xinput, '  xvar: ',xvar
        for i in range(1,5):
            m,v= aRBF.predict(targetNode,xinput,xvar,covoption=i,jitter=1e-1)
#             print '====>m = ',m, ' v= ',v
            if vq > v and v > 0:
                vq = v
        if vq == 10 or vq < 0:
            print 'label: ', label, ' m = ',m, ' v= ',v, ' vq = ',vq, 'xinput: ',xinput, 'xvar: ',xvar
#             sys.exit()
        return m,vq

    @staticmethod
    def predictTS(gplayer,winduGP,windvGP, state_thead_test,tsndepend = 1, order=['x','y','d'],montecarlo=False,nmcsamples=100):
        fileId = 'Interactive'
        featuresize = 3
        if montecarlo:
            state_thead_test = np.tile(state_thead_test[0].reshape(-1,1),nmcsamples).T

        thead= 20
        
        se = np.zeros((1,featuresize))
        total= 1
        pre_states = np.ndarray((state_thead_test.shape[0],thead+1),dtype=np.object)
        print'state_thead_test = ', state_thead_test.shape, 'pre_states = ', pre_states.shape
        for t in range(state_thead_test.shape[0]):
            print '\n*********************Testing point at index ',t,
            seriesindx = 0
            nnodes = len(gplayer)
            tmm = np.array(state_thead_test[t,0].getpuffFeatures()).reshape((1,-1))
            pre_states[t,0] = state_thead_test[t,0]
            tvv = np.zeros(tmm.shape)
            for nindx in range(thead):
                print '\n',t,'- Propagate prediction at thead '+str(nindx)+'======>nodes: ',
                mean = np.zeros((1,nnodes))
                var = np.zeros((1,nnodes))
                for nodeIndx in range(nnodes):
                    print str(nodeIndx),
                    # Using MC sampling
                    if montecarlo:
                        targetNode = gplayer[nodeIndx]
                        xinput = tmm[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                        xvar = tvv[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                        xsamp = np.zeros(xinput.shape)
                        # sample the puff features based on its uncertainty
                        for s in range(xinput.shape[1]):
                            xsamp[0,s] = PuffState.getMontecarloSample(xinput[0,s], np.sqrt(xvar[0,s]))
                        # get the prediction for the wind based on the location x,y
                        um, uv = winduGP.predict(xsamp[0,:2].reshape(1,-1))
                        vm, vv = windvGP.predict(xsamp[0,:2].reshape(1,-1))
                        # sample the wind u,v based on their uncertainty
                        u = PuffState.getMontecarloSample(um[0], np.sqrt(uv[0]))
                        v = PuffState.getMontecarloSample(vm[0],np.sqrt(vv[0]))

                        # create input for the prediction
#                         xsamp = np.concatenate((xsamp,np.array([u,v]).reshape(1,-1)),axis=1)
                        xsamp = xsamp[0]
                        if order[nodeIndx] == 'x':
                            xsamp = np.array([xsamp[nodeIndx],u]).flatten().reshape(1,-1)
#                             xvar = np.array([xvar[nodeIndx],uv]).flatten().reshape(1,-1)
                        elif order[nodeIndx] == 'y':
                            xsamp = np.array([xsamp[nodeIndx],v]).reshape(1,-1)
#                             xvar = np.array([xvar[nodeIndx],vv]).flatten().reshape(1,-1)
                        elif order[nodeIndx] == 'd':
                            xsamp = np.array([xsamp[nodeIndx],u, v]).reshape(1,-1)
#                             xvar = np.array([xvar[nodeIndx],uv,vv]).flatten().reshape(1,-1)
                        
                        prem, prev = targetNode.predict(xsamp)
#                         print '===> prem, prev =', prem, prev 
                        mean[0,nodeIndx:nodeIndx+1],var[0,nodeIndx:nodeIndx+1] = prem, prev 
                    else:
                        targetNode = gplayer[nodeIndx]
                        # using Analytical method
                        xinput = tmm[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                        xvar = tvv[seriesindx:seriesindx+tsndepend,:].flatten().reshape((1,-1))
                        # get the prediction for the wind based on the location x,y
                        um, uv = aRBF.predict(winduGP,xinput[0,:2].reshape(1,-1),xvar[0,:2].reshape(1,-1))
                        vm, vv = aRBF.predict(windvGP,xinput[0,:2].reshape(1,-1),xvar[0,:2].reshape(1,-1))
                        wm = np.array([um[0][0],vm[0][0]])
                        wv = np.array([uv[0][0],vv[0][0]])
                        xinput = xinput[0]
                        xvar = xvar[0]
                        if order[nodeIndx] == 'x':
#                             print 'Order x: xinput', xinput
                            xinput = np.array([xinput[nodeIndx],wm[0]]).flatten().reshape(1,-1)
#                             print 'Order x: modified xinput', xinput
                            xvar = np.array([xvar[nodeIndx],wv[0]]).flatten().reshape(1,-1)
                        elif order[nodeIndx] == 'y':
#                             print 'Order y: xinput', xinput[nodeIndx]
                            xinput = np.array([xinput[nodeIndx],wm[1]]).reshape(1,-1)
#                             print 'Order y: modified xinput', xinput
                            xvar = np.array([xvar[nodeIndx],wv[1]]).flatten().reshape(1,-1)
                        elif order[nodeIndx] == 'd':
#                             print 'Order d: xinput', xinput[nodeIndx]
                            xinput = np.array([xinput[nodeIndx],wm[0], wm[1]]).reshape(1,-1)
#                             print 'Order d: modified xinput', xinput
                            xvar = np.array([xvar[nodeIndx],wv[0],wv[1]]).flatten().reshape(1,-1)

                        mean[0,nodeIndx:nodeIndx+1],var[0,nodeIndx:nodeIndx+1] = PuffState.getaRBFPrediction(targetNode,xinput,xvar,order[nodeIndx])
                seriesindx += 1
                tmm = np.vstack((tmm,mean))
                tvv = np.vstack((tvv,var))
                pre_states[t,nindx+1] = PuffState()
                pre_states[t,nindx+1].setpuffFeatures(mean.reshape(-1),np.sqrt(var.reshape(-1)))
                se += PuffState.SE(pre_states[t,nindx+1],state_thead_test[t,nindx+1])
                print '\n'
                # PuffState.printFeaturesName()
#                 print 'MSE = ',','.join(map(str,se.flatten()/total))
#                 print 'RMSE = ',','.join(map(str,np.sqrt(se.flatten()/total)))
                total += 1
        with open('predicted'+str(fileId)+'_'+str(montecarlo)+'.txt','w') as handle:
            for puf in pre_states:
                st = ''
                for p in puf:
                    st += ','+str(p.x)+','+str(p.x_var)+','+str(p.y)+','+str(p.y_var)+','+str(p.down_wind)+','+str(p.down_wind_var)
                handle.write(st[1:]+'\n')
        handle.close()

        return pre_states,se,state_thead_test

    def getpuffFeatures(self):
        return [self.x,self.y,self.down_wind]
    def setpuffFeatures(self,flist,varlist):
        self.x = flist[0]
        self.x_var = varlist[0]
        self.y = flist[1]
        self.y_var = varlist[1]
        self.down_wind = flist[2]
        self.down_wind_var = varlist[2]
    @staticmethod
    def SE(p1,p2):
        sex = (p1.x - p2.x)**2
        sey = (p1.y - p2.y)**2
        sedw = (p1.down_wind - p2.down_wind)**2
        se = np.array([sex[0],sey[0],sedw[0]])
#         print 'SE = ', se
        return se
    @staticmethod
    def printFeaturesName():
        print 'x, y, down_wind'

#==============================================
#==============================================
# ## ============== WIND ======================================
class Wind:
    def __init__(self, u, v):
        self.u = np.copy(u)
        self.v = np.copy(v)
    
    # generate Wind data   
    @staticmethod
    def genWind_uniformX_uniformY(savefile='wind_uniformX_gaussianY.txt', regenerate=False):
        # initiate locations of the wind sensors
        if regenerate:
            x_samples = np.ones((N_WIND_STATIONS,))
            y_samples = np.ones((N_WIND_STATIONS,))
            indx = 0
            # build wind sensor grid
            for i in [0,4,8,12]:
                for j in [0,4,8,12]:
                    x_samples[indx] = i
                    y_samples[indx] = j
                    indx += 1

            # random sample reading for the sensors
            u_samples = np.random.normal(loc=WIND_U_MEAN, scale=WIND_U_STD, size=N_WIND_STATIONS)
            v_samples = np.random.normal(loc=WIND_V_MEAN, scale=WIND_V_STD, size=N_WIND_STATIONS)
            # save wind data to savefile
            winddata = np.zeros((N_WIND_STATIONS,4))
            with open(savefile, 'w') as outfile:
                for i in range(N_WIND_STATIONS):
                    winddata[i] = np.array([x_samples[i],y_samples[i],u_samples[i],v_samples[i]])
                    outfile.write(str(x_samples[i])+','+str(y_samples[i])+','+str(u_samples[i])+','+str(v_samples[i])+'\n')
            outfile.close()
        else:
            winddata = []
            with open(savefile, 'r') as infile:
                for line in infile:
                    line = line.strip().split(',')
                    winddata.append(map(float,line))
            infile.close()
            winddata = np.array(winddata)
        return winddata
#==============================================
#==============================================

winddata = Wind.genWind_uniformX_uniformY()
x_loc = winddata[:,:2]
ukernel = GPy.kern.RBF(input_dim=2,ARD=True)
winduGP = GPy.models.GPRegression(x_loc,winddata[:,2:3],ukernel, normalizer=False)
windvGP = GPy.models.GPRegression(x_loc,winddata[:,3:4], normalizer=False)
winduGP.optimize()
windvGP.optimize()

# windvGP.plot()

#==============================================
#==============================================
#==============================================

# several levels of loading data.
# first you can load the stored results that exist in the paper by setting create_new_results to False
# or you can load the generated data and run the StackedGP and MonteCarlo by setting load_stored_data to True
# by setting load_stored_data to False you will generate a new random data.
# Also you need to specify if you need to use the stored GPs object or create a new ones.

create_new_results = False
load_stored_data = True
load_stored_GPs = True

if create_new_results:
    if load_stored_data:
        loadnamesList = ['train_states_thead', 'train_data','test_state']
        loadobjList = []
        for i in loadnamesList:
            print 'Loading '+i+' Ojbect...'
            with open(i+'_'+str(fileId)+'.pickle', 'r') as handle:
                loadobjList.append(pickle.load(handle))
            handle.close()
        train_states_thead, train_data, test_state = loadobjList
    else:
        # generate Train puffs
        x_samples = np.ones((N_TRAIN_PUFFS,))*PUFF_X_LOC
        y_samples = np.ones((N_TRAIN_PUFFS,))*PUFF_Y_LOC
        samples = np.hstack((x_samples.reshape(-1,1),y_samples.reshape(-1,1)))

        # samples = scaler.transform(samples)

        init_state = np.ndarray((N_TRAIN_PUFFS, 1), dtype=np.object)
        indx = 0
        for x in samples:
            init_state[indx, 0] = PuffState(x=x[0], y=x[1])
            indx += 1
        train_states, train_winds = PuffState.simulate_puffs(init_state, TIME_PERIOD, DELTA_T, WIND_U_MEAN, WIND_U_STD, WIND_V_MEAN, WIND_V_STD)
        train_data, train_states_thead = PuffState.reshapeDate(train_states, train_winds)
        print 'train_states.shape = ',train_states.shape, ' train_data = ',train_data.shape

        # generate Test puffs
        x_samples = np.ones((N_TEST_PUFFS,))*PUFF_X_LOC
        y_samples = np.ones((N_TEST_PUFFS,))*PUFF_Y_LOC
        samples = np.hstack((x_samples.reshape(-1,1),y_samples.reshape(-1,1)))
        # print samples[0]
        # samples = scaler.transform(samples)
        # print samples[0]

        init_state = np.ndarray((N_TEST_PUFFS, 1), dtype=np.object)
        indx = 0
        for x in samples:
            init_state[indx, 0] = PuffState(x=x[0], y=x[1])
            indx += 1
        test_states, test_winds = PuffState.simulate_puffs(init_state, TIME_PERIOD, DELTA_T, WIND_U_MEAN, WIND_U_STD, WIND_V_MEAN, WIND_V_STD)
        test_data, test_states_thead = PuffState.reshapeDate(test_states,test_winds)

        test_state = np.ndarray((1,21),dtype=np.object)
        # print test_states_thead[0,0]
        test_state[0,0] = test_states_thead[0,0]
        for i in range(0,20):
            test_state[0,i+1] = test_states_thead[i,1]
        
        savenamesList = ['train_states_thead', 'train_data','test_state']
        saveobjList = [train_states_thead, train_data, test_state]
        for i,j in zip(savenamesList,saveobjList):
            print 'Saving '+i+' Ojbect...'
            with open(i+'_'+str(fileId)+'.pickle', 'w') as handle:
                pickle.dump(j,handle)
            handle.close()


    # print 'test_states.shape = ',test_states.shape, ' test_data = ',test_data.shape, ' test_states_thead = ',test_states_thead.shape

    if load_stored_GPs:
        loadnamesList = ['montGPs', 'stackedGP']
        loadobjList = []
        for i in loadnamesList:
            print 'Loading '+i+' Ojbect...'
            with open(i+'_'+str(fileId)+'.pickle', 'r') as handle:
                loadobjList.append(pickle.load(handle))
            handle.close()
        montGPs, stackedGP = loadobjList

    else:
        print 'Build GP objects and optimize...'
        montGPs = []
        stackedGP = []
        xkernel = GPy.kern.RBF(input_dim=2,ARD=True) #+ GPy.kern.White(5, variance=1e-3)
        xgp = GPy.models.GPRegression(train_data[:,[0,3]],train_data[:,5:6].reshape(-1,1), xkernel,normalizer=False)
        xgp.optimize()
        stackedGP.append(xgp)

        ykernel = GPy.kern.RBF(input_dim=2,ARD=True) #+ GPy.kern.White(5, variance=1e-3)
        ygp = GPy.models.GPRegression(train_data[:,[1,4]],train_data[:,6:7].reshape(-1,1), ykernel,normalizer=True)
        ygp.optimize()
        stackedGP.append(ygp)

        dkernel = GPy.kern.RBF(input_dim=3,ARD=True) #+ GPy.kern.White(5, variance=1e-3)
        dgp = GPy.models.GPRegression(train_data[:,[2,3,4]],train_data[:,7:8].reshape(-1,1), dkernel,normalizer=False)
        dgp.optimize()
        stackedGP.append(dgp)
    #=========================================================================================
        xkernel = GPy.kern.RBF(input_dim=2,ARD=True) + GPy.kern.White(2, variance=1e-6)
        xgp = GPy.models.GPRegression(train_data[:,[0,3]],train_data[:,5:6].reshape(-1,1), xkernel,normalizer=True)
        # xgp.kern.white.fix()
        xgp.optimize()
        montGPs.append(xgp)

        ykernel = GPy.kern.RBF(input_dim=2,ARD=True) + GPy.kern.White(2, variance=1e-6)
        ygp = GPy.models.GPRegression(train_data[:,[1,4]],train_data[:,6:7].reshape(-1,1), ykernel,normalizer=True)
        ygp.kern.white.fix()
        ygp.optimize()
        montGPs.append(ygp)

        dkernel = GPy.kern.RBF(input_dim=3,ARD=True) + GPy.kern.White(3, variance=1e-4)
        dgp = GPy.models.GPRegression(train_data[:,[2,3,4]],train_data[:,7:8].reshape(-1,1), dkernel,normalizer=True)
        dgp.kern.white.fix()
        dgp.optimize()
        montGPs.append(dgp)

        savenamesList = ['montGPs','stackedGP']
        saveobjList = [montGPs,stackedGP]
        for i,j in zip(savenamesList,saveobjList):
            print 'Saving '+i+' Ojbect...'
            with open(i+'_'+str(fileId)+'.pickle', 'w') as handle:
                pickle.dump(j,handle)
            handle.close()

    #=========================================================================================
    #=========================================================================================
    # run StackedGP 
    prestates,se,state_thead_test2 = PuffState.predictTS(stackedGP,winduGP,windvGP, test_state,tsndepend = 1, montecarlo=False,nmcsamples=100)
    # run montecarlo
    prestates,se,state_thead_test2 = PuffState.predictTS(montGPs,winduGP,windvGP, test_state,tsndepend = 1, montecarlo=True,nmcsamples=1000)

    # ============================================================
    print '======================================================='
    print '====================summary results========================='
    featuresize= 3
    print 'Number of testing points: ',prestates.shape[0]
    print 'Propagate each point with thead: ', prestates.shape[1]-1

    se = se.flatten()
    PuffState.printFeaturesName()
    print '========total MSE = ',se/prestates.flatten().shape[0]
    print '========total RMSE = ',np.sqrt(se/prestates.flatten().shape[0])

    se2 = np.zeros((1,featuresize))
    for i in range(prestates.shape[0]):
        se2 += PuffState.SE(prestates[i,-1],state_thead_test2[i,-1])

    se2 = se2.flatten()
    print '========MSE at thead = ',se2/prestates.shape[0]
    print '========RMSE at thead = ',np.sqrt(se2/prestates.shape[0])


    # In[16]:


num_bins = 50
stateIndx = 20
print 'Plotting...'
stackedResults = np.genfromtxt('predicted'+str(fileId)+'_False.txt', delimiter=',').reshape(1,-1)
print stackedResults.shape
print stackedResults

mcResutls = np.genfromtxt('predicted'+str(fileId)+'_True.txt', delimiter=',')

fileId = ''
mcMean = mcResutls[:,range(0,mcResutls.shape[1],2)]
stackedMean = stackedResults[:,range(0,stackedResults.shape[1],2)].reshape(1,-1)
stackedstd = stackedResults[:,range(1,stackedResults.shape[1],2)].reshape(1,-1)
font = {'size'   : 22}
matplotlib.rc('font', **font)
print stackedMean
with open('results'+str(fileId)+'/results','w') as handle:
    handle.write('stateIndx,feature,MC_mean, MC_std, STGP_mean, STGP_std')
    for stateIndx in [5,10,15,20]:
        stline = ''
        mcline = ''
        for featureIndex, label in zip([0,1,2],['X','Y','Down_wind']):
            # handle.write('\n'+str(stateIndx)+ ','+label+',')
            indx = stateIndx*3 + featureIndex
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(mcMean[:,indx], num_bins, normed=1)
            stdscale = 3
            x = np.linspace(stackedMean[0,indx]-stdscale*stackedstd[0,indx], stackedMean[0,indx]+stdscale*stackedstd[0,indx], 1000)
            # plt.bar(bin_boundary[:-1],bin_height,width = width)
            v = mlab.normpdf(x, stackedMean[0,indx], stackedstd[0,indx])
            # stdscale = 10
            # x = np.linspace(stackedMean[0,indx]-stdscale*stackedstd[0,indx], stackedMean[0,indx]+stdscale*stackedstd[0,indx], 1000)
            lines = plt.plot(x,v,'r--')
            plt.xlim([np.min(x)-1,np.max(x)+1])
            plt.setp(lines, linewidth=6, color='r')
            fig.tight_layout()
            if label == 'X':
                ax.set_ylabel('Probability density', **font)
            # handle.write('&'+str(stackedMean[0,indx])+ ','+str(stackedstd[0,indx]))
            stline += '& $'+str(np.round(stackedMean[0,indx],2))+ '$ & $'+str(np.round(stackedstd[0,indx],2))+'$'
            mcline += '& $'+str(np.round(np.mean(mcMean[:,indx]),2))+'$ & $'+ str(np.round(np.std(mcMean[:,indx]),2))+'$'
            plt.xlabel(label+' at puff time sample '+str(stateIndx), **font)
            plt.savefig('results'+str(fileId)+'/'+str(stateIndx)+'_'+label+'_'+str(fileId)+'.eps', bbox_inches='tight')
            plt.close()
        handle.write('\n $'+str(stateIndx)+'$'+stline+mcline+'\\\\ \hline')
#         handle.write('\n $'+str(stateIndx)+'\\\\ \hline')
handle.close()
with open('results'+str(fileId)+'/results','a') as handle:
    for stateIndx in [5,10,15,20]:
        for l in ['X','Y','Down_wind']:
            handle.write('\n\\includegraphics[scale=0.35]{../results'+str(fileId)+'/'+str(stateIndx)+'_'+l+'_'+str(fileId)+'}')
            # plt.show()


# In[ ]:



