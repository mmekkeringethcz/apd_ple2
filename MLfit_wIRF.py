# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:03:57 2019
Maximum likelihood fitting of decay data using a measured IRF. Minimizer still  has room for improvement
Note: All times are in ps for now
@author: rober
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import numba as nb
from scipy.optimize import minimize # used for implementation of maximum likelihood exponential fit
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve
import timeit
from iminuit import Minuit
from pprint import pprint 
#%%
 


def MaxLikelihoodFit_c(tlist,ylist,istart,iend,plotbool=False):
    ### Maximum likelihood routine to fit single exponential. Pro: Works also for small amount of data (single bins of 10ms!)
    # tlist: x-axis values, here time in ns; ylist: y-axis values, here cts per tlist-bin; istart and iend: first and last element of tlist and ylist that are considered for the fit.

    # check if istart and iend are good numbers
    if istart<0 or istart>=len(ylist):
        istart = 0
        print('WARNING: adapted istart in MaxLikelihoodExpFit')
    if iend<=istart or iend>len(ylist):
        iend = len(ylist)
        print('WARNING: adapted iend in MaxLikelihoodExpFit')

    # shift t0 to t=0
    ydata = ylist[istart:iend]
    xdata = tlist[istart:iend]

    # do calculations
    initParams = [np.max(ydata), 2500, 400,1.7 ] #initial guess for A and tau
    
    results = minimize(MaxLikelihoodFunction_c, initParams, args=(xdata,ydata),method='Nelder-Mead',options={'disp': True}) # minimize the negative of the maxlikelihood function instead of maximimizing
    
    Aest = results.x[0] # get results of fit, A
    tauest = results.x[1] # get results of fit, tau
    t0est = results.x[2]
    bgest = results.x[3]

#    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        


    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
        yest = np.convolve((xdata>t0est)*Aest*np.exp(-(xdata-t0est)/tauest)+bgest,IRFtouse) 
        plt.figure()
        plt.semilogy(tlist,ylist,'.',xdata,yest[0:len(xdata)],[xdata[1],xdata[-1]],[bgest,bgest],'k--')
        plt.xlim([xdata[1],xdata[-1]])
        plt.show()        
        
    return(tauest,Aest,t0est,bgest,results)

def MaxLikelihoodFitBi_c(tlist,ylist,istart,iend,plotbool=False):
    ### Maximum likelihood routine to fit single exponential. Pro: Works also for small amount of data (single bins of 10ms!)
    # tlist: x-axis values, here time in ns; ylist: y-axis values, here cts per tlist-bin; istart and iend: first and last element of tlist and ylist that are considered for the fit.

    # check if istart and iend are good numbers
    if istart<0 or istart>=len(ylist):
        istart = 0
        print('WARNING: adapted istart in MaxLikelihoodExpFit')
    if iend<=istart or iend>len(ylist):
        iend = len(ylist)
        print('WARNING: adapted iend in MaxLikelihoodExpFit')

    # shift t0 to t=0
    ydata = ylist[istart:iend]
    xdata = tlist[istart:iend]

    # do calculations
    initParams = [np.max(ydata)/5, 2500, np.max(ydata), 100, 200, 1] #initial guess for A and tau
    results = minimize(MaxLikelihoodFunction_c, initParams, args=(xdata,ydata),method='Nelder-Mead',options={'disp': True, 'maxiter': 3000}) # minimize the negative of the maxlikelihood function instead of maximimizing
    A1est = results.x[0] # get results of fit, A
    tau1est = results.x[1] # get results of fit, tau
    A2est = results.x[2] # get results of fit, A2
    tau2est = results.x[3] # get results of fit, tau
    t0est = results.x[4]
    bgest = results.x[5]

#    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        


    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
        yest = np.convolve((xdata>t0est)*(A1est*np.exp(-(xdata-t0est)/tau1est)+A2est*np.exp(-(xdata-t0est)/tau2est)),IRFtouse)+bgest
        plt.figure()
        plt.semilogy(tlist,ylist,'.',xdata,yest[0:len(xdata)],[xdata[1],xdata[-1]],[bgest,bgest],'k--')
        plt.xlim([xdata[1],xdata[-1]])
        plt.show()        
        
    return(results)

def MaxLikelihoodFunction(params,xdata,ydata): 
    # max likelihood function for A*exp(-t/tau), needed in function MakLikelihoodFit
    # params = [A,tau]
    A1 = params[0]
    tau1 = params[1]
    t0 = params[2]
    bg = params[3]
#    A2 = params[2]
#    tau2 = params[2]
    model = A1*np.exp(-(xdata-t0)/tau1)+bg
    E = 0;
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        E = E + ydata[i]*np.log(model[i])-(model[i])
        
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!

def MaxLikelihoodFunctionconv(params,xdata,ydata): 
    # max likelihood function for A*exp(-t/tau), convolved with IRF
    # params = [A,tau]
    A1 = params[0]
    tau1 = params[1]
    t0 = params[2]
    bg = params[3]
#    A2 = params[2]
#    tau2 = params[2]
    E = 0;
    model1 = np.convolve((xdata>t0)*A1*np.exp(-(xdata-t0)/tau1),IRFtouse)+bg 
    model1 [model1 <= 0] = 1e-10
#    print(np.min(model1))
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        
        E = E + ydata[i]*np.log(model1[i])-(model1[i])
        
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!
    
def MaxLikelihoodFunctionconvBi(params,xdata,ydata): 
    # max likelihood function for A*exp(-t/tau), convolved with IRF
    # params = [A,tau]
    A1 = params[0]
    tau1 = params[1]
    A2 = params[2]
    tau2 = params[3]
    t0 = params[4]
    const = params[5]
#    A2 = params[2]
#    tau2 = params[2]
    E = 0;
    model1 = np.convolve((xdata>t0)*(A1*np.exp(-(xdata-t0)/tau1)+A2*np.exp(-(xdata-t0)/tau2)),IRFtouse)+const
    model1 [model1 <= 0] = 1e-10
#    print(np.min(model1))
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        
        E = E + ydata[i]*np.log(model1[i])-(model1[i])
        
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!
    
def Maxliketest(A1,tau1,A2,tau2,t0,const):
    E = 0;
    model1 = np.convolve((xdata>t0)*(A1*np.exp(-(xdata-t0)/tau1)+A2*np.exp(-(xdata-t0)/tau2)),IRFtouse) +const
    model1 [model1 <= 0] = 1e-10
#    print(np.min(model1))
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        
        E = E + ydata[i]*np.log(model1[i])-(model1[i])
        
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!
    
#%%
time=np.linspace(0,32000,32001)


#IRF = pd.read_csv("C:/Users/rober/Documents/Doktorat/Projects/Felipe NPL/20191217 NPL on Si-SiO2 AR4-85 100 dil ultrafast.dat", header = None, low_memory = False, names = ["Time","Intensity"], delimiter = ";")
IRF490_raw = pd.read_csv("C:/Users/rober/Documents/Doktorat/Projects/Felipe NPL/20191217 NPL on Si-SiO2 AR4-85 100 dil ultrafast/20191217_IRF_ultrafast_490nm_5050BS_on_mirror_reflection_100kHz_1e-2_counts_per_pulse.dat", sep='\t', header = None, skiprows = 11 , low_memory = False, names = ["Counts","Counts2","Counts3"])

#MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)

#%% 
IRF490=IRF490_raw
len(IRF490.Counts)
IRF490Time = np.linspace(250,1500,1251) #Time in ps
IRF490rawTime = np.linspace(0,len(IRF490.Counts)-1,len(IRF490.Counts))
plt.figure()
plt.semilogy(IRF490rawTime,IRF490.Counts/np.max(IRF490.Counts))
plt.xlim(0,2000)
interpIRF= interp1d(IRF490rawTime, (IRF490rawTime>200)*IRF490.Counts/np.max(IRF490.Counts),kind='cubic',fill_value='extrapolate')
plt.semilogy(IRF490Time,interpIRF(IRF490Time))
#plt.ylim([1e-6,1.1])


#%% Read in data
data = pd.read_csv("C:/Users/rober/Documents/Doktorat/Projects/Felipe NPL/20191217 NPL on Si-SiO2 AR4-85 100 dil ultrafast/20191217_thickfilm_spot1_430nm_ultrafast_455DCLP_450LP_only_491_nm_peak_100kHz_1e-2_counts_per_pulse_higherpower.dat", sep='\t', header = None, skiprows = 11 , low_memory = False, names = ["Counts","Counts2","Counts3"])
#data = pd.read_csv("C:/Users/rober/Documents/Doktorat/Projects/Felipe NPL/20191217 NPL on Si-SiO2 AR4-85 100 dil ultrafast/20191217_thickfilm_spot1_430nm_ultrafast_455DCLP_450LP_total_signal_100kHz_1e-2_counts_per_pulse.dat", sep='\t', header = None, skiprows = 11 , low_memory = False, names = ["Counts","Counts2","Counts3"])

#plt.semilogy(IRF490rawTime*1e-3,data.Counts/np.max(data.Counts))
IRFtime = IRF490Time
#Fit
IRFtouse = interpIRF(IRFtime)/np.sum(interpIRF(IRFtime))
IRFtouse [IRFtouse <= 1e-5] = 1e-5

counts=data.Counts.to_numpy()

#plt.figure()
plt.semilogy(IRF490rawTime,counts/np.max(counts))

time=np.linspace(0,10000,10501)
taumodel=20;  #Decay time in ps
#modeldecay=1e4*np.exp(-time/taumodel)
shift=60+250-2*7
modeldecay=(time>shift)*1e3*np.exp(-(time-shift)/taumodel)

#plt.semilogy(IRFtime+320,IRFtouse/np.max(IRFtouse))
modelconv=np.convolve(modeldecay,IRFtouse)
plt.semilogy(np.linspace(0,len(modelconv)-1,len(modelconv)),modelconv/np.max(modelconv))
plt.ylim(1e-8,1.1e1)
plt.xlim(0,1000)
#%%
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunctionconv)

result=MaxLikelihoodFit_c(IRF490rawTime,counts,istart = 0,iend = 5000,plotbool=True)

plt.ylim(1e0,1e4)
#%%
time=np.linspace(0,10000,10001)
IRFtime=np.linspace(0,5000,5001)
taumodel=300;  #Decay time in ps
#modeldecay=1e4*np.exp(-time/taumodel)
modeldecay=1e3*np.exp(-time/taumodel)
plt.figure()
#plt.semilogy(time*1e-3,modeldecay/np.max(modeldecay))
#plt.semilogy(time*1e-3+1.67,modeldecay/np.max(modeldecay))
#plt.semilogy(IRFtime*1e-3,interpIRF(IRFtime))
plt.ylim([1e-4,1.1])
IRFtouse = interpIRF(IRFtime)/np.sum(interpIRF(IRFtime))
IRFtouse [IRFtouse < 0] = 0
modelconv=np.convolve(modeldecay,IRFtouse)
modelconv = np.random.poisson(lam=modelconv)
#modelconv=np.sum
len(modelconv)
plt.semilogy(np.linspace(0,len(modelconv)-1,len(modelconv))*1e-3,modelconv/np.max(modelconv))
#plt.semilogy(np.linspace(0,len(modelconv2)-1,len(modelconv2))*1e-3,modelconv2/np.max(modelconv2)+0.1)
#plt.semilogy(time*1e-3,modelconv[0:len(time)])
plt.xlim([0,10])



#%% Biexponential fit on Pippos data
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunctionconvBi)

result=MaxLikelihoodFitBi_c(IRF490rawTime,counts,istart = 0,iend = 12000,plotbool=True)
result.x[0]
result.x[1]
result.x[2]
result.x[3]
result.x[4]
result.x[5]

plt.ylim(1e-1,1e4)

#%% Use iminuit minimizer
xdata = IRF490rawTime[0:12000]
ydata = counts[0:12000]
m = Minuit(Maxliketest, A1=np.max(ydata)/2,tau1=1000,A2=np.max(ydata)/2,tau2=1000,t0=300,const=np.mean(counts[10000:-1]),
           error_A1=np.max(ydata)/100,error_tau1=5,error_A2=np.max(ydata)/100,error_tau2=50,error_t0=10,error_const=0.05,errordef = 0.5,
           limit_A1=(0,None),limit_tau1=(0,None),limit_A2=(0,None),limit_tau2=(0,None),limit_t0=(0,None),limit_const=(0,None))
#%% Run minimization
m.migrad()

#%% Plot result
A1est=m.values['A1']
tau1est=m.values['tau1']
A2est=m.values['A2']
tau2est=m.values['tau2']
t0est=m.values['t0']
constest=m.values['const']


yest = np.convolve((xdata>t0est)*(A1est*np.exp(-(xdata-t0est)/tau1est)+A2est*np.exp(-(xdata-t0est)/tau2est)),IRFtouse)+constest
plt.figure()
plt.semilogy(xdata,ydata,'.',xdata,yest[0:len(xdata)],[xdata[1],xdata[-1]],[constest,constest],'k--')
plt.xlim([xdata[1],xdata[-1]])
plt.show() 





#%%


MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunctionconv)

[tau,A,t0]=MaxLikelihoodFit_c(time,modelconv[0:len(time)],istart = 0,iend = len(time),bgcpb=0,plotbool=True)


#%% Biexponential

taumodel = (300,2000);  #Decay time in ps
Amodel = (1e1,3e0)

modeldecay=Amodel[0]*np.exp(-time/taumodel[0])+Amodel[1]*np.exp(-time/taumodel[1])
plt.figure()

#plt.ylim([1e-4,1.1])

modelconv=np.convolve(modeldecay,IRFtouse)
modelconv = np.random.poisson(lam=modelconv)
#modelconv=np.sum
len(modelconv)
plt.semilogy(np.linspace(0,len(modelconv)-1,len(modelconv))*1e-3,modelconv)
#plt.semilogy(np.linspace(0,len(modelconv2)-1,len(modelconv2))*1e-3,modelconv2/np.max(modelconv2)+0.1)
#plt.semilogy(time*1e-3,modelconv[0:len(time)])
plt.xlim([0,30])
#plt.ylim([])
np.sum(modelconv)
#%%


MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunctionconvBi)

result=MaxLikelihoodFitBi_c(time,modelconv[0:len(time)],istart = 0,iend = len(time),bgcpb=0,plotbool=True)
plt.ylim(1,1e4)
print('A1: '+str(Amodel[0])+' estimated: '+str(result[0]))
print('Tau1: '+str(taumodel[0])+'ps estimated: '+str(result[1])+'ps')
print('A2: '+str(Amodel[1])+' estimated: '+str(result[2]))
print('Tau2: '+str(taumodel[1])+'ps estimated: '+str(result[3])+'ps')
print('t0: '+str(0)+' estimated: '+str(result[4]))

#%%

MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunctionconv)
[tau,A,t0]=MaxLikelihoodFit_c(time,modelconv[0:len(time)],istart = 0,iend = len(time),bgcpb=0,plotbool=True)