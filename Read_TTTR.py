# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:44:49 2019

@author: rober
"""

import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
import multiprocessing
from operator import sub
from joblib import Parallel, delayed
import scipy, lmfit
from scipy.optimize import minimize # used for implementation of maximum likelihood exponential fit
from matplotlib import gridspec
import matplotlib.colors as mcolors
from math import factorial
from math import *
from scipy.stats import poisson
get_ipython().run_line_magic('matplotlib', 'auto')
import matplotlib as mpl
import pickle
import numba as nb
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%matplotlib auto

# In[10]:


#I get errors unless the function that the cores execute in parallel is defined outside the class function 
def hist2(x,y,bins):
    store = np.zeros(len(bins)-1,dtype='float');
    for i in x:
        res = y.searchsorted(bins+i)
        store += res[1:]-res[:-1]
    return store
def load_obj(name, folder ):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# In[23]:


def ImportT3(filename):
    with open(filename, "rb+") as f:
    
        
        ##This section is needed to .ptu files and should be commented for .out
        while True:
            if f.read(1) == b'M':
                if f.read(18) == b'easDesc_Resolution': # recognize time unit entry
                    break
        f.read(21) # rest of the tag
        dtmicro = struct.unpack('d',f.read(8))[0]
        #print('Microtime unit:', dtmicro)
    
        while True:
            if f.read(1) == b'M':
                if f.read(24) == b'easDesc_GlobalResolution': # recognize time unit entry
                    break
        f.read(15) # rest of the tag
        dtmacro = struct.unpack('d',f.read(8))[0]
        #print('Macrotime unit:', dtmacro)
    
        while True:
            if f.read(1) == b'T':
                if f.read(23) == b'TResult_NumberOfRecords': # recognize number of records entry
                    break
        f.read(16) # rest of the tag
        nrrec = struct.unpack('q',f.read(8))[0] # extract number of records
        #print('Number of records in file:', nrrec)

        while True:
            if f.read(1) == b'H':
                if f.read(9) == b'eader_End':
                    #print('Header_End found')
                    break
        f.read(38) # rest of Header_End
        
#        
#        nrrec=HHsettings["overallCounts"]
#        dtmicro=HHsettings["resolution"]*1e-12    #in s
#        dtmacro=1/HHsettings["syncRate"]    #in s
#        dtmacro=1/1946760.0
        macrotimes0 = np.zeros(nrrec,dtype='int64');
        microtimes0 = np.zeros(nrrec,dtype='int64');
        macrotimes1 = np.zeros(nrrec,dtype='int64');
        microtimes1 = np.zeros(nrrec,dtype='int64');
        macrotimesfireA = np.zeros(nrrec,dtype='int64');
        microtimesfireA = np.zeros(nrrec,dtype='int64');
        macrotimesfireB = np.zeros(nrrec,dtype='int64');
        microtimesfireB = np.zeros(nrrec,dtype='int64');
        macrotimesfireC = np.zeros(nrrec,dtype='int64');
        microtimesfireC = np.zeros(nrrec,dtype='int64');
        macrotimesfireD = np.zeros(nrrec,dtype='int64');
        microtimesfireD = np.zeros(nrrec,dtype='int64');
        macrotimecycle0 = np.zeros(nrrec,dtype='int64');
        macrotimecycle1 = np.zeros(nrrec,dtype='int64');
        overflows = 0
        nrphotons0 = 0
        nrphotons1 = 0
        nrfireA = 0
        nrfireB = 0
        nrfireC = 0
        nrfireD = 0
        prevchann = 0
        lastcyclestarttime = 0
        
        for i in range(nrrec):
            entry = f.read(4)
            channel = struct.unpack("I",entry)[0] >> 25 # read channel number, first 7 bits
            if channel == 0:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimes0[nrphotons0] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimes0[nrphotons0] = microtime
                macrotimecycle0[nrphotons0] = macrotime-lastcyclestarttime + 1024*overflows
                nrphotons0 += 1
                prevchann = 0
            elif channel == 1:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimes1[nrphotons1] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimes1[nrphotons1] = microtime
                macrotimecycle1[nrphotons1] = macrotime-lastcyclestarttime + 1024*overflows
                nrphotons1 += 1
                prevchann = 1
            elif channel == 127:
                nroverflows = (struct.unpack("I",entry)[0] & 0x3FF)
                overflows += nroverflows
                prevchann = 127
            elif channel == 65:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireA[nrfireA] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireA[nrfireA] = microtime
                nrfireA += 1
            elif channel == 66:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireB[nrfireB] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireB[nrfireB] = microtime
                nrfireB += 1  
            elif channel == 67:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireC[nrfireC] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireC[nrfireC] = microtime
                nrfireC += 1 
            elif channel == 68:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireD[nrfireD] = macrotime + 1024*overflows
                lastcyclestarttime = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireD[nrfireD] = microtime
                nrfireD += 1                              
            else:
                print('bad channel:',channel)
                
    microtimes0 = microtimes0[:nrphotons0]
    macrotimes0 = macrotimes0[:nrphotons0]
    macrotimecycle0 = macrotimecycle0[:nrphotons0]
    microtimes1 = microtimes1[:nrphotons1]
    macrotimes1 = macrotimes1[:nrphotons1]
    microtimesfireA = microtimesfireA[:nrfireA]
    macrotimesfireA = macrotimesfireA[:nrfireA]
    microtimesfireB = microtimesfireB[:nrfireB]
    macrotimesfireB = macrotimesfireB[:nrfireB]
    microtimesfireC = microtimesfireC[:nrfireC]
    macrotimesfireC = macrotimesfireC[:nrfireC]
    microtimesfireD = microtimesfireD[:nrfireD]
    macrotimesfireD = macrotimesfireD[:nrfireD]
    
    print('nrphotons0:',nrphotons0)
    print('nrphotons1:',nrphotons1)
    print('nrfireA:',nrfireA)
    print('nrfireB:',nrfireB)
    print('overflows:',overflows)
    
    return [dtmicro, dtmacro, microtimes0, macrotimes0, microtimes1, macrotimes1, nrphotons0,nrphotons1,overflows,microtimesfireA,macrotimesfireA,nrfireA,microtimesfireB,macrotimesfireB,nrfireB,macrotimesfireC,nrfireC,macrotimesfireD,nrfireD,macrotimecycle0,macrotimecycle1]

def ShiftPulsedData(microtimes0,microtimes1,macrotimes0,macrotimes1,dtmicro,dtmacro):
    dtmax = 8
    
    [ylist1,xlist1] = np.histogram(microtimes1,int(dtmacro/dtmicro),[0,int(dtmacro/dtmicro)])
    [ylist0,xlist0] = np.histogram(microtimes0,int(dtmacro/dtmicro),[0,int(dtmacro/dtmicro)])
    tlist = (xlist0[:-1]+0.5*(xlist0[1]-xlist0[0]))*dtmicro*1e9

    corrx = []; corry = [] # find shift for which the two decay curves overlap most
    for i in range(-dtmax,dtmax):
        corrx.append(i)
        corry.append(sum(ylist1[dtmax:-dtmax]*ylist0[dtmax+i:-dtmax+i]))
    xmax = corry.index(max(corry))
    shift = corrx[xmax]
    
    tlist0 = (microtimes0-shift) + macrotimes0*int(dtmacro/dtmicro)
    tlist1 = microtimes1 + macrotimes1*int(dtmacro/dtmicro) #in units of dtmicro
    
#    plt.xlabel('time (ns)')
#    plt.ylabel('counts (a.u.)')
#    p1, = plt.plot(tlist,ylist0+ylist1)
#    p2, = plt.plot(tlist,ylist0)
#    p3, = plt.plot(tlist,ylist1)
#    plt.([p1,p2,p3], ["APD0 + APD1","APD0","APD1"])
           
    return(microtimes0-shift,microtimes1,tlist0,tlist1,dtmicro,dtmacro,tlist,ylist0+ylist1)

def GetLifetime(microtimes,dtmicro,dtmacro,dtfit,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=False,method='ML'): 
    # microtimes = microtimes array with photon events
    # dtfit is the time interval considered for the fit [s], tstart [s] is the starting point of the fit within the histogram. If set to -1 it starts at the time with the highest intensity.
    # histbinmultiplier is a multiplier. actual binwidth is given as histbinmultiplier*dtmicro[s]
    # ybg is the background considered for the fit (CHECK UNITS!!). If set to -1 --> try to estimate background based on last bins. set to 0 --> no background subtraction
    # plotbool: plot histogram with fit
#    print('Chosen method is:' + method)
    [ylist,xlist] = np.histogram(microtimes,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9
#    print(histbinmultiplier)
    istart = int(tstart/dtmicro) #find index of maximum element in ylist
    if istart < 0:
        istart = ylist.argmax()
    iend = istart + int(dtfit/(dtmicro*histbinmultiplier))
    if iend>len(tlist):
        iend = len(tlist) 
        
    # get background (by simply looking at last ten data points) and substract from intensity data.
    if ybg < 0:
        ybg = np.mean(ylist[-100:]) # mean background per histogram bin bin of length
            
    if method == 'ML': #maximum likelihood exponential fit
        [tau1fit,A1fit] = MaxLikelihoodFit(tlist,ylist,istart,iend,ybg,False)
    elif method == 'ML_c': #maximum likelihood exponential fit
        [tau1fit,A1fit] = MaxLikelihoodFit_c(tlist,ylist,istart,iend,ybg,False)
    elif method == 'WLS': # weighted least squares fit
        [taufit,Afit] = WeightedLeastSquareFit(tlist,ylist,istart,iend,ybg,plotbool=False)
    else:
        taufit = 0; Afit = 0;
        print('Error: invalid fit method')
    
    if plotbool == True:
        plt.xlabel('time (ns)')
        plt.ylabel('')
        plt.semilogy(tlist,ylist,'.',tlist[istart:iend],A1fit*np.exp(-(tlist[istart:iend]-tlist[istart])/tau1fit)+ybg)
        plt.semilogy([tlist[0],tlist[-1]],[ybg,ybg],'k--')
        plt.show()
        print('Fitted lifetime:',tau1fit,'ns; Amax:',A1fit)

    # Amax is the maximum y-value
    Amax = np.max(ylist)
        
    return(tau1fit,A1fit,ybg,istart)  

def HistPhotons(photontlist,binwidth,Texp): # finds the index of the first photon in each bin. photontlist is in [s]
    histmax = Texp # experiment duration [s]

    nrbins = int(Texp/binwidth)
    limits = np.full(nrbins,len(photontlist))
    counter,i = 0,0
    while counter < nrbins and i < len(photontlist):
        while photontlist[i] > counter*binwidth:
            limits[counter] = i
            counter += 1
        i += 1
    
    return(limits)

def MakeIntTrace(limits0,limits1,binwidth,Texp):
    nrbins = int(Texp/binwidth)
    inttrace = np.array([(limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr]) for binnr in range(nrbins-1)])

    fig = plt.figure(figsize=(15,3))
    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
    ax0 = plt.subplot(gs[0])
    p0 = ax0.plot(np.arange(len(inttrace))*binwidth,inttrace,'-',linewidth=0.5)
    plt.xlabel('time (s)')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.xlim([0,Texp])
    plt.ylim([0,1.1*np.max(inttrace)])

    histogram = np.histogram(inttrace,max(inttrace),[0,max(inttrace)])
    
    ax1 = plt.subplot(gs[1])
    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]))
    plt.xlabel('occurrence')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.ylim([0,1.1*np.max(inttrace)])
    
    return(inttrace)
    
    
#def MakeExTrace(limits0,binwidth,Texp):
#    nrbins = int(Texp/binwidth)
#    inttrace = np.array([(limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr]) for binnr in range(nrbins-1)])
#
#    fig = plt.figure(figsize=(15,3))
#    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
#    ax0 = plt.subplot(gs[0])
#    p0 = ax0.plot(np.arange(len(inttrace))*binwidth,inttrace,'-',linewidth=0.5)
#    plt.xlabel('time (s)')
#    plt.ylabel('counts / %i ms' %(binwidth*1e3))
#    plt.xlim([0,Texp])
#    plt.ylim([0,1.1*np.max(inttrace)])
#
#    histogram = np.histogram(inttrace,max(inttrace),[0,max(inttrace)])
#    
#    ax1 = plt.subplot(gs[1])
#    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]))
#    plt.xlabel('occurrence')
#    plt.ylabel('counts / %i ms' %(binwidth*1e3))
#    plt.ylim([0,1.1*np.max(inttrace)])
#    
#    return(inttrace)

def MakeTauTrace(taubinlist,intbinlist,binwidth,Texp,taumin=0,taumax=100,intmin=0,intmax=100,col='k'):
    nrbins = int(Texp/binwidth)
  
    fig = plt.figure(figsize=(15,7))
    gs = gridspec.GridSpec(2,2,width_ratios=[4,1])
    ax0 = plt.subplot(gs[0])
    p0 = ax0.plot(np.arange(len(intbinlist))*binwidth,intbinlist,'-',linewidth=0.5,color=col)
    plt.xlabel('time (s)')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.xlim([0,Texp])
    plt.ylim([intmin,intmax])

    histogram = np.histogram(intbinlist,int(np.max(intbinlist)),[0,int(np.max(intbinlist))])
    
    ax1 = plt.subplot(gs[1])
    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]),color=col)
    plt.xlabel('occurrence')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.ylim([intmin,intmax])
    
    ax2 = plt.subplot(gs[2])
    p2 = ax2.plot(np.arange(len(intbinlist))*binwidth,taubinlist,'.',markersize=1.5,color=col)
    plt.xlabel('time (s)')
    plt.ylabel('lifetime (ns)')
    plt.xlim([0,Texp])
    plt.ylim([taumin,taumax])

    histogram = np.histogram(taubinlist,taumax-taumin,[taumin,taumax])
    
    ax3 = plt.subplot(gs[3])
    ax3.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]),color=col)
    plt.xlabel('occurrence')
    plt.ylabel('lifetime (ns)')
    plt.ylim([taumin,taumax])
    
    plt.hold(True)
    

def BinIntensity(microtimes0,times0,limits0,microtimes1,times1,limits1,dtmicro,dtmacro,onintlim,offintlim):
    ## select only data with high or low intensity

    plt.title('total decay')
    tauave = GetLifetime(np.append(microtimes0,microtimes1),dtmicro,dtmacro,200e-9,-1)

    nrbins = len(limits0)
    inttrace = np.array([limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr] for binnr in range(nrbins-1)])

    # find photons in on period
    onphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] >= onintlim])
    onphotonlist0 = np.concatenate(onphotonlist0).ravel()
    onphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] >= onintlim])
    onphotonlist1 = np.concatenate(onphotonlist1).ravel()

    onmicrotimes0 = np.array([microtimes0[i] for i in onphotonlist0])
    onmicrotimes1 = np.array([microtimes1[i] for i in onphotonlist1])
    ontimes0 = np.array([times0[i] for i in onphotonlist0])
    ontimes1 = np.array([times1[i] for i in onphotonlist1])
    plt.title('on decay')
    ontauave = GetLifetime(np.append(onmicrotimes0,onmicrotimes1),dtmicro,dtmacro,200e-9,-1)

    # find photons in off period
    offphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] < offintlim])
    offphotonlist0 = np.concatenate(offphotonlist0).ravel()
    offphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] < offintlim])
    offphotonlist1 = np.concatenate(offphotonlist1).ravel()

    offmicrotimes0 = np.array([microtimes0[i] for i in offphotonlist0])
    offmicrotimes1 = np.array([microtimes1[i] for i in offphotonlist1])
    offtimes0 = np.array([times0[i] for i in offphotonlist0])
    offtimes1 = np.array([times1[i] for i in offphotonlist1])
    plt.title('off decay')
    offtauave = GetLifetime(np.append(offmicrotimes0,offmicrotimes1),dtmicro,dtmacro,10e-9,-1)
    
    return(onmicrotimes0,offmicrotimes0,ontimes0,offtimes0,onmicrotimes1,offmicrotimes1,ontimes1,offtimes1)

def SliceHistogram(microtimes0,times0,limits0,macrotimescycle0,microtimes1,times1,limits1,macrotimescycle1,dtmicro,dtmacro,Imin,Imax):
    ## select only data with intensity between Imin and Imax

    nrbins = len(limits0)
    inttrace = np.array([limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr] for binnr in range(nrbins-1)])

    # find photons in bins with intensities in (Imin,Imax] range
    onphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if Imin < inttrace[binnr] <= Imax])
    onphotonlist0 = np.concatenate(onphotonlist0).ravel()
    onphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if Imin < inttrace[binnr] <= Imax])
    onphotonlist1 = np.concatenate(onphotonlist1).ravel()
      
    onmicrotimes0 = np.array([microtimes0[i] for i in onphotonlist0])
    onmicrotimes1 = np.array([microtimes1[i] for i in onphotonlist1])
    ontimes0 = np.array([times0[i] for i in onphotonlist0])
    ontimes1 = np.array([times1[i] for i in onphotonlist1])
    onmacrotimescycle0 = np.array([macrotimescycle0[i] for i in onphotonlist0])
    onmacrotimescycle1 = np.array([macrotimescycle1[i] for i in onphotonlist0])
    
    # count nr of time bins with intensities corresponding to slice intensity
    onbincount = 0
    for binnr in range(nrbins-1):
        if Imin < inttrace[binnr] <= Imax:
            onbincount +=1

    return(onmicrotimes0,ontimes0,onmacrotimescycle0,onmicrotimes1,ontimes1,onmacrotimescycle1,onbincount)



def MaxLikelihoodFit(tlist,ylist,istart,iend,bgcpb,plotbool=False):
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
    initParams = [np.max(ydata), 25] #initial guess for A and tau
    results = minimize(MaxLikelihoodFunction, initParams, args=(xdata,ydata,bgcpb),method='Nelder-Mead') # minimize the negative of the maxlikelihood function instead of maximimizing
    A1est = results.x[0] # get results of fit, A
    tau1est = results.x[1] # get results of fit, tau
#    A2est = results.x[2] # get results of fit, A
#    tau2est = results.x[3] # get results of fit, tau

#    if plotbool == True:
#        yest = np.array([A1est*np.exp(-(xdata[i]-xdata[0])/tau1est)+A2est*np.exp(-(xdata[i]-xdata[0])/tau2est)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        
    return(tau1est,A1est)#,tau2est,A2est)
    
def MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,plotbool=False):
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
    initParams = [np.max(ydata), 25] #initial guess for A and tau
    results = minimize(MaxLikelihoodFunction_c, initParams, args=(xdata,ydata,bgcpb),method='Nelder-Mead') # minimize the negative of the maxlikelihood function instead of maximimizing
    Aest = results.x[0] # get results of fit, A
    tauest = results.x[1] # get results of fit, tau

#    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        


    if plotbool == True:
        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
        plt.figure()
        plt.plot(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
        plt.xlim([xdata[1],xdata[-1]])
        plt.show()        
        
    return(tauest,Aest)

def MaxLikelihoodFunction(params,xdata,ydata,const): 
    # max likelihood function for A*exp(-t/tau), needed in function MakLikelihoodFit
    # params = [A,tau]
    A1 = params[0]
    tau1 = params[1]  
#    A2 = params[2]
#    tau2 = params[2]
    E = 0;
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)
        
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!


def processInput(tbinnr):
    microtimes = np.append(microtimesin[limits[tbinnr]:limits[tbinnr+1]])
    [ylist,xlist] = np.histogram(microtimes,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])    
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
#    plt.clf()
#    plt11 = plt.figure(11)
    initParams = [np.max(ylist), 25]
    result=MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,initParams,plotbool=False)
#    Ampl[tbinnr]=result[0]
#    Alpha[tbinnr]=result[1]
#    Theta[tbinnr]=result[2]
    return(result)


def InVoltage(t,Freq,VPP,VOffset,Verror):
    Period=1/Freq
    Bool1=t<=Period/4
    Bool2=scipy.logical_and(t>Period/4,t<=3*Period/4)
    Bool3=t>3*Period/4
    InVoltage=(VPP/2*t/(Period/4)+VOffset-Verror)*Bool1+(VPP/2-VPP/2*(t-Period/4)/(Period/4)+VOffset+Verror)*Bool2+(-VPP/2+VPP/2*(t-3*Period/4)/(Period/4)+VOffset-Verror)*Bool3
    return(InVoltage)
        
    
def importASC(filename):
    """Import ASC files exported from Andor Solis.
    
    Parameters
    ----------
    filename: String with the complete path to the file.

    Returns
    -------
    raw_data : 2D numpy array containing the counts [NxN]
    x: horizontal pixel number (image) or corresponding wavelength (spectrum), depending on input file type [Nx1]
    y: vertical pixel number [Nx1]
    info: dictionary with additional information such as exposure time

    Notes
    -----
    Export the file from andor as comma (,) separated file with appended aquisition information.

    References
    ----------
    

    Examples
    --------
    >>> [data,x,y,info] = importASC('X:/temp/LAB_DATA/Andor Spectrometer/Felipe/file.asc')
    
    """
    a = pd.read_csv(filename, header = None, low_memory = False)
    
    nr_px = 1024
    x = a.iloc[0:nr_px,0].str.strip().values.astype(float)
    y = np.arange(1,nr_px+1)
    raw_data = a.iloc[0:nr_px,1:-1].values.astype(int)

    
    info = {}
    
    info['nr_px'] = 1024
    
    info['expTime'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1031:1032,0].str.strip().values[0]).group(0))
    #info['EMgain'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1046:1047,0].str.strip().values[0]).group(0))
    #info['centerWL'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1055:1056,0].str.strip().values[0]).group(0))
    #info['gratingDensity'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1056:1057,0].str.strip().values[0]).group(0))
    
    return [raw_data, x, y, info]
def importPLE(filename):
    a = pd.read_csv(filename, delimiter=';', header = 20, low_memory = False)
    datalength=a.shape[0]
    wavelength=a.iloc[0:datalength,0].values
    intensity=a.iloc[0:datalength,1].values
    return [wavelength,intensity]



def MakeG2(times0,times1,dtmicro,g2restime=8e-9,nrbins=200):
    i0=0
    i1=0
    lim1=0
    g2 = np.zeros(2*nrbins)
    #g2B = np.zeros(2*nrbins)
    #g2C = np.zeros(2*nrbins)
    #blindB = 2e-9
    #blindC = 5e-9

    g2res = g2restime/dtmicro #transform g2restime [s] to g2res [microtime units]
    #blindB = blindB/tmicro
    #blindC = blindC/tmicro

    # correlate det0 with det1 (positive time differences)
    for i0 in range(len(times0)):
        t0 = times0[i0]
        i1 = 0
        q = 0
        while q == 0: 
            if lim1 + i1 < len(times1): # check if we've already reached end of photon stream on det1
                dt = times1[lim1+i1]-t0 
                if dt < 0: # find index lim1 of first photon on det1 that came after photon i0 on det0
                    lim1 = lim1 + 1
                else:
                    binnr = int(dt/g2res) # calculate binnr that corresponds to dt
                    if binnr < nrbins: # check if time difference is already large enough to stop correlation
                        g2[nrbins + binnr] += 1 # increase counter in corresponding bin by one
                        #if microtimes0[i0] > blindB and microtimes1[lim1+i1] > blindB:poi
                            #g2B[nrbins + binnr] += 1
                        #if microtimes0[i0] > blindC and microtimes1[lim1+i1] > blindC:
                            #g2C[nrbins + binnr] += 1
                        i1 = i1 + 1 # look at next photon on det1
                    else:
                        q = 1 # dt larger than maximum correlation width. stop. 
            else:
                q = 1 # end of photon stream on det1 reached. stop.

    # correlate det1 with det0 (positive time differences)
    lim1=0
    for i0 in range(len(times1)):
        t0 = times1[i0]
        i1 = 0
        q = 0
        while q == 0:
            if lim1 + i1 < len(times0):
                dt = times0[lim1+i1]-t0
                if dt < 0:
                    lim1 = lim1 + 1
                else:
                    binnr = int(dt/g2res)
                    if binnr < nrbins:
                        g2[nrbins - 1 - binnr] += 1
                        #if microtimes0[lim1+i1] > blindB and microtimes1[i0] > blindB:
                        #    g2B[nrbins - 1 - binnr] += 1
                        #if microtimes0[lim1+i1] > blindC and microtimes1[i0] > blindC:
                        #    g2C[nrbins - 1 - binnr] += 1
                        i1 = i1 + 1
                    else:
                        q = 1
            else:
                q = 1
                                
    g2tlist = np.arange(-g2res*dtmicro*(nrbins-0.5),g2res*dtmicro*nrbins,g2restime)*1e9
    plt.plot(g2tlist,g2)
    #plt.plot(g2tlist,g2B)
    #plt.plot(g2tlist,g2C)
    plt.title('g(2) correlation')
    plt.xlabel('delay (ns)')
    plt.ylabel('occurence (a.u.)')
    plt.ylim([0,max(g2)])
    plt.show()

    return(g2tlist,g2,g2restime,nrbins)
# In[25]:

### GENERAL EVALUATION OF TTTR data     

# parameters (use forward slashes!)
Debugmode=False     #Set to true to make plots in the individual sections of the code to check that everything is working properly
folder = 'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/20190911_PLE_PM111/'
#folder = 'E:/LAB_DATA/Robert/20191003_PLE_realigned/measuremens/PM111/'
namelist = ['Dot2_405']    #Paste the file that ends in _tttrmode.out in here without the .out
#MCLdata=load_obj("Sample1_x174_y187_z11_range20_pts100_MCLdata", folder )
#settingsfile=namelist[0].rsplit('_',1)[0]+'_HHsettings'     #HHsettings file should have the same file name apart from this last identifier
#HHsettings=load_obj(settingsfile, folder )
Texp = 240 # total time [s] of measurements
binwidth = 0.01 # width of time bins to create intensity trace [s]
nrbins = 20 # intensity bin to separate intensity trace in (nrbins) to separately evaluate different intensity levels
histbinmultiplier = 1 #(histbinmultiplier)*dtmicro = bin size for lifetime histogram
dttau = 50e-9 # end time of lifetime fit [s]. Note: this is not the width but the end time! NEED TO CHANGE LATER!

savebool = False;
# subtractbg = True; # not used?! 
singlebintaubool = False;
g2bool = False;

colmap = np.array([[0.0,0.0,0.0],
                   [0.6,0.6,1.0],
                   [0.3,0.3,1.0],
                   [0.0,0.0,1.0],
                   [1.0,0.6,0.6],
                   [1.0,0.3,0.3],
                   [1.0,0.0,0.0],
                  ])

# init some arrays and values
nrmeas = len(namelist) #nr of data files
nrtbins = int(Texp/binwidth) # nr of bins in intensity trace
Ilimslist = np.ndarray(shape=(nrmeas,nrbins),dtype=float)
Alphabinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save Alpha of each intensity bin
Thetabinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save Alpha of each intensity bin
Abinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save maximum histogram amplitudes (at t=t0)
taubinlist = np.zeros(nrtbins-1,dtype=float) # init list to save lifetimes of each time bin (binwidth)
intbinlist = np.zeros(nrtbins-1,dtype=float) # init list to save intensity per bin

for measnr in range(nrmeas): #for each measurement file

    # load data
    data = ImportT3(folder + namelist[measnr] + '.ptu') # import data from .out file or .ptu file
    Texp = round(data[8]*data[1]*1024,0) # macrotime values can store up to 2^10 = 1024 before overflow. overflows*1024*dtmacro gives experiment time [s]
    print('averaged cps on det0 and det1:',np.array(data[6:8])/Texp)
    print('experimental time in s:',Texp)

    # compensate detector offset; here we also make the microtimetrace
    [microtimes0,microtimes1,times0,times1,dtmicro,dtmacro,decaytlist,decayylist] = ShiftPulsedData(data[2],data[4],data[3],data[5],data[0],data[1]) #decaytlist and decayylist are the two variables you want to check for the modulation trace
    histbinmultiplier=4
    [ylist,xlist] = np.histogram(microtimes0,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])    
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
    
    nrnewbins=int(dtmacro/(dtmicro*histbinmultiplier))
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
    istart=0
    iend=max(tlist)
    bgcounts=Texp*80
    MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction) # implement c function
#    GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=400e-9,tstart=20e-9,binwidth=1,ybg=-1,plotbool=True,method='ML_c')
    bgcpb=bgcounts/nrnewbins
#    plt.semilogy(tlist,ylist)
#        plt.clf()


    # Potentially it makes sense to bin this for the fitting
#    plt.show()
    if savebool == True:        
        np.save(folder + namelist[measnr] + '_decay_tlist', decaytlist)
        np.save(folder + namelist[measnr] + '_decay_ylist', decayylist)   
    tmax = decaytlist[decayylist.argmax()] # find histogram time with max photon count [ns]
    
    # get number of first photon per bin for both detectors (APD1 = 0, APD2 = 1)
    limits0 = HistPhotons(times0*dtmicro,binwidth,Texp) #gives index of first photon of det0 in each bin
    limits1 = HistPhotons(times1*dtmicro,binwidth,Texp)

    # make an intensity trace and find Imax
    inttrace = MakeIntTrace(limits0,limits1,binwidth,Texp)  #this gives the trace of integrated counts vs time where we see the blinking. You might need plt.show() to see it
    Imax = np.max(inttrace) # search overall maximum intensity per intensity bin
    Ilims = np.arange(0,Imax*0.95+1,Imax*0.95/nrbins,dtype=float) # separate intensities into (nrbins) intensity bins
    Ilimslist[measnr,:] = np.array([Ilims[binnr]*0.5+Ilims[binnr+1]*0.5 for binnr in range(nrbins)]) # get centered intensity value per intensity bin




#%% Lifetime vs Intensity
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
binwidth=0.01
#taulist=np.zeros(len(limits0)-1)
#tauav=np.zeros(len(limits0)-1)
#Alist=np.zeros(len(limits0)-1)
#photonspbin=np.zeros(len(limits0)-1)
#ybglist=np.zeros(len(limits0)-1)
#buff=np.zeros(len(limits0)-1)
#meanex=np.zeros(len(limits0)-1)
#stdex=np.zeros(len(limits0)-1)
histbinmultiplier=1
plt.figure()
lifetime=GetLifetime(microtimes0,dtmicro,dtmacro,450e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,method='ML_c')
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))

#test=np.zeros((wavelbins,len(limits)-1))
for tbinnr in tqdm(range(len(limits0)-1)):
    microtimes = microtimes0[limits0[tbinnr]:limits0[tbinnr+1]]
    if len(microtimes)>10:
        [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,150e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 
    else:
        [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,20e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 
    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9
    photonspbin[tbinnr]=len(microtimes)
    #using th
fig,ax1 = plt.subplots()
#ax1.plot(times0[limits0[0:len(limits0)-1]]*dtmicro,photonspbin,'b')
ax1.plot(times0[limits0[0:tbinnr-1]]*dtmicro,photonspbin[0:tbinnr-1],'b')
#ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')
#ax1.set_xlim([0,13])
#ax1.set_ylim([0,0.2*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
#ax2.plot(times0[limits0[0:len(limits0)-1]]*dtmicro,tauav,'r')
ax2.plot(times0[limits0[0:tbinnr-1]]*dtmicro,tauav[0:tbinnr-1],'r')
#ax2.set_ylim([586,588])
ax2.set_ylim([0,80])
ax2.set_ylabel('Lifetime (ns)s', color='r')
ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.set_title('"Dot" 4')



#%% Plot FLID map
plt.figure()
plt.hist2d(taulist,photonspbin,(100,np.max(photonspbin)),range=[[0,60],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()
plt.plot(np.arange(0,50),np.arange(0,50)*505/27+8,'w')

