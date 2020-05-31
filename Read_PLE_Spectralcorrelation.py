# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:44:49 2019

@author: rober
"""

import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
import numpy.matlib
import multiprocessing
from operator import sub
from joblib import Parallel, delayed
import scipy#, lmfit
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
        macrotimes0 = np.zeros(nrrec,dtype='int64'); #Macrotime of photons on detector 0 in integer units
        microtimes0 = np.zeros(nrrec,dtype='int64'); #Microtime of photons on detector 0 in integer units
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
        macrotimecycle0 = np.zeros(nrrec,dtype='int64'); #Time since start of current galvo cycle of photons on detector 0 in integer units (macrotime)
        macrotimecycle1 = np.zeros(nrrec,dtype='int64');
        cyclenumber0 = np.zeros(nrrec,dtype='int64'); #Number of galvo cycle of of photons on detector 0
        firstphotonincycle0 = np.zeros(nrrec,dtype='int64'); #Index of the first photon on a given galvo cycle
        overflows = 0
        nrphotons0 = 0
        nrphotons1 = 0
        nrfireA = 0
        nrfireB = 0
        nrfireC = 0
        nrfireD = 0
        prevchann = 0
        lastcyclestarttime = 0
        currentcycle = 0
        
        for i in range(nrrec):
            entry = f.read(4)
            channel = struct.unpack("I",entry)[0] >> 25 # read channel number, first 7 bits
            if channel == 0:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimes0[nrphotons0] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimes0[nrphotons0] = microtime
                cyclenumber0[nrphotons0] = currentcycle
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
                cyclenumber0[nrphotons0] = currentcycle
                currentcycle += 1
                firstphotonincycle0[currentcycle] = nrphotons0 + 1
                lastcyclestarttime = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireD[nrfireD] = microtime
                nrfireD += 1                              
            else:
                print('bad channel:',channel)
                
    microtimes0 = microtimes0[:nrphotons0]
    macrotimes0 = macrotimes0[:nrphotons0]
    cyclenumber0 = cyclenumber0[:nrphotons0]
    firstphotonincycle0 = firstphotonincycle0[:currentcycle]
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

    return [dtmicro, dtmacro, microtimes0, macrotimes0, microtimes1, macrotimes1, nrphotons0,nrphotons1,overflows,microtimesfireA,macrotimesfireA,nrfireA,microtimesfireB,macrotimesfireB,nrfireB,macrotimesfireC,nrfireC,macrotimesfireD,nrfireD,macrotimecycle0,macrotimecycle1,cyclenumber0,firstphotonincycle0]

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
    model = A1*np.exp(-(xdata-xdata[0])/tau1)+const
    model [model <= 0] = 1e-10
#    A2 = params[2]
#    tau2 = params[2]
    E = 0;
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        # E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)
        E = E + ydata[i]*np.log(model[i])-(model[i])
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!

def MaxLikelihoodFunction_biexp(params,xdata,ydata,const):
    A1 = params[0]
    tau1 = params[1]
    A2 = params[2]
    tau2 = params[3]
    model = A1*np.exp(-(xdata-xdata[0])/tau1)+A2*np.exp(-(xdata-xdata[0])/tau2)+const
    model [model <= 0] = 1e-10
    E = 0;
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        # E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)
        E = E + ydata[i]*np.log(model[i])-(model[i])
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
        
    
def importASC(filename,nr_px=1024):
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
    
    x = a.iloc[0:nr_px,0].str.strip().values.astype(float)
    y = np.arange(1,nr_px+1)
    raw_data = a.iloc[0:nr_px,1:-1].values.astype(int)

    
    info = {}
    
    info['nr_px'] = nr_px
    
    info['expTime'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[nr_px+7:nr_px+8,0].str.strip().values[0]).group(0))
    # info['expTime'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1031:1032,0].str.strip().values[0]).group(0))
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


def eVtonm(x):
    return 1240/x
def nmtoeV(x):
    return 1240/x
# In[25]:

### GENERAL EVALUATION OF TTTR data     

# parameters (use forward slashes!)
Debugmode=False     #Set to true to make plots in the individual sections of the code to check that everything is working properly


#folder = 'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/20200303_PM111_PLE_withMartijn/'
folder = 'E:/LAB_DATA/Martijn/20200310_PM111_specdiffusion/'
namelist = ['QD2_4p8MHr_200Hz_500mVpp_60mVoff_pluscamera']    #Paste the file that ends in _tttrmode.out in here without the .out


#MCLdata=load_obj("Sample1_x174_y187_z11_range20_pts100_MCLdata", folder )
#settingsfile=namelist[0].rsplit('_',1)[0]+'_HHsettings'     #HHsettings file should have the same file name apart from this last identifier
#HHsettings=load_obj(settingsfile, folder )
Texp = 240 # total time [s] of measurements
binwidth = 0.01 # width of time bins to create intensity trace [s]
nrbins = 5 # intensity bin to separate intensity trace in (nrbins) to separately evaluate different intensity levels
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
    MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFit) # implement c function
    GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=200e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,method='ML')
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


#%% Wavelength calibration
Voltage=np.array([0,40,80,120,160,200,240])
Wavelength_calib=np.array([541.5,523,504.4,485.9,467.5,449.1,430.6])
calibcoeffs=np.polyfit(Voltage,Wavelength_calib,1)
if Debugmode==True:
    plt.figure()
    plt.plot(Voltage,calibcoeffs[1]+Voltage*calibcoeffs[0],Voltage,Wavelength_calib,'*')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Wavelength (nm)')
    plt.legend(['Linear Fit','Measurement'])
#% Time to wavelength calibration
# Vpp=240
Vpp=200
# Vpp=240
#Vpp=500 #Peak to peak voltage of galvo input
#Freq=1/(np.max(data[19]*dtmacro))
Freq=200 #Frequency of galvo input

#Voffset=60 #Voltage offset of galvo input

Voffset=0
Voffset=160
Voffset=80
Voffset=120
Voffset=-40

Verror=0 #Error to correct for difference between forward and backward
tIn=np.linspace(0,1/Freq,200)     #in [s]
VIn=InVoltage(tIn,Freq,Vpp,Voffset,Verror)
WavelengthIn=calibcoeffs[1]+VIn*calibcoeffs[0]
#plt.plot(tIn,WavelengthIn,'r')


#% Read spectrum
[ylistspec,xlistspec] = np.histogram(data[19],608)
tlistspec = (xlistspec[:-1]+0.5*(xlistspec[1]-xlistspec[0]))*dtmacro
if Debugmode==True:
    fig,ax1 = plt.subplots()
    ax1.plot(tlistspec,ylistspec,'b')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Intensity', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(tIn,WavelengthIn,'r')
    ax2.set_ylabel('Wavelength (nm)', color='r')
    ax2.tick_params('y', colors='r')

#% Plot raw data vs wavelength
#Voffset=18 #for 50Hz
#Verror=84  #for 250Hz

#Verror=55
Verror=22 
# Verror=28.5
# Verror=26.2

# Verror=27
#Verror=18 #for 50Hz
Wavelengthspec=calibcoeffs[1]+InVoltage(tlistspec,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
#InVoltage function is the function which makes a voltage out of the set parameters. Therefore, the wavelength calibration can make a wavelength out of the voltage of the photons.
if Debugmode==True:
#    plt.figure()
    plt.plot(Wavelengthspec,ylistspec)#/max(ylistspec))
#    plt.xlim([440,620])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Photons per bin')
# For now I do the correction by hand. I made some measurements from which I can get at least the voltage to wavelength correction
# 
#    
##%%Wavelength calibration from diode and osci

#reffolder='C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/20191004_Oscilloscope_checks/redone/'
#GalvoPos = pd.read_csv(reffolder+"Galvo_pos.csv",header=None,names=["Time","Voltage"])
#Wavelength = calibcoeffs[1]+GalvoPos.Voltage/2*1000*calibcoeffs[0]
#Diode = pd.read_csv(reffolder+"Diode_10kOhm.csv",header=None,names=["Time","Voltage"])
#Diode_responsivity = pd.read_csv(reffolder+"DET100A_Responsivity.csv",header=None,names=["Wavelength","Responsivity"])
#DDiode_fast = pd.read_csv(reffolder+"Diode_1kOhm.csv",header=None,names=["Time","Voltage"])
#posshift=3
#plt.plot(Wavelength[posshift:-1],Diode.Voltage[0:-1-posshift])

#Wavelarray=np.arange(420,620)
#Diode_resp_interp=interp1d(Diode_responsivity.Wavelength,Diode_responsivity.Responsivity)
#refspecdiode=interp1d(Wavelength[posshift:-1],(Diode.Voltage[0:-1-posshift]+0.0009)/(Diode_resp_interp(Wavelength[posshift:-1]))/np.max((Diode.Voltage[0:-1-posshift]+0.0009)/(Diode_resp_interp(Wavelength[posshift:-1]))))
#plt.plot(Wavelarray,refspecdiode(Wavelarray))

#% Get wavelength calibration from reflection measurement

#reffolder='C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/20191030_PLE_PM111/'
# reffolder='E:/LAB_DATA/Robert/20200122_cryo_Perovskites/'
reffolder = folder
calibspec=importASC(reffolder+'Spec_1p95MHz_200Hz_500mVpp_60mVoff.asc') #this is the file of the reference spectrum.
refspec=np.mean(calibspec[0][:,490:510],1)*1.2398/calibspec[1]

refspec=refspec-np.mean(refspec[925:len(refspec)])
refspec=refspec/np.max(refspec)
refspec=savgol_filter(refspec, 23, 3)
interprefspec= interp1d(calibspec[1], refspec,kind='cubic',fill_value='extrapolate')
if Debugmode==True:
    plt.figure()
    plt.plot(calibspec[1],refspec,Wavelengthspec,interprefspec(Wavelengthspec),'.')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(['Measured backreflection','Evaluated at measurement points'])
#% Plot raw data against reference spec
    fig,ax1 = plt.subplots()
    ax1.plot(Wavelengthspec,ylistspec/max(ylistspec),'b')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity', color='b')
    ax1.tick_params('y', colors='b')
    #ax1.set_ylim([0,0.11])
    ax2 = ax1.twinx()
    ax2.plot(calibspec[1],refspec,'r')
    ax2.set_ylabel('Excitation Intensity (nm)', color='r')
    ax2.tick_params('y', colors='r')
    #plt.plot(calibspec[1],refspec)#,calibspec[1],interprefspec(calibspec[1]))
    plt.xlim([400,630])
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Reflected Intensity')
    plt.legend(['Measured spectrum','Backref of excitation'])


#% Plot corrected spectrum
plt.figure()
plt.plot(Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))


plt.xlim([np.min(Wavelengthspec),np.max(Wavelengthspec)])
plt.ylim([0,np.max(ylistspec/interprefspec(Wavelengthspec))])
#plt.xlim([510,595])
#plt.ylim((0, 6000))

plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('Intensity')
plt.title(namelist[0])

#% Plot dispersion PLE measurement
#RefPLE=importPLE(folder+'PM151_dil_Em622_Ex300-600_step1_Dwell0.3_ExBW5_EmBW0.3_Refcor_Emcor.txt')
#fig,ax1 = plt.subplots()
#ax1.plot(Wavelengthspec,ylistspec/interprefspec(Wavelengthspec),'b')
#ax1.set_xlabel('Excitation Wavelength (nm)')
#ax1.set_ylabel('Film Micro PLE', color='b')
#ax1.set_xlim([430,650])
#ax1.set_ylim([0,0.3*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
#ax1.tick_params('y', colors='b')
#ax2 = ax1.twinx()
#ax2.plot(RefPLE[0],RefPLE[1],'r')
#ax2.set_ylim([0,1.5e6])
#ax2.set_ylabel('Dispersion PLE', color='r')
#ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
#ax1.set_title('"Dot" 2')

#%% Numba approach
def repeatvector(vecin,repeattimes):
    return np.repeat(vecin,repeattimes) 

#%% Parallel Loops
def repeatvecparallel(k):
    return(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],len(data[22])-tau,1))



#%%
#tau = 1
#lambdas0tau1=Parallel(n_jobs=-1, max_nbytes=None)(delayed(repeatvecparallel)(k) for k in tqdm(range(0,len(data[22][:-tau]))))

#%%
if Debugmode == True:
        
    lambdas0tau1=[]
    lambdas1tau1=[]
    tau = 1
    repeatvector_c = nb.jit(nopython=True)(repeatvector)                         
                
    for k in tqdm(range(0,len(data[22][:-tau-1]))): 
        # lambdas0tau1=np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], len(data[22])-tau,1)
        # lambdas0tau1=np.matlib.repeat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], len(data[22])-tau,axis=0)
        if  data[22][k+tau+1]-1-(data[22][k+tau])>0 and data[22][k+1]-1-(data[22][k])>0:
            lambdas0tau1.append(repeatvector_c(calibcoeffs[1]+InVoltage(data[19][range(data[22][k],data[22][k+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+tau+1]-1-(data[22][k+tau])))
            lambdas1tau1.append(np.ndarray.flatten(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+1]-1-(data[22][k]),1))) #for 250Hz)

            # lambdas0tau1=np.append(lambdas0tau1,repeatvector_c(calibcoeffs[1]+InVoltage(data[19][range(data[22][k],data[22][k+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+tau+1]-1-(data[22][k+tau])))
            # lambdas1tau1=np.append(lambdas1tau1,np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+1]-1-(data[22][k]),1))
            
            # for j in range(data[22][k],data[22][k+1]-1):
            # for m in range(data[22][k+tau],data[22][k+1+tau]-1):
                # lambdas0tau1.append(calibcoeffs[1]+InVoltage(data[19][j]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
        #         lambdas1tau1.append(calibcoeffs[1]+InVoltage(data[19][m]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
             
                       
    
    # lambdas0tau100=[]
    # lambdas1tau100=[]
    # tau = 100
    # # for k in range(0,len(data[22][:5])): 
    # #     lambdas0=lambdas0.extend(calibcoeffs[1]+InVoltage(data[19][k],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    # #     # lambdas1=lambdas1.extend(calibcoeffs[1]+InVoltage(data[19][m],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
                                  
                
    # for k in tqdm(range(0,len(data[22][:-tau]))): 
    #     for j in range(data[22][k],data[22][k+1]-1):
    #         for m in range(data[22][k+tau],data[22][k+1+tau]-1):
    #             lambdas0tau100.append(calibcoeffs[1]+InVoltage(data[19][j]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    #             lambdas1tau100.append(calibcoeffs[1]+InVoltage(data[19][m]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
   

    lambdas0tau1000=[]
    lambdas1tau1000=[]
    tau = 1000
    for k in range(0,len(data[22][:-tau-1])): 
         if  data[22][k+tau+1]-1-(data[22][k+tau])>0 and data[22][k+1]-1-(data[22][k])>0:
            lambdas0tau1000.append(repeatvector_c(calibcoeffs[1]+InVoltage(data[19][range(data[22][k],data[22][k+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+tau+1]-1-(data[22][k+tau])))
            lambdas1tau1000.append(np.ndarray.flatten(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+1]-1-(data[22][k]),1))) #for 250Hz)      
            
    lambdas0tau10000=[]
    lambdas1tau10000=[]
    tau = 10000
    for k in range(0,len(data[22][:-tau-1])): 
         if  data[22][k+tau+1]-1-(data[22][k+tau])>0 and data[22][k+1]-1-(data[22][k])>0:
            lambdas0tau10000.append(repeatvector_c(calibcoeffs[1]+InVoltage(data[19][range(data[22][k],data[22][k+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+tau+1]-1-(data[22][k+tau])))
            lambdas1tau10000.append(np.ndarray.flatten(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0], data[22][k+1]-1-(data[22][k]),1))) #for 250Hz)
    #     lambdas0=lambdas0.extend(calibcoeffs[1]+InVoltage(data[19][k],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    #     # lambdas1=lambdas1.extend(calibcoeffs[1]+InVoltage(data[19][m],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
                                  
                
    # for k in tqdm(range(0,len(data[22][:-tau]))): 
    #     for j in range(data[22][k],data[22][k+1]-1):
    #         for m in range(data[22][k+tau],data[22][k+1+tau]-1):
    #             lambdas0tau10000.append(calibcoeffs[1]+InVoltage(data[19][j]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    #             lambdas1tau10000.append(calibcoeffs[1]+InVoltage(data[19][m]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    
    
    # lambdas0tau15000=[]
    # lambdas1tau15000=[]
    # tau = 15000
    # # for k in range(0,len(data[22][:5])): 
    # #     lambdas0=lambdas0.extend(calibcoeffs[1]+InVoltage(data[19][k],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    # #     # lambdas1=lambdas1.extend(calibcoeffs[1]+InVoltage(data[19][m],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
                                  
                
    # for k in tqdm(range(0,len(data[22][:200]))): 
    #     for j in range(data[22][k],data[22][k+1]-1):
    #         for m in range(data[22][k+tau],data[22][k+1+tau]-1):
    #             lambdas0tau15000.append(calibcoeffs[1]+InVoltage(data[19][j]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    #             lambdas1tau15000.append(calibcoeffs[1]+InVoltage(data[19][m]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]) #for 250Hz)
    
    #%%        
    lambdas0tau1array=np.concatenate(lambdas0tau1).ravel()
    lambdas1tau1array=np.concatenate(lambdas1tau1).ravel()
    lambdas0tau1000array=np.concatenate(lambdas0tau1000).ravel()
    lambdas1tau1000array=np.concatenate(lambdas1tau1000).ravel()
    lambdas0tau10000array=np.concatenate(lambdas0tau10000).ravel()
    lambdas1tau10000array=np.concatenate(lambdas1tau10000).ravel()
    #%%
    plt.figure()
    hmattau1=plt.hist2d(lambdas0tau1array,lambdas1tau1array,bins=200,range=[[500,600],[500,600]])
    plt.colorbar()
    
    # plt.figure()
    # hmattau100=plt.hist2d(np.array(lambdas0tau100),np.array(lambdas1tau100),bins=200,range=[[500,600],[500,600]])
    # plt.colorbar()
    
    plt.figure()
    hmattau10000=plt.hist2d(np.array(lambdas0tau10000array),np.array(lambdas1tau10000array),bins=200,range=[[500,600],[500,600]])
    plt.colorbar()
    
    plt.figure()
    hmattau1000=plt.hist2d(np.array(lambdas0tau1000array),np.array(lambdas1tau1000array),bins=200,range=[[500,600],[500,600]])
    plt.colorbar()
    # plt.figure()
    # hmattau15000=plt.hist2d(np.array(lambdas0tau15000),np.array(lambdas1tau15000),bins=200,range=[[500,600],[500,600]])
    # plt.colorbar()
    
    #%%
    lammattau1=hmattau1[0]
    plt.figure()
    tau=1
    hvec=plt.hist(calibcoeffs[1]+InVoltage(data[19][:data[22][-tau-1]-1]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],bins=200,range=[500,600])
    lamvectau1=hvec[0]
    
    # lammattau100=hmattau100[0]
    # plt.figure()
    # hvec=plt.hist(calibcoeffs[1]+InVoltage(data[19][:data[22][201]-1]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],bins=200,range=[500,600])
    # lamvectau100=hvec[0]
    
    lammattau1000=hmattau1000[0]
    plt.figure()
    tau=1000
    hvec=plt.hist(calibcoeffs[1]+InVoltage(data[19][:data[22][-tau-1]-1]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],bins=200,range=[500,600])
    lamvectau1000=hvec[0]
    
    
    lammattau10000=hmattau10000[0]
    plt.figure()
    tau=10000
    hvec=plt.hist(calibcoeffs[1]+InVoltage(data[19][:data[22][-tau-1]-1]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],bins=200,range=[500,600])
    lamvectau10000=hvec[0]
    
    # lammattau15000=hmattau15000[0]
    # plt.figure()
    # hvec=plt.hist(calibcoeffs[1]+InVoltage(data[19][:data[22][201]-1]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],bins=200,range=[500,600])
    # lamvectau15000=hvec[0]
    #%%
# plt.figure()
lmmattau1norm=np.divide(lammattau1,np.outer(lamvectau1,lamvectau1)+1)
# plt.imshow(lmmattau1norm,vmin=np.mean(lmmattau1norm)*0.5,vmax=np.mean(lmmattau1norm)*1.5,extent=[500,600,600,500])
# plt.colorbar()
# plt.gca().invert_yaxis()



# plt.figure()
lmmattau10000norm=np.divide(lammattau10000,np.outer(lamvectau10000,lamvectau10000)+1)
lmmattau1000norm=np.divide(lammattau1000,np.outer(lamvectau1000,lamvectau10000)+1)
# plt.imshow(lmmattau10000norm,vmin=np.mean(lmmattau10000norm)*0.5,vmax=np.mean(lmmattau10000norm)*1.5,extent=[500,600,600,500])
# plt.colorbar()
# plt.gca().invert_yaxis()
    
    # testmat=np.outer(lamvectau100,lamvectau100)
    # plt.figure()
    # plt.imshow(np.divide(lammattau100,testmat)*1000,vmin=0.5,vmax=15)
    # plt.colorbar()
    
    # testmat=np.outer(lamvectau10000,lamvectau10000)
    # plt.figure()
    # plt.imshow(np.divide(lammattau10000,testmat)*1000,vmin=0.5,vmax=15)
    # plt.colorbar()
    
    # testmat=np.outer(lamvectau15000,lamvectau15000)
    # plt.figure()
    # plt.imshow(np.divide(lammattau15000,testmat)*1000,vmin=0.5,vmax=15)
    # plt.colorbar()
plt.figure()
plt.imshow(np.divide(lmmattau1norm,lmmattau10000norm),vmin=0.95,vmax=1.1,extent=[500,600,600,500])
plt.colorbar()
plt.gca().invert_yaxis()

plt.figure()
plt.imshow(np.divide(lmmattau1000norm,lmmattau10000norm),vmin=0.95,vmax=1.1,extent=[500,600,600,500])
plt.colorbar()
plt.gca().invert_yaxis()
#this map shows to be independent of tau. wtf
#%% Tilt it
plt.figure()
hmattau1tilted=plt.hist2d(lambdas0tau1array,lambdas1tau1array-lambdas0tau1array,bins=200,range=[[500,600],[-50,50]])
plt.colorbar()

# plt.figure()
# hmattau100=plt.hist2d(np.array(lambdas0tau100),np.array(lambdas1tau100),bins=200,range=[[500,600],[500,600]])
# plt.colorbar()

plt.figure()
hmattau10000tilted=plt.hist2d(lambdas0tau10000array,lambdas1tau10000array-lambdas0tau10000array,bins=200,range=[[500,600],[-50,50]])
plt.colorbar()
#%%
lammattau1tilted=hmattau1tilted[0]
lammattau10000tilted=hmattau10000tilted[0]
plt.figure()
plt.imshow(lammattau1tilted.T)
plt.imshow((np.divide(lammattau1tilted,lammattau10000tilted)).T/len(lambdas0tau1array)*len(lambdas0tau10000array),vmin=0.95,vmax=1.05,extent=[500,600,50,-50])
plt.colorbar()
plt.gca().invert_yaxis()
#%%
normcor=(np.divide(lammattau1tilted,lammattau10000tilted)).T/len(lambdas0tau1array)*len(lambdas0tau10000array)
plt.figure()
plt.plot(normcor[100,:])
plt.ylim(0.96,1.04)
#%% Plot spectrum in eV
plt.figure()
excitationenergy = nmtoeV(Wavelengthspec)
plt.xlim((min(excitationenergy),max(excitationenergy)))
plt.plot(nmtoeV(Wavelengthspec),ylistspec/interprefspec(Wavelengthspec)/50000)
plt.ylim((0, 1.1*5))
#plt.xlim((445,535))
plt.xlabel('Excitation Energy - Emission Energy (eV)')
plt.ylabel('Intensity')
plt.title(namelist[0])


#%% Import other measurements
# Linewidthspec=importASC(folder+'Refspec_n50mV.asc')
# Linewidthspec[0]=np.mean(Linewidthspec[0][:,509:577],1)
# Linewidthspec[0]=Linewidthspec[0]-471
# Linewidthspec[0]=Linewidthspec[0]/np.max(Linewidthspec[0])
# plt.plot(1240/Linewidthspec[1],Linewidthspec[0])
# plt.xlim((2.195,2.205))

#%%
PLspec=importASC(folder+'QD1_200Hz_200mVpp_offn40Vpp_OD1.ptu')
PLspec[0]=np.mean(PLspec[0][:,509:577],1)
PLspec[0]=PLspec[0]-471*5
#plt.figure()
plt.plot(1240/PLspec[1],PLspec[0]/np.max(PLspec[0]))
# plt.xlim((2.18,2.215))
plt.ylim((0,1.1))
# refspec=np.mean(calibspec[0][:,457:466],1)*1.2398/calibspec[1]
# plt.figure()
# plt.plot(PLspec[0])
#%% Excitation resolved decay
plt.figure()
mtimelims=np.array([0,30])
binning=1
mtimelimsdtmic=np.round(mtimelims/data[0]*1e-9)
nbins=mtimelimsdtmic[1]-mtimelimsdtmic[0]
plt.hist2d(data[2]*data[0]*1e9,calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[mtimelimsdtmic[0]*data[0]*1e9,mtimelimsdtmic[1]*data[0]*1e9],[440,510]],bins=[(nbins-1)*binning,50])#,norm=mcolors.LogNorm())
plt.xlabel('Time (ns)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()
#%% WIth wavelength binning
fig, ax = plt.subplots(constrained_layout=True)
wavelList=calibcoeffs[1]+InVoltage(data[19]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
[ylistspec,xlistspec] = np.histogram(wavelList,600) # 600 is the ratio between the laser repetition rate and the Galvomirror changing in one time interval. Example: 200Hz and 1.95 MHz results in 608 times the laser scans in one galvomirror cycle. This number is a bit wrong -> change frequency galvomirror
tlistspec = (xlistspec[:-1]+0.5*(xlistspec[1]-xlistspec[0]))
rng=range(110,520)

# ax.plot(1240/Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))
ax.plot(1240/tlistspec[rng],ylistspec[rng]/interprefspec(tlistspec[rng])/np.max(ylistspec[rng]/interprefspec(tlistspec[rng])))
ax.plot(1240/PLspec[1],PLspec[0]/np.max(PLspec[0]))
ax.plot((1240/512,1240/512),(0,0.3),'--k')
# plt.xlim((2.31,2.75))
ax.set_ylim((0,1.2))
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Intensity')
ax.set_title(namelist[0])
secax = ax.secondary_xaxis('top', functions=(nmtoeV, eVtonm))
secax.set_xlabel('Wavelength (nm)')
# plt.plot(1240/Linewidthspec[1],5000*Linewidthspec[0])
ax.legend(('PLE','PL'))

#%% Plot spec 2 against 3
fig, ax = plt.subplots(constrained_layout=True)


# ax.plot(1240/Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))
ax.plot(tlistspec2[rng2],ylistspec2[rng2]/interprefspec(tlistspec2[rng2])/np.max(ylistspec2[rng2]/interprefspec(tlistspec2[rng2])))
ax.plot(PLspec2[1],PLspec2[0]/np.max(PLspec2[0]))
ax.plot(tlistspec3[rng3],ylistspec3[rng3]/interprefspec(tlistspec3[rng3])/np.max(ylistspec3[rng3]/interprefspec(tlistspec3[rng3])))
ax.plot(PLspec3[1],PLspec3[0]/np.max(PLspec3[0]))
plt.xlim((520,640))
ax.set_ylim((0,1.2))
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity')
secax = ax.secondary_xaxis('top', functions=(eVtonm,nmtoeV))
secax.set_xlabel('Energy (eV)')
# plt.plot(1240/Linewidthspec[1],5000*Linewidthspec[0])
ax.legend(('PLE Dot2','PL Dot2','PLE Dot3','PL Dot3'))
#%% Find peaks cww
rng=range(110,520)
testdata=ylistspec[rng]/interprefspec(tlistspec[rng])
testwavel=tlistspec[rng]
result=scipy.signal.find_peaks_cwt(testdata,np.arange(5,20))
plt.plot(1240/testwavel,testdata)
for k in range(len(result)):
    plt.plot((1240/testwavel[result[k]],1240/testwavel[result[k]]),(np.min(testdata),testdata[result[k]]),'--k')
    plt.text(1240/testwavel[result[k]],testdata[result[k]]+10000,str(round(1240/testwavel[result[k]],3)),horizontalalignment='center',verticalalignment='center')
plt.xlabel('Excitation Energy (eV)')
# plt.ylim((0,1300))
plt.xlim((2.35,2.83))

#%% Make excitation wavelength resolved decay plot
plt.figure()

mtimelims=np.array([0,100])
binning=1
mtimelimsdtmic=np.round(mtimelims/data[0]*1e-9)
nbins=mtimelimsdtmic[1]-mtimelimsdtmic[0]
plt.hist2d(data[2]*data[0]*1e9,calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[mtimelimsdtmic[0]*data[0]*1e9,mtimelimsdtmic[1]*data[0]*1e9],[445,525]],bins=[(nbins-1)*binning,50])#,norm=mcolors.LogNorm())

plt.xlabel('Time (ns)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()
#%% Ex spectrum vs time
plt.figure()
plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[1,Texp],[530,590]],bins=[120,160])#,norm=mcolors.LogNorm())
plt.xlabel('Time (s)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()

#%% Excitation gate delay time
microtimesred = np.zeros(len(data[2]),dtype='int64')
macrotimesred = np.zeros(len(data[2]),dtype='int64')
nrred = 0
microtimesblue = np.zeros(len(data[2]),dtype='int64')
macrotimesblue = np.zeros(len(data[2]),dtype='int64')
macrotimescycleblue = np.zeros(len(data[2]),dtype='int64')
timesblue = np.zeros(len(data[2]),dtype='int64')
nrblue = 0

wavellimit = 590
wavellimitlow= 500


exwavel = calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
for j in tqdm(range(len(data[2]))):
    if exwavel[j] > wavellimit:
        microtimesred[nrred] = data[2][j]
        macrotimesred[nrred] = data[3][j]
        nrred += 1
    elif exwavel[j] > wavellimitlow:
        microtimesblue[nrblue] = data[2][j]
        macrotimesblue[nrblue] = data[3][j]
        macrotimescycleblue[nrblue] = data[19][j]
        timesblue[nrblue]=times0[j]
        nrblue += 1
microtimesred = microtimesred[:nrred]
macrotimesred = macrotimesred[:nrred]
microtimesblue = microtimesblue[:nrblue]
macrotimesblue = macrotimesblue[:nrblue]
macrotimescycleblue = macrotimescycleblue[:nrblue]


#%% Spectral diffusion
#binwidthspecdiff=0.5
#limitsblue = HistPhotons(timesblue*dtmicro,binwidthspecdiff,Texp)
#Meanexwavel=np.zeros(len(limitsblue)-1)
#nphotons = np.zeros(len(limitsblue)-1)
#for tbinnr in tqdm(range(len(limitsblue)-1)):
#    macrocycle=macrotimescycleblue[limitsblue[tbinnr]:limitsblue[tbinnr+1]]
#    nphotons[tbinnr]=limitsblue[tbinnr+1]-limitsblue[tbinnr]
#    Exwavels=calibcoeffs[1]+InVoltage(macrocycle*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    Meanexwavel[tbinnr]=np.mean(Exwavels)
#plt.figure()
##plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[1,Texp],[520,590]],bins=[2400,40])#,norm=mcolors.LogNorm())
##plt.xlabel('Time (s)')
##plt.ylabel('Excitation wavelength (nm)')
##plt.colorbar()
#Meanexwavel=np.nan_to_num(Meanexwavel)
#wmean=np.sum(Meanexwavel*nphotons)/np.sum(nphotons)
#plt.subplot(211)
#plt.title(namelist[0])
#plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,nphotons)
#plt.ylabel('Photons in feature')
#plt.xlabel('Time (s)')
#plt.subplot(212)
#plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,Meanexwavel)
#plt.ylim([wmean-1,wmean+1])
#plt.ylabel('Peak position')
#plt.plot([0,Texp],[wmean,wmean],'--k')
#
##%% SPectral diffusion 2
#plt.figure()
#plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[1,Texp],[520,590]],bins=[2400,40])#,norm=mcolors.LogNorm())
#plt.xlabel('Time (s)')
#plt.ylabel('Excitation wavelength (nm)')
#plt.colorbar()
#plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,Meanexwavel,'w')
#%% Plot excitation gated decay
# plt.figure()
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
fitdata=GetLifetime(microtimesblue,dtmicro,dtmacro,250e-9,tstart=-1,plotbool=True,ybg=-1,method='ML_c')
# plt.xlim([18,100])
# plt.set_yscale('log')

#%% Lifetime vs Intensity
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
#macrotimesin=data[3]
#microtimesin=data[2]
microtimesin=microtimesblue
macrotimesin=macrotimesblue
macrotimescyclein=macrotimescycleblue
binwidth=0.02
macrolimits = HistPhotons(macrotimesin*dtmacro,binwidth,Texp)
limits=macrolimits
taulist=np.zeros(len(limits)-1)
#wavelbins=150
#Exspeclist=np.zeros([wavelbins,len(limits)-1])
tauav=np.zeros(len(limits)-1)
Alist=np.zeros(len(limits)-1)
photonspbin=np.zeros(len(limits)-1)
ybglist=np.zeros(len(limits)-1)
buff=np.zeros(len(limits)-1)
meanex=np.zeros(len(limits)-1)
stdex=np.zeros(len(limits)-1)
histbinmultiplier=1
plt.figure()
lifetime=GetLifetime(microtimesin,dtmicro,dtmacro,100e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,method='ML_c')
plt.show()
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))
#plt.figure()
#test=np.zeros((wavelbins,len(limits)-1))
for tbinnr in tqdm(range(len(limits)-1)):
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]
#    Exwavels=Wavelengthex=calibcoeffs[1]+InVoltage(macrotimescycle*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    meanex[tbinnr] = np.mean(Exwavels)
    #Plot ex spectrum for each time bin
#    histrange=(wavellimitlow,wavellimit)
#    wavelbins=300
#    [ylistex,xlistex] = np.histogram(Exwavels,wavelbins,histrange)
#    tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
#    Exspeclist[:,tbinnr]=ylistex
#    plt.plot(tlistex,ylistex/max(ylistex)+tbinnr*0.1)
#    extime=macrotimescycleblue[limits[tbinnr]:limits[tbinnr+1]]
#    exwavel = calibcoeffs[1]+InVoltage(extime*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    exrange = []
#    for pcounter in range(len(extime)-1):
#        if exwavel[pcounter]>bluelim and exwavel[pcounter]<redlim:
#            exrange.append(exwavel[pcounter])
#    meanex[tbinnr]=np.mean(exrange)
#    stdex[tbinnr]=np.std(exrange)
#    print(tbinnr)


    # if len(microtimes)>30:
    #     [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,60e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=0*lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 
    # else:
    #     [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,20e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=0*lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 

    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9
    photonspbin[tbinnr]=len(microtimes)
    #using th
fig,ax1 = plt.subplots()
ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,photonspbin,'b')
#ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')
#ax1.set_xlim([0,13])
#ax1.set_ylim([0,0.2*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,tauav,'r')
#ax2.set_ylim([586,588])
ax2.set_ylim([0,40])
ax2.set_ylabel('Lifetime (ns)', color='r')
ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.set_title(namelist[0])

#%%Plot selected spectra
#plt.figure()
#reftoplot=np.sum(Exspeclist,1)
##for j in range(floor(Texp/10)-1):
##toplot=np.sum(Exspeclist[:,j*10:(j+1)*10],1)
#toplot=np.sum(Exspeclist[:,0:30],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#toplot=np.sum(Exspeclist[:,125:165],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#toplot=np.sum(Exspeclist[:,218:230],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Excitation spectrum')
#plt.legend(["0 to 30 s","125 to 165s","176 to 200s"])

#%% Plot histogram
plt.figure()
histrange=(0,np.max(photonspbin))
plt.hist(photonspbin,int(np.max(photonspbin))+1,histrange)
plt.xlabel('Photons per '+str(int(binwidth*1000))+' ms bin')
plt.ylabel('Occurence')
#simcounts=np.random.poisson(lam=285,size=8000)
#plt.hist(simcounts,int(np.max(simcounts))+1,(0,np.max(simcounts)))
#simcounts=np.random.poisson(lam=40,size=100)
#plt.hist(simcounts,int(np.max(simcounts))+1,(0,np.max(simcounts)))
#plt.ylim([1000,1600])
#%% Plot FLID map
plt.figure()
plt.hist2d(tauav,photonspbin,(50,np.max(photonspbin)/2),range=[[0,15],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()

#plt.plot(np.arange(0,50),np.arange(0,50)*505/27+8,'w')

#%% Intensity limits
limitex=180
limittrlow=40
limittrhigh=75
microtimesin=microtimesblue
macrotimesin=macrotimesblue
macrotimescyclein=macrotimescycleblue
microtimes_ex= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_ex= np.zeros(len(data[2]),dtype='int64')
nrex=0
bins_ex=0
microtimes_trion= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_trion= np.zeros(len(data[2]),dtype='int64')
nrtrion=0
bins_mid=0
microtimes_mid= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_mid= np.zeros(len(data[2]),dtype='int64')
nrmid=0
bins_off=0
microtimes_off= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_off= np.zeros(len(data[2]),dtype='int64')
nroff=0
bins_trion=0
for tbinnr in range(len(limits)-1):
    nphots = limits[tbinnr+1]-limits[tbinnr]
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescycleblue[limits[tbinnr]:limits[tbinnr+1]]
    if nphots>limitex:
        microtimes_ex[nrex:nrex+nphots]=microtimes
        macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
        bins_ex+=1
        nrex+=nphots

    elif nphots>limittrlow and nphots<limittrhigh and taulist[tbinnr]<15:# and taulist[tbinnr]>0.05: #and (photonspbin[tbinnr]-7)/taulist[tbinnr]>112/28
#    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limittrhigh and taulist[tbinnr]>0.5 and (photonspbin[tbinnr]-10)/taulist[tbinnr]>505/27:
        microtimes_trion[nrtrion:nrtrion+nphots]=microtimes
        macrotimescycle_trion[nrtrion:nrtrion+nphots]=macrotimescycle
        bins_trion+=1
        nrtrion+=nphots
    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limitex:# and tauav[tbinnr]>6 :
        microtimes_mid[nrmid:nrmid+nphots]=microtimes
        macrotimescycle_mid[nrmid:nrmid+nphots]=macrotimescycle
        bins_mid+=1
        nrmid+=nphots
    elif nphots<limittrlow and taulist[tbinnr]<3:
        microtimes_off[nroff:nroff+nphots]=microtimes
        macrotimescycle_off[nroff:nroff+nphots]=macrotimescycle
        bins_off+=1
        nroff+=nphots
    
microtimes_ex = microtimes_ex[:nrex]
macrotimescycle_ex = macrotimescycle_ex[:nrex]
microtimes_trion = microtimes_trion[:nrtrion]
macrotimescycle_trion = macrotimescycle_trion[:nrtrion]
microtimes_mid = microtimes_mid[:nrmid]
macrotimescycle_mid = macrotimescycle_mid[:nrmid]
microtimes_off = microtimes_off[:nroff]
macrotimescycle_off = macrotimescycle_off[:nroff]
#% Lifetime of Exciton and Trion
plt.figure()
fitex=GetLifetime(microtimes_ex,dtmicro,dtmacro,100e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_ex/(Texp/binwidth),method='ML_c')
fittrion=GetLifetime(microtimes_mid,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')
fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')

# fitoff=GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=0*lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
#fitmid=GetLifetime(microtimes_mid,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')

print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))
#plt.xlim([0,220])
plt.legend(['High cps','High cps fit','Mid cps','Mid cps fit','Low cps','Low cps fit'])

#%% Proper wavelength binning
macrotimescycle_exnew=macrotimescycle_ex[0:len(macrotimescycle_trion)-1]
Wavelengthex=calibcoeffs[1]+InVoltage(macrotimescycle_ex*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthtrion=calibcoeffs[1]+InVoltage(macrotimescycle_trion*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
Wavelengthoff=calibcoeffs[1]+InVoltage(macrotimescycle_off*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthmid=calibcoeffs[1]+InVoltage(macrotimescycle_mid*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
histrange=(np.min(Wavelengthex),np.max(Wavelengthex))
wavelbins=200
[ylistex,xlistex] = np.histogram(Wavelengthex,wavelbins,histrange)
tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
[ylisttrion,xlisttrion] = np.histogram(Wavelengthtrion,wavelbins,histrange)
tlisttrion = (xlisttrion[:-1]+0.5*(xlisttrion[1]-xlisttrion[0]))
[ylistoff,xlistoff] = np.histogram(Wavelengthoff,wavelbins,histrange)
tlistoff = (xlistoff[:-1]+0.5*(xlistoff[1]-xlistoff[0]))
[ylistmid,xlistmid] = np.histogram(Wavelengthmid,wavelbins)
tlistmid = (xlistmid[:-1]+0.5*(xlistmid[1]-xlistmid[0]))
plt.figure()
plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex))/interprefspec(tlistex))
plt.plot(tlistmid,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))/interprefspec(tlistmid)-5000)
plt.plot(tlisttrion,ylisttrion/interprefspec(tlisttrion)-10000)
# plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
# dotsmid,=plt.plot(tlistmid,ylistmid*(np.sum(ylistmid)/np.sum(ylistmid)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),2),color=dotsex.get_color())

#dotsoff,=plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistmid,ylistmid/interprefspec(tlistmid),'.')
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid/interprefspec(tlistmid),5))#,color=dotsex.get_color())

#dotstrion,=plt.plot(tlisttrion,ylisttrion,'.')#/max(ylistspec))
#plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion*1.,2),color=dotstrion.get_color())

#plt.plot(tlistex,ylisttrion-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,4)-scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),4))
#plt.plot(tlistex,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistex,np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
#plt.plot(tlisttrion,np.cumsum(ylisttrion))#/max(ylistspec))
#plt.plot(tlistex,np.cumsum(ylisttrion)-np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
plt.xlim([445,wavellimit])
plt.ylim((-10000,50000))
#plt.ylim([0,20])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('(normalized) Counts')
plt.legend(['High','Medium','Low'])

#%% Improper wavel binning
histrange=(np.min(data[19])*dtmacro,np.max(data[19])*dtmacro)
wavelbins=150
[ylistex,xlistex] = np.histogram(macrotimescycle_ex*dtmacro,wavelbins,histrange)
tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
[ylisttrion,xlisttrion] = np.histogram(macrotimescycle_trion*dtmacro,wavelbins,histrange)
tlisttrion = (xlisttrion[:-1]+0.5*(xlisttrion[1]-xlisttrion[0]))
[ylistoff,xlistoff] = np.histogram(macrotimescycle_off*dtmacro,wavelbins,histrange)
tlistoff = (xlistoff[:-1]+0.5*(xlistoff[1]-xlistoff[0]))


Wavelengthex=calibcoeffs[1]+InVoltage(tlistex,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthtrion=calibcoeffs[1]+InVoltage(tlisttrion,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
Wavelengthoff=calibcoeffs[1]+InVoltage(tlistoff,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]


plt.figure()
plt.plot(Wavelengthex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistmid,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid)))
plt.plot(Wavelengthtrion,ylisttrion,'.')
plt.plot(Wavelengthoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#dotsex,=plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#dotsmid,=plt.plot(tlistmid,ylistmid*(np.sum(ylistmid)/np.sum(ylistmid)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),2),color=dotsex.get_color())

#dotsoff,=plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistmid,ylistmid/interprefspec(tlistmid),'.')
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid/interprefspec(tlistmid),5))#,color=dotsex.get_color())

#dotstrion,=plt.plot(tlisttrion,ylisttrion,'.')#/max(ylistspec))
#plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion*1.,2),color=dotstrion.get_color())

#plt.plot(tlistex,ylisttrion-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,4)-scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),4))
#plt.plot(tlistex,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistex,np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
#plt.plot(tlisttrion,np.cumsum(ylisttrion))#/max(ylistspec))
#plt.plot(tlistex,np.cumsum(ylisttrion)-np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
plt.xlim([wavellimitlow,wavellimit])
#plt.ylim([0,20])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('(normalized) Counts')
plt.legend(['Ex','Trion','Off'])
#%% Plot normalized spectra
plt.figure()
plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex))/interprefspec(tlistex))
plt.plot(tlisttrion,ylisttrion/interprefspec(tlisttrion))
#plt.ylim([0,1400])
plt.legend(['Bright','Gray'])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('normalized Counts')
plt.title(namelist[0])
#%% Plot difference spectrum
plt.figure()
plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,5)-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid)),5)-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))

#%%Lifetime sorting

microtimesfast = np.zeros(len(data[2]),dtype='int64')
macrotimesfast = np.zeros(len(data[2]),dtype='int64')
macrotimescyclefast = np.zeros(len(data[2]),dtype='int64')
nrfast = 0
microtimesslow = np.zeros(len(data[2]),dtype='int64')
macrotimesslow = np.zeros(len(data[2]),dtype='int64')
macrotimescycleslow = np.zeros(len(data[2]),dtype='int64')
timesslow = np.zeros(len(data[2]),dtype='int64')
nrslow = 0
wavellimit = 590
startmicro = lifetime[3]
delaylimit = lifetime[3]+round(3/dtmicro*1e-9)
delayend = round(40/dtmicro*1e-9)
delayend = lifetime[3]+round(10/dtmicro*1e-9)

for j in tqdm(range(len(data[2]))):
    if data[2][j] > startmicro and data[2][j] < delaylimit:
        microtimesfast[nrfast] = data[2][j]
        macrotimesfast[nrfast] = data[3][j]
        macrotimescyclefast[nrfast] = data[19][j]
        nrfast += 1
    elif data[2][j] > delaylimit and data[2][j] < delayend:
        microtimesslow[nrslow] = data[2][j]
        macrotimesslow[nrslow] = data[3][j]
        macrotimescycleslow[nrslow] = data[19][j]
        timesslow[nrslow]=times0[j]
        nrslow += 1
microtimesfast = microtimesfast[:nrfast]
macrotimesfast = macrotimesfast[:nrfast]
macrotimescyclefast = macrotimescyclefast[:nrfast]
microtimesslow = microtimesslow[:nrslow]
macrotimesslow = macrotimesslow[:nrslow]
macrotimescycleslow = macrotimescycleslow[:nrslow]
wavelbins=300
[ylistspecfast,xlistspecfast] = np.histogram(macrotimescyclefast,wavelbins)
tlistspecfast = (xlistspecfast[:-1]+0.5*(xlistspecfast[1]-xlistspecfast[0]))*dtmacro
Wavelengthspecfast=calibcoeffs[1]+InVoltage(tlistspecfast,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
[ylistspecslow,xlistspecslow] = np.histogram(macrotimescycleslow,wavelbins)
tlistspecslow = (xlistspecslow[:-1]+0.5*(xlistspecslow[1]-xlistspecslow[0]))*dtmacro
Wavelengthspecslow=calibcoeffs[1]+InVoltage(tlistspecslow,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
plt.figure()
plt.plot(Wavelengthspecfast,ylistspecfast)#/np.sum(ylistspecfast))
plt.plot(Wavelengthspecslow,ylistspecslow)#/np.sum(ylistspecslow))#+0.2*np.max(ylistspecslow)/np.sum(ylistspecslow))
plt.legend(['Fast','Slow'])
#%%

plt.plot(Wavelengthspecfast,ylistspecfast/1.55)#/np.sum(ylistspecfast))
plt.plot(Wavelengthspecslow,ylistspecslow)#/np.sum(ylistspecslow))#+0.2*np.max(ylistspecslow)/np.sum(ylistspecslow))
plt.legend(['Fast','Slow'])
#%% Intensity slices
aggregatedphot=0
plt.figure()
ylistbinspec=np.zeros([300,20])
Wavelengthbinspec=np.zeros([300,20])
[ylistspecav,xlistspecav] = np.histogram(data[19],300)
for binnr in range(nrbins): # for each intensity bin
    # select photons in bins with intensity within Ilims(binnr) and Ilims(binnr+1)
    if binnr>-1:
        histbinmultiplier=16
        [binmicrotimes0,bintimes0,binmacrotimescycle0,binmicrotimes1,bintimes1,binmacrotimescycle1,onbincount] = SliceHistogram(microtimes0,times0,limits0,data[19],microtimes1,times1,limits1,data[20],dtmicro,dtmacro,Ilims[binnr],Ilims[binnr+1])  
#        [ylist,xlist] = np.histogram(binmicrotimes0,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])    
#        tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
        [ylistbinspec[:,binnr],xlistbinspec] = np.histogram(binmacrotimescycle0,300)
        tlistbinspec = (xlistbinspec[:-1]+0.5*(xlistbinspec[1]-xlistbinspec[0]))*dtmacro
        [ylistbinspec[:,binnr],xlistbinspec] = np.histogram(binmacrotimescycle0,300)
        tlistbinspec = (xlistbinspec[:-1]+0.5*(xlistbinspec[1]-xlistbinspec[0]))*dtmacro
        Wavelengthbinspec[:,binnr]=calibcoeffs[1]+InVoltage(tlistbinspec,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
        plt.plot(Wavelengthbinspec[:,binnr],ylistbinspec[:,binnr]/np.sum(ylistbinspec[:,binnr],0)+1*0.0015*binnr)
        # plt.plot(Wavelengthbinspec[:,binnr],ylistbinspec[:,binnr]/np.sum(ylistbinspec[:,binnr])-ylistspecav/np.sum(ylistspecav)+0.005*binnr)
        plt.xlim([520,592])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photons per bin')
        photinbin=np.sum(ylistbinspec[:,binnr])
        aggregatedphot+=photinbin
#        istart=0
#        iend=max(tlist)
#        nrnewbins=int(dtmacro/(dtmicro*histbinmultiplier))
#        bgcounts=binwidth*onbincount*80
#        bgcpb=bgcounts/nrnewbins
#        initParams = [np.max(ylist), 25]
##        [Abinlist[measnr,binnr],Alphabinlist[measnr,binnr],Thetabinlist[measnr,binnr]]=MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,initParams,plotbool=True)
#        GetLifetime(binmicrotimes0,dtmicro,dtmacro,20e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=1+0*lifetime[2]*binwidth/Texp,plotbool=True,method='ML_c') 
#    # calculate average lifetimes
##        plt.figure(15)
#        plt.title(['binnr:', str(binnr),'Alpha: ', str(round(Alphabinlist[measnr,binnr],2)),'Theta: ', str(round(Thetabinlist[measnr,binnr]*180/np.pi))])
    
    
#        [tauavelist[measnr,binnr],Alist[measnr,binnr],_] = GetLifetime(np.append(binmicrotimes0,binmicrotimes1),dtmicro,dtmacro,dttau,-1,histbinmultiplier,ybg*onbincount)
        print('onbincount:',onbincount)
        print('Overall photons:', photinbin)






#%% Other plots and legacy
#%%
totalPhots=np.zeros(100)
correlation=np.zeros(100)
correlationtrion=np.zeros(100)
timepassed=np.zeros(100)
timepassed_trion=np.zeros(100)
for j in range(0,100):
    macrotimescycle_exnew=macrotimescycle_ex[0:(100*(j+1)-1)]
    macrotimescycle_trionnew=macrotimescycle_trion[0:(10*(j+1)-1)]
    timepassed[j]=macrotimescycle_exnew[-1]-macrotimescycle_exnew[0]
    timepassed_trion[j]=macrotimescycle_trionnew[-1]-macrotimescycle_trionnew[0]
    totalPhots[j]=50*(j+1)
    Wavelengthmid=calibcoeffs[1]+InVoltage(macrotimescycle_exnew*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
    Wavelengthtrion=calibcoeffs[1]+InVoltage(macrotimescycle_trionnew*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
    [ylistmid,xlistmid] = np.histogram(Wavelengthmid,wavelbins)
    [ylisttrion,xlisttrion] = np.histogram(Wavelengthtrion,wavelbins)
    a=np.corrcoef(ylistex,ylistmid)
    b=np.corrcoef(ylistex,ylisttrion)
    correlation[j]=a[0,1]
    correlationtrion[j]=b[0,1]
plt.plot(totalPhots,np.cumsum(timepassed)*dtmacro)
#plt.plot(timepassed_trion,correlationtrion)
#plt.figure()
#plt.plot(totalPhots,timepassed)
#plt.plot(totalPhots,timepassed_trion)
#%%G2 eval
test=MakeG2(times0,times1,dtmicro,g2restime=dtmicro,nrbins=8000)
#%%
g2axis=test[0]
g2vals=test[1]
centerindex=np.argmin(np.abs(g2axis))
avwindow=round(dtmacro/dtmicro/4)
npeaks=floor(abs(g2axis[0])*2*1e-9/dtmacro)
peakarea=np.zeros(npeaks,dtype=float)
peakcenter=np.zeros(npeaks,dtype=float)
for j in range(npeaks):
    peakcenter[j]=np.round(centerindex+(j-(npeaks-1)/2)*dtmacro/dtmicro)
    peakarea[j]=np.sum(g2vals[int(np.round(peakcenter[j]-avwindow/2)):int(np.round(peakcenter[j]+avwindow/2))])
#    peakarea[j]=np.sum(g2vals)
plt.plot(g2axis[peakcenter.astype(int)],peakarea/np.mean(peakarea),'*')
plt.ylim([0,1.2*np.max(peakarea/np.mean(peakarea))])


#%% Filtering
#plt.plot(ylistex)
#Subtratc baseline
baseline=(ylisttrion[-1]-ylisttrion[0])/(tlisttrion[-1]-tlisttrion[0])*(tlisttrion-tlisttrion[0])+ylisttrion[0]
testspec=ylisttrion-baseline
#plt.plot(tlisttrion,ylisttrion,tlisttrion,baseline)
freq=np.fft.rfftfreq(tlisttrion.shape[-1])
#testfourier=np.fft.fft(testspec)
fourierreal=np.fft.rfft(testspec)
testnobasel=np.fft.rfft(ylisttrion)
Intfourierreal=np.square(np.absolute(fourierreal))
Fourieramp=np.absolute(fourierreal)
variance=np.var(fourierreal[round(len(fourierreal)/2):-1])
Noise=2*variance/(wavelbins/2)
plt.semilogy(Intfourierreal/Noise)
#plt.legend(['baseline subtracted','Raw'])
#plt.xlim([])
