# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:57:02 2020

@author: rober
"""

import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
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
import matplotlib as mpl
#mpl.rcParams['figure.dpi']= 300
import socket #enables to find computer name to make less of a mess with folders
# %matplotlib auto

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


def ImportT3(filename,HHsettings):
    with open(filename, "rb+") as f:
    
        
        ##This section is needed to .ptu files and should be commented for .out
#        while True:
#            if f.read(1) == b'M':
#                if f.read(18) == b'easDesc_Resolution': # recognize time unit entry
#                    break
#        f.read(21) # rest of the tag
#        dtmicro = struct.unpack('d',f.read(8))[0]
#        #print('Microtime unit:', dtmicro)
#    
#        while True:
#            if f.read(1) == b'M':
#                if f.read(24) == b'easDesc_GlobalResolution': # recognize time unit entry
#                    break
#        f.read(15) # rest of the tag
#        dtmacro = struct.unpack('d',f.read(8))[0]
#        #print('Macrotime unit:', dtmacro)
#    
#        while True:
#            if f.read(1) == b'T':
#                if f.read(23) == b'TResult_NumberOfRecords': # recognize number of records entry
#                    break
#        f.read(16) # rest of the tag
#        nrrec = struct.unpack('q',f.read(8))[0] # extract number of records
#        #print('Number of records in file:', nrrec)
#
#        while True:
#            if f.read(1) == b'H':
#                if f.read(9) == b'eader_End':
#                    #print('Header_End found')
#                    break
#        f.read(38) # rest of Header_End
        
#        
        nrrec=HHsettings["overallCounts"]
        dtmicro=HHsettings["resolution"]*1e-12    #in s
        dtmacro=1/HHsettings["syncRate"]    #in s
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
        macrotimesfireE = np.zeros(nrrec,dtype='int64');
        microtimesfireE = np.zeros(nrrec,dtype='int64');
        macrotimecycle0 = np.zeros(nrrec,dtype='int64'); #Time since start of current galvo cycle of photons on detector 0 in integer units (macrotime)
        macrotimeemissioncycle0 = np.zeros(nrrec,dtype='int64'); #Time since start of current emission cycle of photons on detector 0 in integer units (macrotime)
        macrotimecycle1 = np.zeros(nrrec,dtype='int64');
        cyclenumber0 = np.zeros(nrrec,dtype='int64'); #Number of galvo cycle of of photons on detector 0
        emissioncyclenumber0 = np.zeros(nrrec,dtype='int64'); #Number of emission cycle of of photons on detector 0
        firstphotonincycle0 = np.zeros(nrrec,dtype='int64'); #Index of the first photon on a given galvo cycle
        firstphotoninemissioncycle0 = np.zeros(nrrec,dtype='int64'); #Index of the first photon on a given emission cycle
        overflows = 0
        nrphotons0 = 0
        nrphotons1 = 0
        nrfireA = 0
        nrfireB = 0
        nrfireC = 0
        nrfireD = 0
        nrfireE = 0
        prevchann = 0
        lastcyclestarttime = 0
        lastemissioncyclestarttime = 0
        currentcycle = 0
        currentemissioncycle = 0
        
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
                macrotimeemissioncycle0[nrphotons0] = macrotime-lastemissioncyclestarttime + 1024*overflows
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
            elif channel == 72:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireE[nrfireE] = macrotime + 1024*overflows
                emissioncyclenumber0[nrphotons0] = currentemissioncycle
                currentemissioncycle += 1            
                firstphotoninemissioncycle0[currentemissioncycle] = nrphotons0 + 1
                lastemissioncyclestarttime = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireE[nrfireE] = microtime
                nrfireE += 1                              
            else:
                print('bad channel:',channel)
                
    microtimes0 = microtimes0[:nrphotons0]
    macrotimes0 = macrotimes0[:nrphotons0]
    cyclenumber0 = cyclenumber0[:nrphotons0]
    emissioncyclenumber0 = emissioncyclenumber0[:nrphotons0]
    firstphotonincycle0 = firstphotonincycle0[:currentcycle]
    firstphotoninemissioncycle0 = firstphotoninemissioncycle0[:currentemissioncycle]
    macrotimecycle0 = macrotimecycle0[:nrphotons0]
    macrotimeemissioncycle0 = macrotimeemissioncycle0[:nrphotons0]
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
    microtimesfireE = microtimesfireE[:nrfireE]
    macrotimesfireE = macrotimesfireE[:nrfireE]
    
    print('nrphotons0:',nrphotons0)
    print('nrphotons1:',nrphotons1)
    print('nrfireA:',nrfireA)
    print('nrfireB:',nrfireB)
    print('overflows:',overflows)
    # print('firstphotonincycle0',firstphotonincycle0)
    # print('firstphotoninemissioncycle0',firstphotoninemissioncycle0)
    

    return [dtmicro, dtmacro, microtimes0, macrotimes0, microtimes1, macrotimes1, nrphotons0,nrphotons1,overflows,microtimesfireA,macrotimesfireA,nrfireA,microtimesfireB,macrotimesfireB,nrfireB,macrotimesfireC,nrfireC,macrotimesfireD,nrfireD,macrotimecycle0,macrotimecycle1,cyclenumber0,firstphotonincycle0, macrotimeemissioncycle0, emissioncyclenumber0, firstphotoninemissioncycle0, macrotimesfireE, microtimesfireE]
#           0      , 1      , 2          , 3          , 4          , 5          , 6         , 7        , 8       , 9             , 10             , 11   , 12            , 13            , 14    , 15            , 16    , 17            , 18    , 19            , 20            , 21          , 22               ,  23                         24                 ,25

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

def InVoltagenew(t,Freq,VPP,VOffset,tshift):
    Period=1/Freq
    t=t+tshift
    Bool1=t<=Period/4
    Bool2=scipy.logical_and(t>Period/4,t<=3*Period/4)
    Bool3=t>3*Period/4
    InVoltage=(VPP/2*t/(Period/4)+VOffset)*Bool1+(VPP/2-VPP/2*(t-Period/4)/(Period/4)+VOffset)*Bool2+(-VPP/2+VPP/2*(t-3*Period/4)/(Period/4)+VOffset)*Bool3
    return(InVoltage)

def importASC(filename,nr_px=1024):
    """Import ASC files exported from Andor Solis.
    
    Parameters
    ----------
    filename: String with the complete path to the file.

    Returns
    -------
    raw_data : Video: 2D numpy array containing the counts [yPixels,xPixels], Video: 3D numpy array containing the counts [yPixels,xPixels,Frames]
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
    
# reffolder+'HR63_10MHz_200Hz_240mVpp_off0mV_OD1_cryo_camera.asc'
    
    info = {}
    #check length of series
    if a[0].str.contains('Number in Series').any()==True: # so in some datas this thing is called .. Series and otherwise ..Kinetic series
        buff = a.iloc[:,0].str.find('Number in Series',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['noFrames']=1
        elif pd.Series.max(buff)>=0:       #Cropped
            info['noFrames']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            #Data structure is: ["left","right",bottom","top"]
    elif a[0].str.contains('Number in Kinetics Series').any()==True:
        buff = a.iloc[:,0].str.find('Number in Kinetics Series',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['noFrames']=1
        elif pd.Series.max(buff)>=0:       #Cropped
            info['noFrames']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            #Data structure is: ["left","right",bottom","top"]
    else:
        info['noFrames']=1
   
    buff=a.iloc[:,0].str.find('Exposure Time',0)   #Gives number of column where it finds the string and -1 if not found
    info['expTime']=float(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    buff=a.iloc[:,0].str.find('Gain level',0)   #Gives number of column where it finds the string and -1 if not found
    info['Gain']=int(a.iloc[buff[buff>-1].index[0],0].split(":")[1])   
    
    buff=a.iloc[:,0].str.find('Readout Mode',0)
    info['ReadMode']=(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    if a.iloc[buff[buff==0].index[0],0].find('Image')>-1:
        # buff=a.iloc[:,0].str.find('Horizontal Binning',0)   #Gives number of column where it finds the string and -1 if not found
        # info['Horbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
        
        # buff=a.iloc[:,0].str.find('Vertical Binning',0)   #Gives number of column where it finds the string and -1 if not found
        # info['Vertbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
        
        #Find cropping
        buff = a.iloc[:,0].str.find('left',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['cropped']=False
            info['nrP_x'] = 1024
            info['nrP_y'] = 1024
        elif pd.Series.max(buff)>=0:       #Cropped
            info['cropped']=True
            info['Imlimits']=[int(a.iloc[buff[buff>=0].index[0],3].split(":")[1]),int(a.iloc[buff[buff>=0].index[0],4]),int(a.iloc[buff[buff>=0].index[0],5]),int(a.iloc[buff[buff>=0].index[0],6])]
            info['nrP_x'] = info['Imlimits'][1]-info['Imlimits'][0]+1
            info['nrP_y'] = info['Imlimits'][3]-info['Imlimits'][2]+1
        #Data structure is: ["left","right",bottom","top"]

   
        buff=a.iloc[:,0].str.find('Vertical Binning',0)
        if pd.Series.max(buff)<0:
            info['vertBin']=1
        else:
            info['vertBin']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            info['nrP_y']=int(info['nrP_y']/info['vertBin'])
            
        buff=a.iloc[:,0].str.find('Horizontal Binning',0)
        if pd.Series.max(buff)<0:
            info['horzBin']=1
        else:
            info['horzBin']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            info['nrP_x']=int(info['nrP_x']/info['horzBin'])
    elif a.iloc[buff[buff==0].index[0],0].find('Multi-Track')>-1:
        info['nrP_x']=1024
        info['nrP_y']=1
    
    x = a.iloc[0:(info['nrP_x']),0].str.strip().values.astype(float)
    y = np.arange(1,info['nrP_y']+1)
    
    raw_data = np.transpose(np.reshape(np.transpose(a.iloc[0:info['nrP_x']*info['noFrames'],1:-1].values.astype(int)),(info['nrP_y'],info['noFrames'],info['nrP_x'])),(0,2,1))
    if info['noFrames']==1:
        raw_data=raw_data[:,:,0]
    #Has dimensions ("yaxis","xaxis","numberofFrame")
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
                                
    g2tlist = np.arange(g2res*dtmicro*nrbins,g2restime)*1e9
    plt.plot(g2tlist,g2)
    #plt.plot(g2tlist,g2B)
    #plt.plot(g2tlist,g2C)
    plt.title('g(2) correlation')
    plt.xlabel('delay (ns)')
    plt.ylabel('occurence (a.u.)')
    plt.ylim([0,max(g2)])
    plt.show()

    return(g2tlist,g2,g2restime,nrbins)

def Autocor1APD(times0,nrbins=200):
    i0=0
    i1=0
    lim1=0
    g2 = np.zeros(2*nrbins)
    #g2B = np.zeros(2*nrbins)
    #g2C = np.zeros(2*nrbins)
    #blindB = 2e-9
    #blindC = 5e-9

    g2res = 1
    #blindB = blindB/tmicro
    #blindC = blindC/tmicro

    # correlate det0 with det1 (positive time differences)
    for i0 in range(len(times0)):
        t0 = times0[i0]
        i1 = 0
        q = 0
        while q == 0: 
            if lim1 + i1 < len(times0): # check if we've already reached end of photon stream on det1
                dt = times0[lim1+i1]-t0 
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

    
                                
    g2tlist = np.arange(-g2res*dtmicro*(nrbins-0.5),g2res*dtmicro*nrbins,nrbins*g2res)*1e9
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

def Findtshift(Freq,Vpp,Voffset,calibcoeffs,macrocyclelist,dtmacro,matchrange=(500,570),shiftrange=(-6e-4,-2e-4),steps=30,Debugmode=False):
    InVoltagenew_c=nb.jit(nopython=True)(InVoltagenew) #compile to C to speed it up
    threshlow=1/Freq/4
    threshhigh=3/Freq/4
    #Sort microtimes in two halves
    Z = np.logical_and(threshlow<(macrocyclelist*dtmacro),(macrocyclelist*dtmacro)<= threshhigh)
    tforward=macrocyclelist[np.where(Z)]
    tbackward=macrocyclelist[np.where(np.logical_not(Z))]
    histbinnumber = 100 # 608 was for the entire range. For a matchrange of 520 to 590, this should be 4 times as small than the original to prevent aliasing
    #First coarse sweep
    # matchrange=(500, 570) #Wavelengthrange in which it should match. Maybe exclude the boundaries a bit
    tshift=np.zeros(steps)
    autocorr=np.zeros(steps)
    for k in tqdm(range(0,steps)):
        tshift[k]=shiftrange[0]+(shiftrange[1]-shiftrange[0])*k/steps
        lamforward=calibcoeffs[1]+InVoltagenew_c(tforward*dtmacro,Freq,Vpp,Voffset,tshift[k])*calibcoeffs[0]
        lambackward=calibcoeffs[1]+InVoltagenew_c(tbackward*dtmacro,Freq,Vpp,Voffset,tshift[k])*calibcoeffs[0]
        [ylistforward,xlistforward] = np.histogram(lamforward,histbinnumber,range=matchrange)
        # tlistforward = (xlistforward[:-1]+0.5*(xlistforward[1]-xlistforward[0]))
        [ylistbackward,xlistbackward] = np.histogram(lambackward,histbinnumber,range=matchrange)
        # tlistbackward = (xlistbackward[:-1]+0.5*(xlistbackward[1]-xlistbackward[0]))
        autocorr[k]=np.sum(ylistforward*ylistbackward)
    if Debugmode==True:
        plt.figure()
        plt.plot(tshift,autocorr,'.')
    optimumshift=tshift[np.argmax(autocorr)]
    if Debugmode==True:
        tshifttest=optimumshift
        lamforward=calibcoeffs[1]+InVoltagenew(tforward*dtmacro,Freq,Vpp,Voffset,tshifttest)*calibcoeffs[0]
        lambackward=calibcoeffs[1]+InVoltagenew(tbackward*dtmacro,Freq,Vpp,Voffset,tshifttest)*calibcoeffs[0]
        [ylistforward,xlistforward] = np.histogram(lamforward,50,range=matchrange)
        tlistforward = (xlistforward[:-1]+0.5*(xlistforward[1]-xlistforward[0]))
        [ylistbackward,xlistbackward] = np.histogram(lambackward,50,range=matchrange)
        tlistbackward = (xlistbackward[:-1]+0.5*(xlistbackward[1]-xlistbackward[0]))
        plt.figure()
        plt.plot(tlistforward,ylistforward)
        plt.plot(tlistbackward,ylistbackward)
    return optimumshift

def Easyhist(rawdata,lowestbin,highestbin,stepsize):
    edges=np.linspace(lowestbin-stepsize/2,highestbin+stepsize/2,int((highestbin-lowestbin+stepsize)/stepsize)+1)
    wavelbins=np.linspace(lowestbin,highestbin,int((highestbin-lowestbin)/stepsize)+1)
    histdata=np.histogram(rawdata,bins=edges)
    return wavelbins,histdata[0],edges
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)