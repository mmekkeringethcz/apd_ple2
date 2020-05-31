# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:29:05 2020

@author: rober
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

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
    

    
    info = {}
    buff=a.iloc[:,0].str.find('Horizontal Binning',0)   #Gives number of column where it finds the string and -1 if not found
    info['Horbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    
    buff=a.iloc[:,0].str.find('Vertical Binning',0)   #Gives number of column where it finds the string and -1 if not found
    info['Vertbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    
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
    

    #check length of series
    buff = a.iloc[:,0].str.find('Number in Series',0)    #This one does not appear when using the whole image
    if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
        info['noFrames']=1
    elif pd.Series.max(buff)>=0:       #Cropped
        info['noFrames']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
        #Data structure is: ["left","right",bottom","top"]

    
    info['expTime'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[nr_px+7:nr_px+8,0].str.strip().values[0]).group(0))
    # info['expTime'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1031:1032,0].str.strip().values[0]).group(0))
    #info['EMgain'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1046:1047,0].str.strip().values[0]).group(0))
    #info['centerWL'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1055:1056,0].str.strip().values[0]).group(0))
    #info['gratingDensity'] = float(re.search(r"[-+]?\d*\.\d+|\d+", a.iloc[1056:1057,0].str.strip().values[0]).group(0))
    
    x = a.iloc[0:(info['nrP_x']),0].str.strip().values.astype(float)
    y = np.arange(1,info['nrP_y']+1)
    
    raw_data = np.transpose(np.reshape(np.transpose(a.iloc[0:info['nrP_x']*info['noFrames'],1:-1].values.astype(int)),(info['nrP_y'],info['noFrames'],info['nrP_x'])),(0,2,1))
    if info['noFrames']==1:
        raw_data=raw_data[:,:,0]
    #Has dimensions ("yaxis","xaxis","numberofFrame")
    return [raw_data, x, y, info]


#%%

#%% Test function

folder='C:/Users/rober/Documents/Doktorat/Projects/AniketClusters/20200311_D164/'
data=importASC(folder+'Spec_405BP_405DC_442LP_freshsamplewithzeonex_2daysoldsample-1.asc')


#%%

from skimage.feature import blob_dog, blob_log, blob_doh
#%%
video=(data[0]-590)*5.36/750
meanim=np.mean(video,2)
# plt.imshow(meanim)

fig,ax=plt.subplots()
ax.imshow(meanim)

blobs_log = blob_log(meanim, max_sigma=2, threshold=0.1)
for j in range(blobs_log.shape[0]):
    y, x, r = blobs_log[j]
    c = plt.Circle((x, y), r, color='white', linewidth=0.5, fill=False)
    ax.add_patch(c)
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
print(str(blobs_log.shape[0])," Particles found")