
# coding: utf-8

# In[7]:


import sqlite3
import os
from skimage.feature import register_translation
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np


# In[8]:


# import LAM library for logbook, data analysis...
from pfs.sacFileHandling import *


# In[9]:


imgPath = '/home/pfs/shared/Pictures/SM1'


# In[10]:


smId = 1


# In[11]:


def correlate(reference, image):
    return register_translation(reference, image, 100)

    
def sacShift(filelist, duplicate):
    __, reference = stackedImage(filelist, 0, duplicate=duplicate)
    res = []
    
    for i in range(len(filelist)//duplicate):
        hdr, image = stackedImage(filelist, i, duplicate=duplicate)
        ccdPos = hdr['ait.sac.ccd.position']
        pentaPos= hdr['ait.sac.penta.position']

        shift, error, diffphase = correlate(reference=reference, image=image)
        res.append((ccdPos,pentaPos,shift[0], shift[1]))
        
    return pd.DataFrame(data=res, columns=['ccdPosition', 'pentaPosition', 'shift_x', 'shift_y'])


# ## Data required : 
# To be able to find the right focus, some data acquisition is required.
# 
# you need to use ics_spsaitActor to perform your sequence
# 
# example : <b>sac align exptime=1.0 focus=0 lowBound=-450 upBound=450 nbPosition=10 duplicate=3</b>
# 
# -450 : 450 is the range of the pentaprism linear stage
# 
# focus is the position of the ccd linear stage
# 
# the goal here is to take several sequences for difference focus value, here we have taken data for [0,2,4,6,8]
# 

# ## Input Parameters : 
# The only parameters needed is the experimentIds that match your data acquisition sequence

# In[12]:


experimentStart = 27 #the first experimentId is 12
experimentEnd = 35    #the last experimentId is 16


# In[13]:


data = []

plt.figure(figsize=(12,8))
plt.xlabel('Pentaprism Position (mm)')
plt.ylabel('offset (pixels)')

for experimentId in range(experimentStart, experimentEnd+1):
    visitStart, visitEnd = Logbook.visitRange(experimentId=experimentId)
    focus = float(Logbook.getParameter(experimentId=experimentId, param='focus'))
    duplicate = int(Logbook.getParameter(experimentId=experimentId, param='duplicate'))
    
    filelist = constructFilelist(visitStart=visitStart, visitEnd=visitEnd)

    df = sacShift(filelist=filelist, duplicate=duplicate)
    plt.plot(df['pentaPosition'], df['shift_x'], 'o-', label='ccdPosition : %.2f'%df['ccdPosition'][0])
    
    [slope,off] = np.polyfit(df['pentaPosition'], df['shift_x'], deg=1)
    data.append((focus, slope))
    
plt.title('Spot displacement vs the pentaprism position')
plt.grid()
plt.legend()

doSave = False

if doSave:
    plt.savefig(os.path.join(imgPath, 'SM1_SACALIGN_EXP%i-%i_SPOT_DISPLACEMENT.png'%(experimentStart, experimentEnd)))


# In[8]:


df = pd.DataFrame(data=data, columns=['focus', 'slope'])


# In[9]:


x = np.arange(np.min(df['focus']), np.max(df['focus'])+0.01, 0.01)
popt = np.polyfit(df['focus'], df['slope'], deg=1)


plt.figure(figsize=(12,8))
plt.xlabel('Ccd Position (mm)')
plt.ylabel('Slope')

plt.plot(df['focus'], df['slope'], 'o')
plt.plot(x, np.polyval(popt, x), '-')

calc_focus = -popt[1]/popt[0]
plt.vlines(x=calc_focus, ymin=np.min(df['slope']), ymax=np.max(df['slope']))
plt.title('Calculated Focus = %.3f mm'%calc_focus)
plt.grid()

if doSave:
    plt.savefig(os.path.join(imgPath, 'SM1_SACALIGN_EXP%i-%i_CALC_FOCUS.png'%(experimentStart, experimentEnd)))

