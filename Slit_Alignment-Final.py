
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


# import LAM library for logbook, data analysis...
from pfs.sacFileHandling import stackedImage, Logbook, constructFilelist
import pfs.imageAnalysis as imeas
from pfs.slitFocusAnalysis import *
from pfs.style import colors


# In[3]:


imgPath = '/home/pfs/shared/Pictures/SM1/SLITALIGN'


# In[4]:


smId = 1


# In[5]:


# filepath of data model
pmodel = '/home/pfs/dev/ait-notebook/optical/input/slit'
file = 'slit-defParam-Ouverture100-fiber65.csv'


# In[6]:


zemaxData = pd.read_csv('%s/%s' % (pmodel, file), delimiter=" ")
zemaxMidFit = imeas.fitparabola(x=zemaxData.Slitdefocus, y=zemaxData.MidFiber, deg=15, focus='max')
zemaxEndFit = imeas.fitparabola(x=zemaxData.Slitdefocus, y=zemaxData.ExtremeFiber, deg=15, focus='max')


# In[7]:


experiments = pd.read_sql_query('select * from Experiment where type="slitAlignment" order by experimentId desc',
                                con='sqlite:////data/ait/experimentLog-sac.db', index_col='experimentId')

experiments['exptime'] = [Logbook.getParameter(experimentId, 'exptime') for experimentId in experiments.index]
experiments['fiber'] = [Logbook.getParameter(experimentId, 'fiber', doRaise=False) for experimentId in experiments.index]


# In[8]:


fiberId = {126:'engtopmid',127:'engbotmid',128:'engtopmid',129:'engtopmid', 130:'engtopmid', 131:'engbotmid', 132:'engbotmid', 133:'engbotmid', 135:'engtopmid'}

experiments['fiber']  = [experiments.fiber[experimentId] if experimentId not in fiberId.keys() else fiberId[experimentId] for experimentId in experiments.index ]


# In[9]:


experiments


# ## Data required : 
# To be able to find the right focus, some data acquisition is required.
# 
# you need to use ics_spsaitActor to perform your sequence
# 
# example : <b>slit throughfocus exptime=6.0 lowBound=-0.5 upBound=1.5 fiber=engbotend nbPosition=10 duplicate=1 </b>
# 
# -0.5 : 1.5 is the range of the slit focus
# 

# ## Input Parameters : 
# The only parameters needed is the experimentIds that match your data acquisition sequence

# In[10]:


com = True
doBck = True
corrector = True
experimentIds = [126, 127, 128, 129, 130, 131, 132, 133]
dfs = []


# In[ ]:


for experimentId in experimentIds:
    dfs.append(getSlitTF(experimentId=experimentId, com=com, doBck=doBck, doPlot=False))

cube = pd.concat(dfs)


# In[ ]:


thfocModel= fitFocusData(cube, corrector=corrector, doPlot=False)


# In[ ]:


focusModel = getFocusModel(thfocModel)


# In[ ]:


vline = True
plotModel = False
criteria = 'EE20'


# In[ ]:


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
j=2

if plotModel:
    ax1.plot(zemaxData.Slitdefocus, zemaxData.MidFiber, 'o', color=colors[0], label='Zemax_MidFiber = %.3f' % zemaxMidFit.focus)
    ax1.plot(zemaxMidFit.x, zemaxMidFit.y, '--', color=colors[0])
    if vline:
        ax1.vlines(color=colors[0], **zemaxMidFit.vline)

    ax1.plot(zemaxData.Slitdefocus, zemaxData.ExtremeFiber, 'o', color=colors[1], label='Zemax_EndFiber = %.3f' % zemaxEndFit.focus)
    ax1.plot(zemaxEndFit.x, zemaxEndFit.y, '--', color=colors[1])
    if vline:
        ax1.vlines(color=colors[1], **zemaxEndFit.vline)

for experimentId, df in cube.groupby('experimentId'):
    fit = thfocModel.query("experimentId==%d"%(experimentId))
    foc = focusModel.query("experimentId==%d and criteria=='%s'"%(experimentId, criteria))
    ax1.plot(df.fca_x, df[criteria], 'o', color=colors[j], 
             label='expId%d:%s = %.3f' % (experimentId, experiments.fiber[experimentId], foc.focus))
    ax1.plot(fit.fca_x, fit[criteria], '--', color=colors[j])
    if vline:
        ax1.vlines(x=foc.focus, ymin=fit[criteria].min(), ymax = fit[criteria].max(), color=colors[j])
    j+=1
    
lns = [line for i, line in enumerate(ax1.get_lines()) if not i % 2]
labs = [line.get_label() for line in lns]

ax1.legend(lns, labs)
ax1.set_xlabel('FCA_X(mm)')
ax1.set_ylabel(criteria)

plt.title('Slit Through focus : Zemax vs Engineering_Fibers \n Criteria : %s' %criteria)
plt.grid()

