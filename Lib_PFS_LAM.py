import sqlite3
import os
from skimage.feature import register_translation
import pandas as pd
from astropy.io import fits
from datetime import datetime as dt
import numpy as np
from scipy.optimize import curve_fit


# Exposure logbook class

class Logbook:
    engine = '///data/ait/ait-alignment.db'
    
    @staticmethod
    def lastExperimentId():

        conn = sqlite3.connect(Logbook.engine)
        c = conn.cursor()
        c.execute("""SELECT MAX(experimentId) FROM Experiment""")
        (experimentId,) = c.fetchone()
        return experimentId

    @staticmethod
    def visitRange(experimentId=False):
        experimentId = Logbook.lastExperimentId() if not experimentId else experimentId
        
        conn = sqlite3.connect(Logbook.engine)
        c = conn.cursor()

        c.execute('''select visitStart,visitEnd from Experiment where ExperimentId=%i''' % experimentId)
        (visitStart, visitEnd) = c.fetchone()

        return visitStart, visitEnd
    
    @staticmethod
    def cmdStr(experimentId=False):
        experimentId = Logbook.lastExperimentId() if not experimentId else experimentId
        
        conn = sqlite3.connect(Logbook.engine)
        c = conn.cursor()

        c.execute('''select cmdStr from Experiment where ExperimentId=%i''' % experimentId)
        (cmdStr,) = c.fetchone()

        return cmdStr
    
    @staticmethod
    def getParameter(experimentId, param):
        cmdStr = Logbook.cmdStr(experimentId=experimentId)
        res = cmdStr.split('%s='%param)
        if len(res)==1:
            raise ValueError('parameter %s in not in the command'%param)
            
        return res[1].split(" ")[0]
    
    @staticmethod
    def newAnomalies(experimentId, anomalies):
        sqlRequest = 'UPDATE Experiment SET anomalies = "%s" WHERE experimentId=%i' % (anomalies.replace('"', ""),
                                                                                       experimentId)
        Logbook.newRow(sqlRequest=sqlRequest)
        
        
    @staticmethod
    def newRow(sqlRequest):
        conn = sqlite3.connect(Logbook.engine)
        c = conn.cursor()
        try:
            c.execute(sqlRequest)
            conn.commit()

        except sqlite3.IntegrityError:
            pass
        
        
def constructFilelist(visitStart, visitEnd):
    visits = {}
    directory = "/data/ait/sac"
    datefolders = os.listdir(directory)
    datefolders.remove('nextSeqno')

    for datefolder in datefolders:
        files = os.listdir(os.path.join(directory, datefolder))
        for file in files:
            try: 
                visits[int(file[3:9])] = os.path.join(directory, datefolder, file)

            except:
                pass
    
    filelist = []
    for visit in range(visitStart, visitEnd+1):
        if visit not in visits.keys():
            print ('visit %i is missing !'%visit)
        else:
            filelist.append(visits[visit])
    
    filelist.sort()
    
    return filelist


def stackedImage(filelist, ind, duplicate):
    sublist = filelist[ind*duplicate:ind*duplicate+duplicate]
    first = sublist[0]
    img = fits.open(first)
    hdr =  img[0].header
    data = img[0].data
    
    for filepath in sublist[1:]:
        img = fits.open(filepath)
        data += img[0].data
    
    return hdr, data/duplicate

        
class TFocusDf(pd.DataFrame):
    def __init__(self, data, model='gaussian'):
        pd.DataFrame.__init__(self, data=data, columns=['x', 'y'])
        self.model = model
        
    @property
    def focus(self):
        if self.model=='gaussian':
            return self.x[self.y.idxmax()]
        else:
            return self.x[self.y.idxmin()]
    
    @property
    def vline(self):
        return dict(x=self.focus, ymin=min(self.y), ymax=max(self.y))


# Data Analysis 

def getEE(image, peak, inner_size=20, outer_size=100):

    indx = peak['objy']
    indy = peak['objx']

    inner_data = image[int(indx-inner_size/2):int(indx+inner_size/2), int(indy-inner_size/2):int(indy+inner_size/2)]
    outer_data = image[int(indx-outer_size/2):int(indx+outer_size/2), int(indy-outer_size/2):int(indy+outer_size/2)]
    
    return np.sum(inner_data)/np.sum(outer_data)


def oneD_Gaussian(x, amp, mu, sig, offset):
    return offset + amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def fitgauss1D(x, y):
    popt1, pcov = curve_fit(oneD_Gaussian, x, y, p0=[np.max(y), x[y.idxmax()], 0.3, 0], maxfev=10000)
    newx = np.arange(np.min(x), np.max(x) + 0.001, 0.001)
    data = np.zeros((len(newx), 2))
    data[:,0] = newx     
    data[:,1] = oneD_Gaussian(newx, *popt1)

    return TFocusDf(data=data)

def fitparabola(x, y):
    c = np.polyfit(x, y, 2)
    newx = np.arange(np.min(x), np.max(x) + 0.001, 0.001)
    data = np.zeros((len(newx), 2))
    data[:,0] = newx     
    data[:,1] = np.polyval(c, newx)
    
    return TFocusDf(data=data, model='parabola')
    