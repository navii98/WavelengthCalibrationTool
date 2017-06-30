""" This module contains utility functions used by other tools """

import numpy as np
from astropy.modeling import models, fitting


def NearestIndex(Array,value):
    """ Returns the index of element in numpy 1d Array nearest to value """
    return np.abs(Array-value).argmin()

def FitLineToData(SpecX,SpecY,Pos,Amp,AmpisBkgSubtracted=True,Sigma = 3):
    """ Fits a model line to the SpecX SpecY data 
    Model as well as parameters to fit are defined inside this function
    Sigma = 3 # Approximate size of the line width sigma in pixels.
    Half of the window size to use for fitting line in pixels will be FitWindowH = 5*Sigma  """

    FitWindowH = 5*Sigma  # window to fit the line is 5* expected Sigma of line
    StartIdx = NearestIndex(SpecX,Pos) - FitWindowH
    EndIdx = NearestIndex(SpecX,Pos) + FitWindowH
    SliceToFitX = SpecX[StartIdx:EndIdx+1] 
    SliceToFitY = SpecY[StartIdx:EndIdx+1]
    MedianBkg = np.median(SliceToFitY)
    dw = np.abs(np.median(SliceToFitX[1:]-SliceToFitX[:-1]))
    if not AmpisBkgSubtracted:
        Amp = Amp - MedianBkg

    #Define the line model to fit 
    LineModel = models.Gaussian1D(amplitude=Amp, mean=Pos, stddev=Sigma*dw)+models.Linear1D(slope=0,intercept=MedianBkg)

    #Define fitting object
    Fitter = fitting.LevMarLSQFitter()#SLSQPLSQFitter()
    
    #Fit the model to data
    Model_fit = Fitter(LineModel, SliceToFitX, SliceToFitY)  
    return Model_fit
