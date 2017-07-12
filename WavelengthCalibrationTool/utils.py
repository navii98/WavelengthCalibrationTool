""" This module contains utility functions used by other tools """

import numpy as np
from astropy.modeling import models, fitting


def NearestIndex(Array,value):
    """ Returns the index of element in numpy 1d Array nearest to value """
    #return np.searchsorted(Array,value)  # Works for sorted array only
    return np.abs(Array-value).argmin()


def FitLineToData(SpecX,SpecY,Pos,Amp,AmpisBkgSubtracted=True,Sigma = 3, WindowStartEnd = None):
    """ Fits a model line to the SpecX SpecY data 
    Model as well as parameters to fit are defined inside this function
    Sigma = 3 # Approximate size of the line width sigma in pixels.
    If WindowStartEnd = None
          Half of the window size to use for fitting line in pixels will be FitWindowH = 5*Sigma  
    else WindowStartEnd = (StartW,EndW)
         will result in data in the wavelength range StartW to EndW to be used for fitting
    """

    if WindowStartEnd is None:
        FitWindowH = 5*Sigma  # window to fit the line is 5* expected Sigma of line
        StartIdx = NearestIndex(SpecX,Pos) - FitWindowH
        EndIdx = min(NearestIndex(SpecX,Pos) + FitWindowH +1, len(SpecX)-1)
    else:
        StartIdx = NearestIndex(SpecX,WindowStartEnd[0])
        EndIdx = min(NearestIndex(SpecX,WindowStartEnd[1]) +1, len(SpecX)-1)

    SliceToFitX = SpecX[StartIdx:EndIdx] 
    SliceToFitY = SpecY[StartIdx:EndIdx]

    if WindowStartEnd is None:
        MedianBkg = np.median(SliceToFitY)
    else:
        MedianBkg = np.median(SliceToFitY[[0,-1]]) # median of first and last values in the window

    dw = np.abs(np.median(np.diff(SliceToFitX)))
    if not AmpisBkgSubtracted:
        Amp = Amp - MedianBkg

    #Define the line model to fit 
    LineModel = models.Gaussian1D(amplitude=Amp, mean=Pos, stddev=Sigma*dw)+models.Linear1D(slope=0,intercept=MedianBkg)

    #Define fitting object
    Fitter = fitting.LevMarLSQFitter()#SLSQPLSQFitter()
    
    #Fit the model to data
    Model_fit = Fitter(LineModel, SliceToFitX, SliceToFitY)  
    return Model_fit
