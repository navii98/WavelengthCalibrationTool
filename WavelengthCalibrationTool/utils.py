""" This module contains utility functions used by other tools """

import numpy as np
from astropy.modeling import models, fitting
from scipy.optimize import least_squares

def NearestIndex(Array,value):
    """ Returns the index of element in numpy 1d Array nearest to value """
    #return np.searchsorted(Array,value)  # Works for sorted array only
    return np.abs(Array-value).argmin()


def FitLineToData(SpecX,SpecY,Pos,Amp,AmpisBkgSubtracted=True,Sigma = 1.5, WindowStartEnd = None,SpecY_Var=None):
    """ Fits a model line to the SpecX SpecY data 
    Model as well as parameters to fit are defined inside this function
    Sigma = 1.5 # Approximate size of the line width sigma in pixels.
    If WindowStartEnd = None
          Half of the window size to use for fitting line in pixels will be FitWindowH = 5*Sigma  
    else WindowStartEnd = (StartW,EndW)
         will result in data in the wavelength range StartW to EndW to be used for fitting
    SpecY_Var : (optional) if Varience of the `SpecY` is provided weights to line mode fit will be 1/sqrt(SpecY_Var) 
    """

    if WindowStartEnd is None:
        FitWindowH = int(np.rint(5*Sigma))  # window to fit the line is 2*5 expected Sigma of line
        StartIdx = max(NearestIndex(SpecX,Pos) - FitWindowH, 0)
        EndIdx = min(NearestIndex(SpecX,Pos) + FitWindowH +1, len(SpecX)-1)
    else:
        StartIdx = NearestIndex(SpecX,WindowStartEnd[0])
        EndIdx = min(NearestIndex(SpecX,WindowStartEnd[1]) +1, len(SpecX)-1)

    SliceToFitX = SpecX[StartIdx:EndIdx] 
    SliceToFitY = SpecY[StartIdx:EndIdx]
    if SpecY_Var is not None:
        SliceToFitY_Var = SpecY_Var[StartIdx:EndIdx]
    else:
        SliceToFitY_Var = None

    # For fit stability, create a new X coordinates centered on 0
    Xoffset = np.mean(SliceToFitX)
    SliceToFitX_off = SliceToFitX - Xoffset

    if WindowStartEnd is None:
        MedianBkg = np.percentile(SliceToFitY,10)
    else:
        MedianBkg = np.median(SliceToFitY[[0,-1]]) # median of first and last values in the window

    dw = np.abs(np.median(np.diff(SliceToFitX)))
    if not AmpisBkgSubtracted:
        Amp = Amp - MedianBkg

    #Define the line model to fit 
    LineModel = models.Gaussian1D(amplitude=Amp, mean=Pos-Xoffset, stddev=Sigma*dw)+models.Linear1D(slope=0,intercept=MedianBkg)

    #Define fitting object
    Fitter = fitting.LevMarLSQFitter()#SLSQPLSQFitter()
    
    if SliceToFitY_Var is not None:
        weights = 1./np.sqrt(SliceToFitY_Var)
    else :
        weights = None
    #Fit the model to data
    Model_fit = Fitter(LineModel, SliceToFitX_off, SliceToFitY,weights=weights)
    # Add back the offset in X
    Model_fit.mean_0.value = Model_fit.mean_0.value + Xoffset
    Model_fit.intercept_1.value = Model_fit.intercept_1.value - Model_fit.slope_1.value*Xoffset
    return Model_fit
