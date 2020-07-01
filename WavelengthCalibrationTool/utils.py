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

def create_orthogonal_polynomials_ttr(deg,x,weights=None):
    """ Creates Discrete Orthogonal Polynomial Basis function of degree `deg`, at locations `x`, weights by given by `weights` 
    Parameters
    ----------
    deg : int
          Degree of the polynomials to return
    x  : 1D numpy array
          Discrete points in domain to evaluvate the polynomial
    weights: 1D numpy array (optional; default: np.ones(len(x)))
          Weights under which the ploynomials should be orthogonal

    Returns
    -------
    base_polynomials : list of polynomials
         List of orthogonal np.polynomial.Polynomial objects

    Algorithm
    --------
    Uses the Three Term Recurrence formula to calculate the orthogonal polynomials. This is more stable than Gram-Schmidt procedure.
    Reference : A First Course in Numerical Analysis 2nd ed. Ralston, Rabinowitz. Page 256
    Equations: 6.4-14,15,17,20,21
    """
    if weights is None:
        weights = np.ones(len(x))
    p_m1 = np.polynomial.Polynomial(coef=[0.])
    p_0 = np.polynomial.Polynomial(coef=[1.])
    base_polynomials = [p_m1,p_0]
    p_x = np.polynomial.Polynomial(coef=[0,1.])

    for jp1 in range(1,deg+1):  # loop through j+1 start with j+1=1
        p_j = base_polynomials[jp1-1 +1]  # +1 since the first element is p_m1
        p_jm1 = base_polynomials[jp1-2 +1]
        if jp1 == 1 :
            alpha_jp1 = np.sum(weights*x)/np.sum(weights)
            beta_j = 0  # setting it to zero to avoid divide by zero
        else:
            alpha_jp1 = np.sum(weights*x*p_j(x)**2)/np.sum(weights*p_j(x)**2)
            beta_j = np.sum(weights*p_j(x)**2)/np.sum(weights*p_jm1(x)**2)
        p_jp1 = (p_x-alpha_jp1)*p_j - beta_j*p_jm1
        base_polynomials.append(p_jp1)

    return base_polynomials[1:] # skip the first p_m1 term


def eval_polynomial_basis(x,coeffs,p_list):
    """ Returns the evaluvated values at `x` using the lineary combination of `p_list` with linear combination coefficent `coeffs` """
    x = np.array(x)
    return np.sum([c*p(x) for c,p in zip(coeffs,p_list)],axis=0)
    
def fit_polynomial_basis(X,Y,p_list,full_output=False):
    """ Fits `X` versus `Y` using a linear combination of the polynomial list `p_list` """
    def error_func(c_values):
        return eval_polynomial_basis(X,c_values,p_list) - np.array(Y)
    
    c0 = np.ones(len(p_list))
    res = least_squares(error_func, c0)
    # print(res.message,res.status,res.success)
    if full_output:
        return res.x, res
    else:
        return res.x
