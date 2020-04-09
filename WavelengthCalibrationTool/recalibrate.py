#!/usr/bin/env python
""" This is a non-interactive tool to re-calibrate wavelength solution based on
an existing calibrated solution of the same lamp in same instrument."""
import sys
import argparse
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as optimize
from scipy import linalg 
from scipy.constants import speed_of_light
from functools32 import partial

def scale_interval_m1top1(x,a,b,inverse_scale=False):
    """ Scales input x in interval a to b to the range -1 to 1 """
    if inverse_scale: # convert form -1 to 1 scale back to a to b scale
        return ((b-a)*x + a+b)/2.0
    else:
        return (2.0*x - (b+a))/(b-a)

def LCTransformMatrixPV(v,p,deg=6):
    """ Returns the Legendre coefficent transform matrix pf velocity and pixel shift"""
    TM_vel = np.matrix(np.identity(deg+1))*(1+v)  # Velocity compoent alone
    TM_pix = np.matrix(np.identity(deg+1))
    for i in range(TM_pix.shape[1]):
        for j in range(i+1,TM_pix.shape[0],2):
            TM_pix[i,j]=p*(i*2 +1)
    return np.matmul(TM_pix,TM_vel)

def TransformLegendreCoeffs(LC,PVWdic):
    """ Returns the transfromed Legendre coefficent based on the pixel shift, velocity shft, and Wavelength shift values 
    LC: Input coefficents
    PVWdic: Dictionary containing the transformation coefficents for p,v, and w.
    Returns :
    T_LC: Transformed LC = P*V*(LC+w)
    """
    ldeg = len(LC)-1
    w = np.zeros(ldeg+1)
    w[0] = PVWdic['w']

    v = PVWdic['v']
    p = PVWdic['p']#*2./len(RefWavl)

    PV = LCTransformMatrixPV(v,p,deg=ldeg)
    T_LC = np.matmul(PV,LC+w)
    return T_LC


def transformed_spectrum(FluxSpec, *params, **kwargs):
    """ Returns transformed and interpolated scaled FluxSpec for fitting with data .
    Allowed **kwargs are 
           method: 'p' for normal polynomial
                   'c' for chebyshev  (default)
           WavlCoords : Coordinates to fit the transformation; (default: pixel coords)
    """
    
    if 'WavlCoords' in kwargs:
        Xoriginal = kwargs['WavlCoords']
    else:
        Xoriginal = np.arange(len(FluxSpec))

    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = 'c'

    # First paramete is the flux scaling
    scaledFlux = FluxSpec*params[0]

    # Remaing parameters for defining the ploynomial drift of coordinates
    if len(params[1:]) == 1:  # Zero offset coeff only
        coeffs =  params[1:] + (1,)  # Add fixed 1 slope
    else:   
        coeffs =  params[1:]  # Use all the coefficents for transforming polynomial 

    if method == 'p':
        Xtransformed = np.polynomial.polynomial.polyval(Xoriginal, coeffs)
    elif method == 'c':
        Xtransformed = np.polynomial.chebyshev.chebval(Xoriginal, coeffs)
    elif method == 'v': # (1+v/c) shift
        Xtransformed = Xoriginal*(1+coeffs[0]/speed_of_light)
    elif method == 'x': # (w + w v/c +P dw/dp) combined shift
        Xtransformed = Xoriginal*(1+coeffs[0]/speed_of_light) + coeffs[1]*np.gradient(Xoriginal)
    else:
        print('method {0} is not implemented'.format(method))
        return None
        
    # interpolate the original spectrum to new coordinates
    tck = interp.splrep(Xoriginal, scaledFlux)
    return interp.splev(Xtransformed, tck)


def errorfunc_tominimise(params,method='l',Reg=0,RefSpectrum=None,DataToFit=None,sigma=None,defaultParamDic=None,**kargs ):
    """ Error function to minimise to fit model.
    Currently implemented for only the regularised fitting of Legendre coefficent transform
    Reg is the Regularisation coefficent for LASSO regularisation.
    defaultParamDic: is the dictionary of the deafult values for all the parameters which can include parameters not being fitted.
                      For example: for method=l , defaultParamDic = {'v':0,'p':0,'w':0}"""
    
    # First paramete is the flux scaling
    scaledFlux = RefSpectrum*params[0]
    if method == 'l':
        grid = np.linspace(-1,1,len(RefSpectrum))
        if 'WavlCoords' in kwargs:
            Xoriginal = kargs['WavlCoords']
        else:
            Xoriginal = None
        if 'LCRef' in kwargs:
            LCRef = kargs['LCRef']
            if Xoriginal is None:
                Xoriginal = np.polynomial.legendre.legval(grid,LCRef) 
        else:
            if ('ldeg' in kwargs['ldeg']) and ('WavlCoords' in kwargs) :
                LCRef = np.polynomial.legendre.legfit(grid,Xoriginal,deg=ldeg)
        paramstring = kwargs['paramstring']
        if defaultParamDic is None:
            PVWdic = {'v':0,'p':0,'w':0}
        else:
            PVWdic = defaultParamDic
        for i,s in enumerate(paramstring):
            PVWdic[s] = params[i+1]
        LCnew = TransformLegendreCoeffs(LCRef,PVWdic)

        Xtransformed = np.polynomial.legendre.legval(grid,LCnew) 
    else:
        print('method {0} is not implemented'.format(method))
        return None

    # interpolate the original spectrum to new coordinates
    tck = interp.splrep(Xoriginal, scaledFlux)
    PredictedSpectrum = interp.splev(Xtransformed, tck)
    if sigma is None:
        sigma=1
    return  np.concatenate(((PredictedSpectrum-DataToFit)/sigma,Reg*np.abs(params[1:])))

    

def ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3',sigma=None,cov=False,Reg=0,defaultParamDic=None):
    """ Recalibrate the dispertion solution of SpectrumY using 
    RefSpectrum by fitting the relative drift using the input method.
    Input:
       SpectrumY: Un-calibrated Spectrum Flux array
       RefSpectrum: Wavelength Calibrated reference spectrum (Flux vs wavelegnth array:(N,2))
       method: (str, default: p3) the method used to model and fit the drift in calibration
       sigma: See sigma arg of scipy.optimize.curve_fit ; it is the inverse weights for residuals
       cov: (bool, default False) Set cov=True to return an estimate of the covarience matrix of parameters 
       Reg: Regularisation parameter for LASSO (Currently implemented only for multi parameter Legendre polynomials) 
       defaultParamDic: Default values for parameters in a multi parameter model. Example for l* methods. 
      Available methods: 
               pN : Fits a Nth order polynomial distortion  
               cN : Fits a Nth order Chebyshev polynomial distortion 
               v  : Fits a velocity redshift distortion
               x  : Fits a velocity redshift distortion and a 0th order pixel shift distortion 
               lwN: Fits Nth order Legendre coefficent transform with wavelenth shift as the single parameter 
               lvN: Fits Nth order Legendre coefficent transform with velocity shift as the single parameter
               lpN: Fits Nth order Legendre coefficent transform with pixel shift as the single parameter 
               lpwN: Fits Nth order Legendre coefficent transform with pixel shift and wavelenth shift as two parameters
               lvwN: Fits Nth order Legendre coefficent transform with velocity shift and wavelenth shift as two parameters
               lpvN: Fits Nth order Legendre coefficent transform with pixel shift and velocity shift as two parameters
               lpvwN: Fits Nth order Legendre coefficent transform with pixel shift, velocity shift, and wavelength shift as three parameters
    Returns:
        wavl_sln : Output wavelength solution
        fitted_drift : the fitted calibration drift coeffients 
                    (IMP: These coeffs is for the method and scaling done inside this function)
    """
    RefFlux = RefSpectrum[:,1]
    RefWavl = RefSpectrum[:,0]

    # For stability and fast convergence lets scale the wavelength to -1 to 1 interval. (Except for doppler shift method)
    if method[0] in ['p','c']:
        scaledWavl = scale_interval_m1top1(RefWavl,a=min(RefWavl),b=max(RefWavl))

    if (method[0] == 'p') and method[1:].isdigit():
        # Use polynomial of p* degree.
        deg = int(method[1:])
        # Initial estimate of the parameters to fit
        # [scalefactor,*p] where p is the polynomial coefficents
        if deg > 0:
            p0 = [1,0,1]+[0]*(deg-1)
        else:
            p0 = [1,0]
        poly_transformedSpectofit = partial(transformed_spectrum,method='p',WavlCoords=scaledWavl)
        popt, pcov = optimize.curve_fit(poly_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        if deg < 1: # Append slope 1 coeff
            popt = np.concatenate([popt, [1]])
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        transformed_scaledWavl = np.polynomial.polynomial.polyval(scaledWavl, popt[1:])

    elif (method[0] == 'c') and method[1:].isdigit():
        # Use chebyshev polynomial of c* degree.
        deg = int(method[1:])
        # Initial estimate of the parameters to fit
        # [scalefactor,*c] where c is the chebyshev polynomial coefficents
        if deg > 0:
            p0 = [1,0,1]+[0]*(deg-1)
        else:
            p0 = [1,0]
        cheb_transformedSpectofit = partial(transformed_spectrum,method='c',WavlCoords=scaledWavl)
        popt, pcov = optimize.curve_fit(cheb_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        if deg < 1: # Append slope 1 coeff
            popt = np.concatenate([popt, [1]])
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        transformed_scaledWavl = np.polynomial.chebyshev.chebval(scaledWavl, popt[1:])

    elif (method[0] == 'v'):
        # Use the 1+v/c formula to shift the spectrum. Usefull in grating sectrogrpahs where flexure is before grating.
        # Initial estimate of the parameters to fit  [1 for scaling, and 100 for velocity]
        p0 = [1,100]
        vel_transformedSpectofit = partial(transformed_spectrum,method='v',WavlCoords=RefWavl)
        popt, pcov = optimize.curve_fit(vel_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        wavl_sln = RefWavl *(1+ popt[1]/speed_of_light)

    elif (method[0] == 'x'):
        # Use the w+ w*v/c + deltaP*dw/dp formula to shift the spectrum. Usefull in grating sectrogrpahs where flexure is before grating as well as after grating.
        # Initial estimate of the parameters to fit  [1 for scaling, and 100 for velocity,0 for pixshift]
        p0 = [1,100,0]
        velp_transformedSpectofit = partial(transformed_spectrum,method='x',WavlCoords=RefWavl)
        popt, pcov = optimize.curve_fit(velp_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        wavl_sln = RefWavl *(1+ popt[1]/speed_of_light) + popt[2]*np.gradient(RefWavl)


    elif (method[0] == 'l'):
        # Usefull in grating sectrogrpahs where flexure is before grating as well as after grating.
        # Parameters to fit
        if defaultParamDic is None:
            PVWdic = {'v':0,'p':0,'w':0}
        else:
            PVWdic = defaultParamDic

        paramstring = [s for s in method[1:] if not s.isdigit()]
        ldeg = int(''.join([s for s in  method[1:] if s.isdigit()]))

        # Legendre coefficents for the polynomial
        grid = np.linspace(-1,1,len(RefWavl))
        LCRef = np.polynomial.legendre.legfit(grid,RefWavl,deg=ldeg)
        
        Initp={'v':1e-4,'p':0.001,'w':0.01}
        # Initial estimate of the parameters to fit  [0 for each parameter to fit]
        p0 = [1]+[Initp[s] for s in paramstring]  # 1 is for scaling, rest are the parameters
        l_errorfunc_tominimise = partial(errorfunc_tominimise,method='l',Reg=Reg,paramstofit=paramstring,
                                         WavlCoords=RefWavl,RefSpectrum=RefFlux,DataToFit=SpectrumY,sigma=sigma,
                                         LCRef=LCRef,defaultParamDic=PVWdic) 
        fitoutput = optimize.least_squares(l_errorfunc_tominimise,p0)
        popt = fitoutput['x'] 
        if cov :
            # Calculate pcov based on scipy.curve_fit code ##################################
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = linalg.svd(fitoutput.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(fitoutput.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            pcov = np.dot(VT.T / s**2, VT)
        ############################################################# End of code form scipy.curve_fit
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        for i,s in enumerate(paramstring):
            PVWdic[s] = popt[i+1]
        LCnew = TransformLegendreCoeffs(LCRef,PVWdic)

        wavl_sln = np.polynomial.legendre.legval(grid,LCnew) 

    else:
        raise NotImplementedError('Unknown fitting method {0}'.format(method))

    if method[0] in ['p','c']:
        wavl_sln = scale_interval_m1top1(transformed_scaledWavl,
                                         a=min(RefWavl),b=max(RefWavl),
                                         inverse_scale=True)

    if cov :
        return wavl_sln, popt, pcov
    else:
        return wavl_sln, popt
        

def parse_args(raw_args=None):
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Non-Interactive Wavelength Re-Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the uncalibrated Spectrum Flux array")
    parser.add_argument('RefSpectrumFile', type=str,
                        help="Reference Spectrum file which is calibrated, containing Flux vs wavelengths for the same pixels")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    args = parser.parse_args(raw_args)
    return args
    
def main(raw_args=None):
    """ Standalone Interactive Line Identify Tool """
    args = parse_args(raw_args)
    SpectrumY = np.load(args.SpectrumFluxFile)
    RefSpectrum = np.load(args.RefSpectrumFile)
    Output_fname = args.OutputWavlFile
    wavl_sln, fitted_drift = ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3')
    np.save(Output_fname,wavl_sln)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
