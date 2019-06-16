#!/usr/bin/env python
""" This is a non-interactive tool to re-calibrate wavelength solution based on
an existing calibrated solution of the same lamp in same instrument."""
import sys
import argparse
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as optimize
from scipy.constants import speed_of_light
from functools32 import partial

def scale_interval_m1top1(x,a,b,inverse_scale=False):
    """ Scales input x in interval a to b to the range -1 to 1 """
    if inverse_scale: # convert form -1 to 1 scale back to a to b scale
        return ((b-a)*x + a+b)/2.0
    else:
        return (2.0*x - (b+a))/(b-a)

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
        
    # interpolate the original spectrum to new coordinates
    tck = interp.splrep(Xoriginal, scaledFlux)
    return interp.splev(Xtransformed, tck)

def ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3',sigma=None):
    """ Recalibrate the dispertion solution of SpectrumY using 
    RefSpectrum by fitting the relative drift using the input method.
    Input:
       SpectrumY: Un-calibrated Spectrum Flux array
       RefSpectrum: Wavelength Calibrated reference spectrum (Flux vs wavelegnth array:(N,2))
       method: (str, default: p3) the method used to model and fit the drift in calibration
       sigma: See sigma arg of scipy.optimize.curve_fit ; it is the inverse weights for residuals

    Returns:
        wavl_sln : Output wavelength solution
        fitted_drift : the fitted calibration drift coeffients 
                    (IMP: These coeffs is for the method and scaling done inside this function)
    """
    RefFlux = RefSpectrum[:,1]
    RefWavl = RefSpectrum[:,0]

    # For stability and fast convergence lets scale the wavelength to -1 to 1 interval. (Except for doppler shift method)
    if method[0] != 'v':
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
        # Initial estimate of the parameters to fit  [1 for scaling, and 0 for velocity]
        p0 = [1,0]
        vel_transformedSpectofit = partial(transformed_spectrum,method='v',WavlCoords=RefWavl)
        popt, pcov = optimize.curve_fit(vel_transformedSpectofit, RefFlux, SpectrumY, p0=p0,sigma=sigma)
        # Now we shall use the transformation obtained for scaled Ref Wavl coordinates
        # to transform the calibrated wavelength array.
        wavl_sln = RefWavl *(1+ popt[1]/speed_of_light)
    else:
        raise NotImplementedError('Unknown fitting method {0}'.format(method))

    if method[0] != 'v':
        wavl_sln = scale_interval_m1top1(transformed_scaledWavl,
                                         a=min(RefWavl),b=max(RefWavl),
                                         inverse_scale=True)

    return wavl_sln, popt
        

def parse_args():
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Non-Interactive Wavelength Re-Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the uncalibrated Spectrum Flux array")
    parser.add_argument('RefSpectrumFile', type=str,
                        help="Reference Spectrum file which is calibrated, containing Flux vs wavelengths for the same pixels")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    args = parser.parse_args()
    return args
    
def main():
    """ Standalone Interactive Line Identify Tool """
    args = parse_args()    
    SpectrumY = np.load(args.SpectrumFluxFile)
    RefSpectrum = np.load(args.RefSpectrumFile)
    Output_fname = args.OutputWavlFile
    wavl_sln, fitted_drift = ReCalibrateDispersionSolution(SpectrumY,RefSpectrum,method='p3')
    np.save(Output_fname,wavl_sln)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
