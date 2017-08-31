#!/usr/bin/env python
""" This is a non-interactive tool to re-identify calibration lamp lines from 
an already identified dispersion file """
import os
import argparse
import numpy as np
from .iidentify import TryToFitNewLinesinSpectrum, FittedFunction, read_dispersion_inputfile

def parse_args():
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Non-Interactive Wavelength Re-Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the Spectrum's Flux array")
    parser.add_argument('RefDispTableFile', type=str,
                        help="Reference file containing the table of Wavelengths, Pixels, Error")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    parser.add_argument('OutDispTableFile', type=str,
                        help="Output Filename to write the table of Wavelengths, Pixels, Error")
    args = parser.parse_args()
    return args
    
def main():
    """ Standalone Non-Interactive Line Re-Identify Tool """
    args = parse_args()    
    SpectrumY = np.load(args.SpectrumFluxFile)
    Refdisp_fname = args.RefDispTableFile
    Output_fname = args.OutputWavlFile
    Outdisp_fname = args.OutDispTableFile
    if os.path.isfile(Outdisp_fname):
        print('Output dispersion file {0} exists.'.format(Outdisp_fname))
        print('Only non-calibrated wavelenghts in it will be re-fitted..')
    else:
        # Create the dispersion file with all the wavelengths in Reference file
        wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(Refdisp_fname)
        with open(Outdisp_fname,'w') as outdispfile:
            outdispfile.write('# Wavelengths Pixel Sigma # comments\n')
            for wavel in wavelengths_inp + wavelengths_tofit :
                outdispfile.write('{0} # Re-Fitted\n'.format(wavel))
        
    # Now recalibrate the positions of the line by fitting lines again to new positions
    TryToFitNewLinesinSpectrum(SpectrumY,Outdisp_fname,LineSigma=3,
                               reference_dispfile = Refdisp_fname)
    # Read back the calibration
    _ , (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(Outdisp_fname)

    # Dispertion function
    disp_func = FittedFunction(pixels=pixels_inp,
                               wavel=wavelengths_inp,
                               sigma=sigma_inp, method='c3')

    wavl = disp_func(np.arange(len(SpectrumY)))
    print('Dispersion ASCII input file saved in file://{0}'.format(Outdisp_fname))
    np.save(Output_fname,wavl)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
