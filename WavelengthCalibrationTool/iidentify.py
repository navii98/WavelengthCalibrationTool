#!/usr/bin/env python
""" This is an interactive tool to identify calibration lamp lines from line atlas """
from .utils import NearestIndex, FitLineToData
import sys
import shutil
import uuid
import time
import readline
import argparse
from multiprocessing import Process, Pipe
import numpy as np
import matplotlib.pyplot as plt


def SelectLinesInteractively(SpectrumX,SpectrumY,ExtraUserInput=None,comm_pipe=None, LineSigma=3):
    """ Fits Gaussian to all points user press m and c to confirm
    ExtaUserInput : (str) If provided will ask user from the extra input for each line 
                        Example: 'Wavelength'

    comm_pipe : One end of the Pipe which needs to be used for getting raw_input from parent process.
    LineSigma : Approximated Sigma width of the line in pixels. 
                Size of the window region to use for fitting line will be 5*LineSigma.
    Returns : List of Fitted Line Models (and user input if any).

    IMP: This should always be run as a seperate Process.
    """
    print('Close the poped up plot window after done with selection')
    figi = plt.figure()
    ax = figi.add_subplot(111)
    ax.set_title('Press m to select line, then c to confirm the selection.')
    ax.plot(SpectrumX,SpectrumY,linestyle='--', drawstyle='steps-mid',marker='.',color='k')
    junk, = ax.plot([],[])
    PlotedLines=[junk]
    LinesConfirmedToReturn = []
    LatestLineModel = [None]
    # Define the function to run while key is pressed
    def on_key(event):
        if event.key == 'm' :
            Xpos = event.xdata
            Model_fit = FitLineToData(SpectrumX,SpectrumY,Xpos,event.ydata,
                                      AmpisBkgSubtracted=False,Sigma = LineSigma)
            if PlotedLines:
                ax.lines.remove(PlotedLines[-1])  # Remove the last entry in the plotedlines
            IndxCenter = NearestIndex(SpectrumX,Model_fit.mean_0.value)
            SliceToPlotX = SpectrumX[IndxCenter-4*LineSigma:IndxCenter+4*LineSigma+1]
            linefit, =  ax.plot(SliceToPlotX,Model_fit(SliceToPlotX),color='r')
            PlotedLines.append(linefit)
            ax.figure.canvas.draw()

            LatestLineModel[0] = Model_fit
            print(Model_fit)


        elif event.key == 'c' :
            print('Fit acceptence confirmed.')
            junk, = ax.plot([],[])
            PlotedLines.append(junk)  # Add a new junk plot to end of ploted lines

            if ExtraUserInput is not None:  # Ask user for the extra input
                comm_pipe.send_bytes(ExtraUserInput+ ' : ')
                Uinput = comm_pipe.recv_bytes()
                LinesConfirmedToReturn.append((LatestLineModel[0],Uinput))
            else:
                LinesConfirmedToReturn.append(LatestLineModel[0])
            
    cid = figi.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    comm_pipe.send_bytes('')
    return LinesConfirmedToReturn

def FittedFunction(pixels,wavel,sigma=None,method='p3'):
    """ Returns the fitted function f(pixels) = wavel .
    Define all the methods to use for fitting here """
    output_object = None

    if (method[0] == 'p') and method[1:].isdigit():
        # Use polynomial of p* degree.
        deg = int(method[1:])
        p,residuals,rank,sing_values,rcond = np.polyfit(pixels,wavel,deg,w=1/np.array(sigma),full=True)
        print('Stats of the poly fit of degree {0}'.format(deg))
        print('Coeffs p: {0}'.format(p))
        print('residuals:{0},  rank:{1}, singular_values:{2}, rcond:{3}'.format(residuals,
                                                                                rank,sing_values,
                                                                                rcond))
        output_object = np.poly1d(p)
    else:
        print('Error: unknown fitting method {0}'.format(method))

    return output_object

def read_dispersion_inputfile(filename):
    """ Reads the Dispersion Input text file.
    Returns a list of wavelengths without pixel ref
    And a tuple of (wavelengths, pixels, sigma) containing lists"""
    wavelengths_without_pixels = []
    wavelengths = []
    pixels = []
    sigma = []
    
    with open(filename) as f:
        for line in f:
            if line[0] == '#' : continue  # commented lines
            line = line.rstrip().split('#')[0] # remove any inline comments
            if len(line.split()) == 1:
                wavelengths_without_pixels.append(float(line))
            elif len(line.split()) == 3:
                wavelengths.append(float(line.split()[0]))
                pixels.append(float(line.split()[1]))
                sigma.append(float(line.split()[2]))
            elif len(line.split()) == 2:
                wavelengths.append(float(line.split()[0]))
                pixels.append(float(line.split()[1]))
                sigma.append(1) # default 1 pix as sigma
    return wavelengths_without_pixels, (wavelengths,pixels,sigma)

def writeto_dispersion_inputfile(filename,wavelengths,pixels,sigma,backupsuffix='.bak'):
    """ writes the fitted pixel positions to inputfile .
    Oldfile is backedup with .bak prefix"""
    # First backup file # Old backup will be overwritten
    shutil.move(filename,filename+backupsuffix)
    
    already_addedinfile = []
    with open(filename+backupsuffix,'r') as oldfile, open(filename,'w') as newfile:
        for line in oldfile:
            try:
                wavel = float(line.rstrip().split()[0])
            except ValueError:
                pass
            else:
                if wavel in wavelengths:
                    ind = wavelengths.index(wavel)
                    textindx = len(line.rstrip().split()[0])
                    line = line[:textindx] +\
                           ' {0} {1} #'.format(pixels[ind],sigma[ind]) +\
                           line[textindx:]
                    already_addedinfile.append(wavel)
            finally:
                newfile.write(line)

        # Now append any remaining new lines
        for wavel,pix,w in zip(wavelengths,pixels,sigma):
            if wavel not in already_addedinfile:
                newfile.write('{0} {1} {2}\n'.format(wavel,pix,w))

def update_main_figure(fig_main,SpectrumY,wavl_pix_sigma):
    """ Updates the main plot of the latest dispersion fit.
    Also returns the latest dispersion function used"""
    print(wavl_pix_sigma)
    wavelengths_inp,pixels_inp,sigma_inp = wavl_pix_sigma
    disp_func = FittedFunction(pixels=pixels_inp,
                               wavel=wavelengths_inp,
                               sigma=sigma_inp, method='p3')
    fig_main.clf()
    ax = fig_main.add_subplot(111)
    ax.plot(disp_func(np.arange(len(SpectrumY))),SpectrumY,
            linestyle='-', drawstyle='steps-mid',color='k') #marker='.'
    #Colormap which is red when maximum
    colormap = plt.get_cmap('PuRd')
    pointYlevel = np.median(SpectrumY)
    residuesW = [np.abs(w-disp_func(pix))/np.abs(disp_func(pix)-disp_func(pix+sig)) \
                 for w,pix,sig in zip(*wavl_pix_sigma)]

    NresiduesW = np.array(residuesW)/max(residuesW)
    print('RMS of residue: {0}'.format(np.sqrt(np.mean(np.power(residuesW,2)))))
    print('RMS of Normalised residue: {0}'.format(np.sqrt(np.mean(np.power(NresiduesW,2)))))
    print('Individual Normalised Residues of fit:')
    print('Wavel  Pixel  Nresidue')
    for w,pix,Nresid in zip(wavelengths_inp,pixels_inp,NresiduesW):
        ax.plot([w,disp_func(pix)],[pointYlevel]*2,color=colormap(int(Nresid*255)))
        ax.plot([w],[pointYlevel],'.',color=colormap(int(Nresid*255)))
        print('{0}   {1}  {2}'.format(w,pix,Nresid))

    print('>>')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Counts')
    fig_main.show()
    return disp_func

def AddlinesbyInteractiveSelection(SpectrumY,disp_filename,comm_pipe=None):
    """ Adds more data points to disp_filename by interactively selecting lines """
    print('Enter more point in the file, or select lines '
          'interatively to obtain Zeroth order solution')
    ListOfLines = SelectLinesInteractively(np.arange(len(SpectrumY)),
                                               SpectrumY,
                                           ExtraUserInput='Wavelength',
                                           comm_pipe=comm_pipe)
    wavl,pix,sigma = [] , [], []
    for linefit in ListOfLines:
        wavl.append(float(linefit[1]))
        pix.append(linefit[0].mean_0.value)
        sigma.append(1)
    if wavl:
        writeto_dispersion_inputfile(disp_filename,wavl,pix,sigma)


def TryToFitNewLinesinSpectrum(SpectrumY,disp_filename,LineSigma=3):
    """ Try to fit gaussian at the line wavelgnths without pixel position in the disp_filename.
    And add the fitted pixels values to the file """
    # First read the existing data
    wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(disp_filename)
    # Dispertion function
    disp_func = FittedFunction(pixels=pixels_inp,
                               wavel=wavelengths_inp,
                               sigma=sigma_inp, method='p3')

    pix_fitted = []
    sigma_fitted = []
    for wavel in wavelengths_tofit:
        XPixarray = np.arange(len(SpectrumY))
        Xpos = NearestIndex(disp_func(XPixarray),wavel)
        Ampl_init = np.max(SpectrumY[Xpos-LineSigma:Xpos+LineSigma]) - \
                    np.min(SpectrumY[Xpos-LineSigma:Xpos+LineSigma])
        # Fit a Guassian line
        Model_fit = FitLineToData(XPixarray,SpectrumY,Xpos,Ampl_init,
                                  AmpisBkgSubtracted=False,Sigma = LineSigma)
        pix_fitted.append(Model_fit.mean_0.value)
        sigma_fitted.append(1) # to be updated later with actual error
    # Write the fitted pixel positions
    if pix_fitted:
        writeto_dispersion_inputfile(disp_filename,wavelengths_tofit,pix_fitted,sigma_fitted)


def DisplayDispersionSolution(SpectrumY,disp_filename,comm_pipe=None):
    """ This keeps a window open with the plot of the latest dispersion solution.
    IMP: Needs to be run as a Process, and comminicated via Pipe """
    fig_main = plt.figure()
    ax = fig_main.add_subplot(111)
    ax.set_title('Press r to refresh the plot.')
    # Define the function to run while key is pressed
    disp_func_holder = [update_main_figure(fig_main,SpectrumY,read_dispersion_inputfile(disp_filename)[1])]
    def on_key(event):
        if event.key == 'r' :
            disp_func_holder[0] = update_main_figure(fig_main,SpectrumY,read_dispersion_inputfile(disp_filename)[1])
    cid = fig_main.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    disp_func = disp_func_holder[0]
    if disp_func is not None:
        comm_pipe.send(tuple((disp_func(np.arange(len(SpectrumY))), disp_filename)))
    else:
        comm_pipe.send(tuple((None, disp_filename)))
    
    
def InteractiveDispersionSolution(SpectrumY,disp_filename=None):
    """ Identify lines in the spectrum and find dispersion solution"""
    # First read the disp file
    if disp_filename is None:
        disp_filename = str(uuid.uuid4())
        with open(disp_filename,'w') as f:
            f.write('# Wavel Pixel Sigma \n')
    print('Open the file://{0} in a non-blocking text editor to edit'.format(disp_filename))

    wavelengths_tofit, (wavelengths_inp,pixels_inp,sigma_inp) = read_dispersion_inputfile(disp_filename)

    if len(wavelengths_inp) < 2:
        parent_conn, child_conn = Pipe()
        fp = Process(target=AddlinesbyInteractiveSelection,args=(SpectrumY,disp_filename,child_conn))
        fp.start()
        Clientmsg = '>>'
        while Clientmsg:
            Clientmsg = parent_conn.recv_bytes()
            if Clientmsg:
                parent_conn.send_bytes(raw_input(Clientmsg))
        fp.join()
    # Plot main figure
    Mainparent_conn, Mainchild_conn = Pipe()
    Mp = Process(target=DisplayDispersionSolution,args=(SpectrumY,disp_filename,Mainchild_conn))
    Mp.start()
    DoneWithFitting = False
    print('Edit the file://{0} to add or remove points to fit'.format(disp_filename))
    print('Enter "done" to exit the interactive fitting')
    print('Enter f to start an interative fit window')
        
    while not DoneWithFitting:
        usr_input = raw_input('>> ')
        if usr_input == 'done':
            print('Exiting the fitting tool')
            DoneWithFitting = True
        elif usr_input == 'f':
            print('Starting Interactive fit Window')
            parent_conn, child_conn = Pipe()
            fp = Process(target=AddlinesbyInteractiveSelection,args=(SpectrumY,disp_filename,child_conn))
            fp.start()
            Clientmsg = '>>'
            while Clientmsg:
                Clientmsg = parent_conn.recv_bytes()
                if Clientmsg:
                    parent_conn.send_bytes(raw_input(Clientmsg))

            fp.join()
        elif usr_input == 'l':
            print('Fitting lines which do not have pixel coordinates in the text file.')
            TryToFitNewLinesinSpectrum(SpectrumY,disp_filename)

        
    wavel_soln_n_disp_filename =  Mainparent_conn.recv()
    Mp.join()
    # Return the dispertion wavelength solution and disp_filename 
    return wavel_soln_n_disp_filename
            
    

def parse_args():
    """ Parses the command line input arguments """
    parser = argparse.ArgumentParser(description="Interactive Wavelength Calibration Tool")
    parser.add_argument('SpectrumFluxFile', type=str,
                        help="File containing the Spectrum's Flux array")
    parser.add_argument('DispTableFile', type=str,
                        help="File containing the table of Wavelengths, Pixels, Error")
    parser.add_argument('OutputWavlFile', type=str,
                        help="Output filename to write calibrated Wavelength array")
    args = parser.parse_args()
    return args
    
def main():
    """ Standalone Interactive Line Identify Tool """
    args = parse_args()    
    SpectrumY = np.load(args.SpectrumFluxFile)
    disp_fname = args.DispTableFile
    Output_fname = args.OutputWavlFile
    wavl,df = InteractiveDispersionSolution(SpectrumY,disp_filename=disp_fname)
    print('Dispersion ASCII input file saved in file://{0}'.format(df))
    np.save(Output_fname,wavl)
    print('Wavelength solution saved in {0}'.format(Output_fname))

if __name__ == "__main__":
    main()
