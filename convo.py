import numpy as np
from scipy.signal import convolve
from .synchro import normalization_Pnu
from .synchro import x0,F0
from .constants import *

def gxy(x):
    """
    Convenience function for convolution
    """
    return np.interp(np.exp(x),x0,F0)

def j_nu(nu,dn_dg,B,pitch,m=me,q=e,gmin=1.,gmax=1e8,res=0.1):
    """
    Specific synchrotron emissivity (power per unit frequency, per unit 
    volume, per unit steradian) of a population of particles whose density
    is distributed in Lorentz factor space as dn_dg, assuming a fixed
    pitch angle.
    
    Parameters:
    - nu: frequency in Hz
    - dn_dg: particle density distribution in Lorentz factor space. This
             must be a callable that gives the density of particles with
             Lorentz factors in the range (g,g+dg) in cm-3, evaluated at g.
    - B: the magnetic field strength in Gauss
    - pitch: the pitch angle in radians
    
    Keyword arguments:
    - m: particle mass
    - q: particle charge
    - gmin: minimum Lorentz factor to be considered
    - gmax: maximum Lorentz factor to be considered
    - res: resolution, understood as the amplitude of logarithmic 
           frequency steps in the grid
    """
    
    # define convenience function for convolution
    def fy(y):
        return 0.5*np.exp(y/2.)*dn_dg(np.exp(y/2.))
    
    # define arrays over which the convenience functions are
    # evaluated
    y = np.arange(2.*np.log(gmin),2.*np.log(gmax),res)
    x = np.arange(-18,15,res) # FIXME: hardcoded limits
    
    # this encodes the particle distribution, which represents the
    # signal in the convolution
    fyv = fy(y)
    
    # this encodes the single-particle spectrum, which represents the
    # window in the convolution
    gxyv = gxy(x)
    
    # do the convolution (scipy automatically switches to FFT if convenient)
    hx = convolve(fyv,gxyv,mode='full')
    
    # reconstruct the frequency range
    numin = gmin**2*1.5*np.sin(pitch)*q*B/(2*np.pi*m*c)*np.exp(x[0]) 
    numax = gmax**2*1.5*np.sin(pitch)*q*B/(2*np.pi*m*c)*np.exp(x[-1])
    nu0 = np.geomspace(numin,numax,len(hx))
    
    # compute normalization
    K = normalization_Pnu(B,pitch,m,q)/(4.*np.pi)
    
    return np.interp(nu,nu0,K*hx)

 

    

    
