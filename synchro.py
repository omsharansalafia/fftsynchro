import numpy as np
from scipy.special import kv
from scipy.special import gamma
from scipy.integrate import cumtrapz
from .constants import *

# pre-compute the single-particle spectrum
x0 = np.logspace(-10,3,10000) # this is nu/nu_c, FIXME: the resolution and grid extent are hardcoded

# compute the cumulative integral of the K_5/3 modified Bessel function of
# the second kind, represented by kv(5./3.,x) in scipy, using the trapezoidal
# rule 
G0 = cumtrapz(kv(5./3.,x0),x0,initial=0.)
G0 = G0[-1]-G0

# this is the synchrotron spectral shape F(nu/nu_c)
F0 = x0*G0
# fix bad points along the exponential decay leveraging on the analytic
# asymptotic form
F0[x0>10.]=(np.pi/2.)**0.5*(x0[x0>10.])**0.5*np.exp(-x0[x0>10.])


def nu_cyc(B,m=me,q=e):
    """
    Cyclotron frequency
    """
    return q*B/(2*np.pi*m*c)

def normalization_Pnu(B,pitch,m=me,q=e):
    """
    Normalization of the synchrotron single particle power spectral density
    """
    return 3**0.5*q**3*B*np.sin(pitch)/(m*c**2)

def P_nu(nu,g,B,pitch,m=me,q=e):
    """
    Power spectral density, dP/dnu, emitted by a single particle at frequency
    nu.
    
    Parameters:
    - nu: frequency in Hz
    - g: Lorentz factor
    - B: magnetic field in Gauss
    - pitch: pitch angle in radians
    
    Keyword arguments:
    - m: particle mass
    - q: particle charge
    
    """
    # cyclotron frequency
    nucyc = nu_cyc(B,m,q)
    
    # synchrotron characteristic frequency
    nuc = 1.5*g**2*nucyc*np.sin(pitch)
    
    # interpolate over pre-computed single-particle spectrum
    x = nu/nuc
    F = np.interp(x,x0,F0)
    
    # normalization
    K = normalization_Pnu(B,pitch,m,q)
    
    return K*F
