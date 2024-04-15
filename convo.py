import numpy as np
from scipy.signal import convolve
from scipy.special import kv
from .synchro import normalization_Pnu
from .synchro import x0,F0
from .synchro import P_nu,P_nu_isopitch
from .constants import *

def gxy(x):
    """
    Convenience function for convolution
    """
    return np.interp(np.exp(x),x0,F0)

def gxy_iso(x):
    """
    Convenience function for convolution, isotropic pitch angle case
    """
    y = np.exp(x)
    K43 = kv(4./3.,y)
    K13 = kv(1./3.,y)

    return y**2*(K43*K13-0.6*y*(K43**2-K13**2))

def j_nu(nu,dn_dg,B,pitch='iso',m=me,q=e,gmin=1.,gmax=1e8,res=0.1,mode='fft'):
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
    
    Keyword arguments:
    - pitch: either the pitch angle in radians (for fixed pitch angle), or
             'iso', if the pitch angle distribution is assumed isotropic
    - m: particle mass
    - q: particle charge
    - gmin: minimum Lorentz factor to be considered
    - gmax: maximum Lorentz factor to be considered
    - res: resolution, understood as the amplitude of logarithmic 
           frequency steps in the grid
    - mode: either 'fft' or 'direct'
    """
    
    if mode=='fft':
        
        # define convenience function for convolution
        def fy(y):
            return 0.5*np.exp(y/2.)*dn_dg(np.exp(y/2.))
        
        # define arrays over which the convenience functions are
        # evaluated
        y = np.arange(2.*np.log(gmin),2.*np.log(gmax),res)
        x = np.arange(-25,15,res) # FIXME: hardcoded limits
        
        # this encodes the particle distribution, which represents the
        # signal in the convolution
        fyv = fy(y)
        
        # this encodes the single-particle spectrum, which represents the
        # window in the convolution
        if pitch=='iso':
            gxyv = gxy_iso(x)
        else:    
            gxyv = gxy(x)
        
        # do the convolution (scipy automatically switches to FFT if convenient)
        hx = convolve(fyv,gxyv,mode='full')*res
        
        # reconstruct the frequency range
        if pitch=='iso':
            numin = gmin**2*2.*q*B/(2*np.pi*m*c)*np.exp(x[0]) 
            numax = gmax**2*4.*q*B/(2*np.pi*m*c)*np.exp(x[-1])
        else:
            sinpitch = 1.5*np.sin(pitch)
            numin = gmin**2*sinpitch*q*B/(2*np.pi*m*c)*np.exp(x[0]) 
            numax = gmax**2*sinpitch*q*B/(2*np.pi*m*c)*np.exp(x[-1])

            
        nu0 = np.geomspace(numin,numax,len(hx))
        
        # compute normalization
        if pitch=='iso':        
            K = 2*normalization_Pnu(B,np.pi/2.,m,q)/(4.*np.pi)
        else:
            K = normalization_Pnu(B,pitch,m,q)/(4.*np.pi)

        return np.interp(nu,nu0,K*hx)

    else:
        
        g = np.geomspace(dn_dg.gmin(),dn_dg.gmax,int(res**-1))[:,None]
        
        if pitch=='iso':
            Pnu = P_nu_isopitch(nu,g,B,m,q)
        else:
            Pnu = P_nu(nu,g,B,pitch,m,q)
            
        ng = dn_dg(g)
        
        jnu = np.trapz(ng*Pnu,g,axis=0)/(4*np.pi)
        
        return jnu
        
                
    
 

    

    
