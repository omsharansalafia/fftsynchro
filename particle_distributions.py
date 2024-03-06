import numpy as np
from scipy.special import gamma,gammaincc


class powerlaw:
    
    def __init__(s,n,gm,p,gmax=1e8):
        """
        Class for power law particle Lorentz factor distribution.
        
        Init parameters:
        - n: total density (cm-3)
        - gm: minimum Lorentz factor
        - p: power law index, dn_dg propto g^-p
        
        Optional:
        - gmax: maximum Lorentz factor
        
        Methods:
        __call__(): return density within (g,g+dg)
        """
        
        s.n = n
        s.gm = gm
        s.gmax = gmax
        s.p = p
        
    
    def __call__(s,g):
        
        return (g>=s.gm)*(g<s.gmax)*(s.p-1.)*s.n/s.gm*(g/s.gm)**(-s.p)
    
    def gmin(self):
        return self.gm


class cooled_powerlaw:
    
    def __init__(s,n,gm,gc,p,gmax=1e8):
        """
        Class for broken power law particle Lorentz factor distribution,
        such as that resulting from synchrotron cooling.
        
        Init parameters:
        - n: total density (cm-3)
        - gm: minimum injected Lorentz factor
        - gc: Lorentz factor above which cooling is efficient
        - p: injected power law index, dn_dg propto g^-p
        
        Optional:
        - gmax: maximum Lorentz factor
        
        Methods:
        __call__(): return density within (g,g+dg)
        """
        
        s.n = n
        s.gm = gm
        s.gmax = gmax
        s.p = p
        s.gp = np.minimum(gm,gc)
        s.g0 = np.maximum(gm,gc)
        s.q = np.where(gm<=gc,p,2.)
        x0 = s.g0/s.gp
        s.inv_norm = ((x0**(1-s.q)-1.)/(1-s.q)+x0**(1-s.q)/s.p*(1.-(s.gmax/s.g0)**(-s.p)))
    
    def gmin(self):
        return self.gp
    
    def __call__(s,g):
        dndg = s.n/s.gp*np.where(g<=s.g0,(g/s.gp)**(-s.q),(s.g0/s.gp)**(-s.q)*(g/s.g0)**(-s.p-1.))
        
        return (g>=s.gp)*(g<s.gmax)*dndg/s.inv_norm

class relativistic_maxwellian:

    def __init__(s,n,TH,gmax=np.inf):
        """
        Class for broken power law particle Lorentz factor distribution,
        such as that resulting from synchrotron cooling.
        
        Init parameters:
        - n: total density (cm-3)
        - TH: temperature parameter
                
        Methods:
        __call__(): return density within (g,g+dg)
        """
        
        s.n = n
        s.TH = TH
        if gmax==np.inf:
            s.gmax = 1e4*TH
        else:
            s.gmax = gmax

    def gmin(self):
        return np.maximum(1.,self.TH/1e4)
    
    def __call__(s,g):
        dndg = 0.5*s.n*g**2*np.exp(-g/s.TH)/(gamma(2)*gammaincc(2,s.TH**-1)*s.TH**3)*(g<s.gmax)
        
        return dndg
    
