import numpy as np

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
        s.inv_norm = (s.g0/s.gp)**(1.-s.q)*(1./(1.-s.q)+1./s.p)-1./(1.-s.q)
    
    def __call__(s,g):
        dndg = s.n/s.gp*(g/s.gp)**(-s.q)
        dndg[g>s.g0] = s.n/s.gp*(s.g0/s.gp)**(-s.q)*(g[g>s.g0]/s.g0)**(-s.p-1.)
        
        return (g>=s.gp)*(g<s.gmax)*dndg/s.inv_norm
