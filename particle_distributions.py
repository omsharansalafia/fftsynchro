import numpy as np
from scipy.special import gamma,gammaincc,kv

class superposition:
    
    def __init__(s,distrib_list):
        """
        Class for sum of two or more distributions.
        
        Init parameters:
        - distrib_list: list of instances of particle distribution classes.
        
        Methods:
        __call
        """

        s.distrib_list = distrib_list
        s.gmax = np.max([d.gmax for d in distrib_list])
            
    def __call__(s,g):
        return np.sum([d(g) for d in s.distrib_list],axis=0)
    
    def gmin(self):
        return np.min([[d.gmin() for d in self.distrib_list]])

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
        gmin(): minimum Lorentz factor
        """
        
        s.n = n
        s.gm = gm
        s.gmax = gmax
        s.p = p
        
    
    def __call__(s,g):
        
        return (g>=s.gm)*(g<s.gmax)*(s.p-1.)*s.n/s.gm*(g/s.gm)**(-s.p)
    
    def gmin(self):
        return self.gm

class powerlaw_with_lower_thermal_tail:
    
    def __init__(s,n,gm,p,gmax=1e8):
        """
        Class for power law particle Lorentz factor distribution, with a gamma^2 tail below gamma_m.
        This is for testing purposes. Note that the normalization is the same as for the powerlaw,
        hence the total density is actually larger than n.
        
        Init parameters:
        - n: total density (cm-3)
        - gm: minimum Lorentz factor
        - p: power law index, dn_dg propto g^-p
        
        Optional:
        - gmax: maximum Lorentz factor
        
        Methods:
        __call__(): return density within (g,g+dg)
        gmin(): minimum Lorentz factor
        """
        
        s.n = n
        s.gm = gm
        s.gmax = gmax
        s.p = p
        
    
    def __call__(s,g):
        
        return np.where(g>=s.gm,(g/s.gm)**(-s.p),(g/s.gm)**2.)*(g<s.gmax)*(s.p-1.)*s.n/s.gm # fix normalization
    
    def gmin(self):
        return 1.





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
        gmin(): minimum Lorentz factor
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
        gmin(): minimum Lorentz factor
        """
        
        s.n = n
        s.TH = TH
        if gmax==np.inf:
            s.gmax = 1e4*TH
        else:
            s.gmax = gmax
        
        s.norm = 2.*TH**2/kv(2.,1./TH)

    def gmin(self):
        return np.maximum(1.,self.TH/1e4)
    
    def __call__(s,g):
        dndg = s.n*s.norm*np.sqrt(1.-g**-2)*g**2*np.exp(-g/s.TH)/(2*s.TH**3)*(g<s.gmax)
        
        return dndg


class cooled_relativistic_maxwellian:

    def __init__(s,n,TH,gc,gmax=np.inf,R=2.8):
        """
        Class for broken power law particle Lorentz factor distribution,
        such as that resulting from synchrotron cooling.
        
        Init parameters:
        - n: total density (cm-3)
        - TH: temperature parameter
        - gc: cooling Lorentz factor
        - gmax: maximum injected Lorentz factor (default=inf)
        - R: connect cooled g^-2 law with super-exponential tail at R*TH
                
        Methods:
        __call__(): return density within (g,g+dg)
        gmin(): minimum Lorentz factor
        """
        
        s.n = n
        s.TH = TH
        s.gc = gc
        s.R = R
        if gmax==np.inf:
            s.gmax = 1e4*TH
        else:
            s.gmax = gmax
        
        g = np.geomspace(1,s.gmax,100)
        dndg = np.select([g<gc,g>(R*TH)],[g**2/(2*TH**3)*np.exp(-g/TH),2*g**2*np.exp(-g/TH)/(2*TH**3*(g/gc))],2*(g/(R*TH))**-2.*(R*TH)**2.*np.exp(-R)/(2*TH**3*(R*TH/gc)))
        
        s.norm = 1./np.trapz(dndg*g,np.log(g))

    def gmin(self):
        return 1.
    
    def __call__(s,g):
        dndg = s.n*s.norm*(g<(s.gmax))*np.select([g<s.gc,g>(s.R*s.TH)],[g**2/(2*s.TH**3)*np.exp(-g/s.TH),2*g**2*np.exp(-g/s.TH)/(2*s.TH**3*(g/s.gc))],2*(g/(s.R*s.TH))**-2.*(s.R*s.TH)**2.*np.exp(-s.R)/(2*s.TH**3*(s.R*s.TH/s.gc)))
        
        return dndg

class freeform:
    
    def __init__(s,g0,dndg0):
        """
        Class for free-form particle distribution, e.g. obtained from a numerical
        calculation. The class takes in the distribution evaluated on an array
        of Lorentz factors, and it interpolates over it.

        Init parameters:
        - g0: Lorentz factor array
        - dndg0: density of particle in Lorentz factor space evaluated over g0  
        """
        
        s.g0 = g0
        s.dndg0 = dndg0
        s.gmax = g0[-1]
        
    def gmin(self):
        return self.g0[0]
    
    def __call__(self,g):
        return np.interp(g,self.g0,self.dndg0,left=0,right=0)


class maxwellian_with_powerlaw_tail:
    
    def __init__(s,n,gth,gm,p,gmax=1e8):
        """
        Class for power law particle Lorentz factor distribution, with a relativistic maxwellian
        below gamma_m, and a power law above that.
        
        Init parameters:
        - n: total density (cm-3)
        - gth: "thermal" electron Lorentz factor
        - gm: minimum Lorentz factor
        - p: power law index, dn_dg propto g^-p
        
        Optional:
        - gmax: maximum Lorentz factor
        
        Methods:
        __call__(): return density within (g,g+dg)
        gmin(): minimum Lorentz factor
        """
        
        s.n = n
        s.gth = gth
        s.gm = gm
        s.bm = np.sqrt(1.-gm**-2)
        s.gmax = gmax
        s.p = p
        
        g = np.geomspace(1.,gmax,1000)
        dn_dg = np.where(g<gm,np.sqrt(1.-g**-2)*g**2*np.exp(-g/gth),s.bm*gm**2*np.exp(-gm/gth)*(g/gm)**-p)
        s.norm = np.trapz(dn_dg,g)

    
    def __call__(s,g):
        
        return s.n*np.where(g<s.gm,np.sqrt(1.-g**-2)*g**2*np.exp(-g/s.gth),s.bm*s.gm**2*np.exp(-s.gm/s.gth)*(g/s.gm)**-s.p)/s.norm
    
    def gmin(self):
        return 1.

