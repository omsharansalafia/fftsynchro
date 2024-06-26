# fftsynchro
Python package to compute the synchrotron spectrum of a population of particles using the fast fourier transform

## Dependencies

This package is written in python 3.x and it depends on numpy and scipy


## Installation

You can just clone this package in a directory and then add that directory
to your PYTHONPATH environment variable. You will then be able to import
the package by issuing "import fftsynchro"

## Usage

The main function exposed to the user is fftsynchro.j_nu, which computes
the synchrotron emissivity from a particle population assuming a fixed
pitch angle. One of the arguments is a callable (i.e. another function)
that gives the density of particles with Lorentz factors within (g,g+dg):
for that purpose, the classes in fftsynchro.particle_distributions can
be used, or a user-defined function or class can be provided.

### Example code

```python
import numpy as np
import matplotlib.pyplot as plt
import fftsynchro

# Magnetic field
B = 1. #Gauss

# Particle distribution
n = 1.  #cm-3
gm = 1e4  #Minimum Lorentz factor
p = 2.2  #Power law index

dndg = fftsynchro.particle_distributions.powerlaw(n,gm,p)  #Class that holds the particle dist

# compute emissivity
nu = np.logspace(8,24,100)
jnu = fftsynchro.j_nu(nu,dndg,B,pitch='iso')  #compute emissivity assuming an isotropic pitch angle distribution

# plot it
plt.loglog(nu,jnu/jnu.max())

plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel(r'$j_\nu/j_{\nu,\mathrm{max}}$')

plt.xlim([1e8,1e24])
plt.ylim([1e-10,3])

plt.show()

```
