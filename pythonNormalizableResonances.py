#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:35:36 2019

@author: bradc
"""

import numpy as np
from numba import jit
from scipy.special import eval_jacobi, gamma
from scipy.misc import derivative as D
from scipy.integrate import quad
import functools as ft


# Initial imputs
d = 4.
m = 0.
Dpl = d/2 + 0.5 * np.sqrt(d * d + 4.0 * m * m)
Dmin = d/2 - 0.5 * np.sqrt(d * d + 4.0 * m * m)

# Helpful secondary functions
@jit
def mu(x):
    return (np.tan(x)) ** (d-1)

@jit
def nu(x):
    return (d-1) / D(mu, x, dx=1.e-8)

@jit
def omega(i):
    return Dpl + 2 * i

# Normalizable mode basis functions
@jit
def k(i):
    return 2. * np.sqrt((i + Dpl / 2) * gamma(i + 1) * gamma(Dpl + i) / 
                        ( gamma(i + d/2) * gamma(i + Dpl + 1 - d/2.)))

@jit
def e(x, i):
    return k(i) * (np.cos(x)) ** Dpl * eval_jacobi(i, d/2 - 1, 
            Dpl - d/2, np.cos(2 * x))    

@jit
def ep(x, j):
    return D(ft.partial(e, i=j), x, dx=1.e-8)   


# Integrands    
@jit
def V_integrand(x,i,j,k,l):
    return nu(x) * (mu(x)) ** 2 * e(x, i) * e(x, j) \
        * ep(x, k) * e(x, l) * (1. / np.cos(x)) ** 2
        
@jit
def X_integrand(x,i,j,k,l):
    return nu(x) * (mu(x)) ** 2 * ep(x, i) * e(x, j) * e(x, k) * e(x, l)

@jit
def Y_integrand(x,i,j,k,l):
    return nu(x) * (mu(x)) ** 2 * ep(x, i) * ep(x, j) * e(x, k) * e(x, l)

        
# Integral values
@jit
def X(i,j,k,l):
    # Output is (value, error)
    return quad(X_integrand, 0, np.pi/2, args=(i,j,k,l), limit=100)[0]

@jit
def Y(i,j,k,l):
    # Output is (value, error)
    return quad(Y_integrand, 0, np.pi/2, args=(i,j,k,l), limit=100)[0]

@jit
def V(i,j,k,l):
    # Output is (value, error)
    return quad(V_integrand, 0, np.pi/2, args=(i,j,k,l), limit=100)[0]


# Use integration by parts to write H_ijkl in terms of other,
# more well-behaved, integrals
@jit
def H(i,j,k,l):
    return omega(i) ** 2 * X(k,i,j,l) + omega(k) ** 2 * X(i,j,k,l) \
        - Y(i,j,l,k) - Y(l,k,j,i) - m ** 2 * (V(i,j,k,l) + V(k,j,i,l))

# Combinations of integrals
@jit
def Zpl(i,j,k,l):
    return omega(i) * omega(j) * (X(k,l,i,j) - X(l,k,i,j)) \
        + (Y(i,j,l,k) - Y(i,j,k,l))
        
@jit
def Zmin(i,j,k,l):
    return omega(i) * omega(j) * (X(k,l,i,j) - X(l,k,i,j)) \
        - (Y(i,j,l,k) - Y(i,j,k,l))
        

# Source term for +++ resonance channel
@jit
def Omega(i,j,k,l):
    return - 1./12 * H(i,j,k,l) * omega(j) * (omega(i) + omega(k) +        
        2 * omega(j)) / ((omega(i) + omega(j)) * (omega(j) + omega(k))) \
        -1./12 * H(i,k,j,l) * omega(k) * (omega(i) + omega(j) + 
        2 * omega(k)) / ((omega(i) + omega(k)) * (omega(j) + omega(k))) \
        - 1./12 * H(j,i,k,l) * omega(i) * (omega(j) + omega(k) + 
        2 * omega(i)) / ((omega(i) + omega(j)) * (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(i,j,k,l) * (1 + omega(j) / (omega(j) + omega(k))
        + omega(i) / (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(j,k,i,l) * (1 + omega(j) / (omega(i) + omega(j))
        + omega(k) / (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(k,i,j,l) * (1 + omega(i) / (omega(i) + omega(j))
        + omega(k) / (omega(j) + omega(k))) \
        + 1./6 * X(j,k,i,l) * omega(i) * omega(k) * (1 + omega(i) / (omega(j)
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        + 1./6 * X(k,i,j,l) * omega(i) * omega(j) * (1 + omega(i) / (omega(j) 
        + omega(k)) + omega(j) / (omega(i) + omega(k))) \
        + 1./6 * X(i,j,k,l) * omega(j) * omega(k) * (1 + omega(j) / (omega(i) 
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        - 1./12 * Zmin(i,j,k,l) * omega(k) / (omega(i) + omega(j)) \
        - 1./12 * Zmin(i,k,j,l) * omega(j) / (omega(i) + omega(k)) \
        - 1./12 * Zmin(j,k,i,l) * omega(i) / (omega(j) + omega(k))



"""
- 1./12 * H(i,j,k,l) * omega(j) * (omega(i) + omega(k) +        
        2 * omega(j)) / ((omega(i) + omega(j)) * (omega(j) + omega(k))) \
        -1./12 * H(i,k,j,l) * omega(k) * (omega(i) + omega(j) + 
        2 * omega(k)) / ((omega(i) + omega(k)) * (omega(j) + omega(k))) \
        - 1./12 * H(j,i,k,l) * omega(i) * (omega(j) + omega(k) + 
        2 * omega(i)) / ((omega(i) + omega(j)) * (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(i,j,k,l) * (1 + omega(j) / (omega(j) + omega(k))
        + omega(i) / (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(j,k,i,l) * (1 + omega(j) / (omega(i) + omega(j))
        + omega(k) / (omega(i) + omega(k))) \
        - 1./12 * m ** 2 * V(k,i,j,l) * (1 + omega(i) / (omega(i) + omega(j))
        + omega(k) / (omega(j) + omega(k))) \
        + 1./6 * X(j,k,i,l) * omega(i) * omega(k) * (1 + omega(i) / (omega(j)
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        + 1./6 * X(k,i,j,l) * omega(i) * omega(j) * (1 + omega(i) / (omega(j) 
        + omega(k)) + omega(j) / (omega(i) + omega(k))) \
        + 1./6 * X(i,j,k,l) * omega(j) * omega(k) * (1 + omega(j) / (omega(i) 
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        - 1./12 * Zmin(i,j,k,l) * omega(k) / (omega(i) + omega(j)) \
        - 1./12 * Zmin(i,k,j,l) * omega(j) / (omega(i) + omega(k)) \
        - 1./12 * Zmin(j,k,i,l) * omega(i) / (omega(j) + omega(k))
"""



        
##################################
##################################
        
    
Lmax = 5
print("Calculating +++ resonance terms...")
for i in range(Lmax):
    for j in range(Lmax):
        for l in range(Lmax):
            if (l - i - j - Dpl >= 0):
                print("Omega =", Omega(i,j,l-i-j-Dpl,l), "\n")
            else:
                print("Prohibited value: (%d,%d,%d,%d)\n" 
                      % (i,j,l-i-j-Dpl,l))
                
print("Done!\n")
    
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    