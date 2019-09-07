#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:35:36 2019

@author: bradc
"""

import math
import functools as ft
import numpy as np
from numba import njit, vectorize, types
from numba.extending import overload
from scipy.special import eval_jacobi, gamma
from scipy.misc import derivative
from scipy.integrate import quadrature, romberg, romb

"""
# NOTES ON NUMBA
# When using numba to increase performance, only functions called with
# nopython=True or @njit will experience significant speed-up

# Functions that may need to take arrays of values should be called with
# the @vectorize wrapper
"""

#################################
# Initial imputs
#################################

d = 4
m = 0
Dpl = d/2 + 0.5 * np.sqrt(d * d + 4.0 * m * m)
Dmin = d/2 - 0.5 * np.sqrt(d * d + 4.0 * m * m)
NMAX = 100

#################################
# Helpful secondary functions
#################################

@njit
def mu(x):
    return (np.tan(x)) ** (d-1)

# Evaluate the derivative analytically
@njit
def nu(x):
    return (np.cos(x)) ** 2 * (np.tan(x)) ** (2 - d)

@njit
def omega(i):
    return Dpl + 2 * i

# Call functions once to compile
mu(np.ones((5,)))
nu(np.ones((5,)))
omega(0)
print("Done compiling secondary functions")

#################################
# Normalizable mode basis functions
#################################

# Numba doesn't like functions from packages outside of numpy,
# so overload the call to scipy special functions to make it
# compatable with @njit

    
@overload(eval_jacobi)
def jacobi_value(n, a, b, x):
    if (isinstance(n, types.int32) and isinstance(a, types.float64) and\
        isinstance(b, types.float64) and isinstance(x, types.float64)):
        outval = eval_jacobi(n, a, b, x)
        def my_jacobi(n, a, b, x):
            return outval
        return my_jacobi
    

@njit
def K(i):
    return 2. * np.sqrt((i + Dpl / 2) * math.gamma(i + 1) * math.gamma(Dpl + i) /\
                        (math.gamma(i + d/2) * math.gamma(i + Dpl + 1 - d/2.)))


@njit('f8(f8, f8)')
def e(x, i):
    return K(i) * (np.cos(x)) ** Dpl * eval_jacobi(i, d/2 - 1, Dpl - d/2,
            np.cos(2 * x))

# This derivative is also evaluated analytically
@njit('f8(f8, f8)')
def ep(x, j):
    return -1. * np.tan(x) * (np.cos(x)) ** Dpl * (2 * (Dpl + j) *\
            np.cos(x) ** 2 * eval_jacobi(j - 1, d/2, Dpl + 1 - d/2, np.cos(2 * x))\
            + Dpl * eval_jacobi(j, d/2 - 1, Dpl - d/2, np.cos(2 * x)))
    
# Compile
K(0.)
e(0.,0.)
ep(0.,0.)
print("Compiled basis functions")
        
#################################
# Integrands
#################################
    
# Remove the need to include nu(x) in the integrals by using 
# the identity mu(x)nu(x) = sin(x)cos(x)
def V_integrand(x, i, j, k, l):
    return mu(x) * np.sin(x) * np.cos(x) * e(x, i) * e(x, j) \
        * ep(x, k) * e(x, l) * (1. / np.cos(x)) ** 2
        
def X_integrand(x,i,j,k,l):
    return mu(x) * np.sin(x) * np.cos(x) * ep(x, i) * e(x, j) * e(x, k) * e(x, l)

def Y_integrand(x,i,j,k,l):
    return mu(x) * np.sin(x) * np.cos(x) * ep(x, i) * ep(x, j) * e(x, k) * ep(x, l)

#################################        
# Integral values
#################################

def X(i,j,k,l):
    """
    # Output is (value, error)
    result = quadrature(X_integrand, 0, np.pi/2, args=(i,j,k,l), rtol=1.e-8,
                  maxiter=500)
    """
    result = romberg(X_integrand, 0, np.pi/2, args=(i,j,k,l), show=False,
                         vec_func=True, divmax=20)
    return result

def Y(i,j,k,l):
    """
    # Output is (value, error)
    result = quad(Y_integrand, 0, np.pi/2, args=(i,j,k,l), rtol=1.e-8,
                  maxiter=500)
    """
    result = romberg(Y_integrand, 0, np.pi/2, args=(i,j,k,l), show=False,
                         vec_func=True, divmax=20)
    return result

def V(i,j,k,l):
    """
    # Output is (value, error)
    result = quad(V_integrand, 0, np.pi/2, args=(i,j,k,l), rtol=1.e-8,
                  maxiter=500)
    """
    result = romberg(V_integrand, 0, np.pi/2, args=(i,j,k,l), show=False,
                         vec_func=True, divmax=20)
    return result

# Use integration by parts to write H_ijkl in terms of other,
# more well-behaved, integrals
def H(i, j, k, l):
    return omega(i) ** 2 * X(k,i,j,l) + omega(k) ** 2 * X(i,j,k,l) \
        - Y(i,j,l,k) - Y(l,k,j,i) - m ** 2 * (V(i,j,k,l) + V(k,j,i,l))

# Combinations of integrals
def Zpl(i,j,k,l):
    return omega(i) * omega(j) * (X(k,l,i,j) - X(l,k,i,j)) \
        + (Y(i,j,l,k) - Y(i,j,k,l))
        
def Zmin(i,j,k,l):
    return omega(i) * omega(j) * (X(k,l,i,j) - X(l,k,i,j)) \
        - (Y(i,j,l,k) - Y(i,j,k,l))
        
#################################
# Source term for +++ resonance channel
#################################
        
def Omega(i,j,k,l):
    return (-1./12) * H(i,j,k,l) * omega(j) * (omega(i) + omega(k) +        
        2 * omega(j)) / ((omega(i) + omega(j)) * (omega(j) + omega(k))) \
        - (1./12) * H(i,k,j,l) * omega(k) * (omega(i) + omega(j) + 
        2 * omega(k)) / ((omega(i) + omega(k)) * (omega(j) + omega(k))) \
        - (1./12) * H(j,i,k,l) * omega(i) * (omega(j) + omega(k) + 
        2 * omega(i)) / ((omega(i) + omega(j)) * (omega(i) + omega(k))) \
        - (1./12) * m ** 2 * V(i,j,k,l) * (1 + omega(j) / (omega(j) + omega(k))
        + omega(i) / (omega(i) + omega(k))) \
        - (1./12) * m ** 2 * V(j,k,i,l) * (1 + omega(j) / (omega(i) + omega(j))
        + omega(k) / (omega(i) + omega(k))) \
        - (1./12) * m ** 2 * V(k,i,j,l) * (1 + omega(i) / (omega(i) + omega(j))
        + omega(k) / (omega(j) + omega(k))) \
        + (1./6) * X(j,k,i,l) * omega(i) * omega(k) * (1 + omega(i) / (omega(j)
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        + (1./6) * X(k,i,j,l) * omega(i) * omega(j) * (1 + omega(i) / (omega(j) 
        + omega(k)) + omega(j) / (omega(i) + omega(k))) \
        + (1./6) * X(i,j,k,l) * omega(j) * omega(k) * (1 + omega(j) / (omega(i) 
        + omega(k)) + omega(k) / (omega(i) + omega(j))) \
        - (1./12) * Zmin(i,j,k,l) * omega(k) / (omega(i) + omega(j)) \
        - (1./12) * Zmin(i,k,j,l) * omega(j) / (omega(i) + omega(k)) \
        - (1./12) * Zmin(j,k,i,l) * omega(i) / (omega(j) + omega(k))
        
#################################
# Source term for the +-- resonance channel
#################################
        
def Gamma(i,j,k,l):
    return (1./4) * H(i,j,k,l) * omega(j) * (omega(k) - omega(i)
            + 2 * omega (j)) / ((omega(i) - omega(j)) * (omega(j) + omega(k)))\
            + (1./4) * H(j,k,i,l) * omega(k) * (omega(j) - omega(i)
            + 2 * omega(k)) / ((omega(i) - omega(k)) * (omega(j) + omega(k)))\
            + (1./4) * H(k,i,j,l) * omega(i) * (omega(j) + omega(k)
            - 2 * omega(i)) / ((omega(i) - omega(j)) * (omega(i) - omega(k)))\
            - (1./2) * omega(j) * omega(k) * X(i,j,k,l) * (omega(k) / (omega(i) \
            - omega(j)) + omega(j) / (omega(i) - omega(k)) - 1.)\
            + (1./2) * omega(i) * omega(k) * X(j,k,i,l) * (omega(k) / (omega(i) \
            - omega(j)) + omega(i) / (omega(j) + omega(k)) - 1.)\
            + (1./2) * omega(i) * omega(j) * X(k,i,j,l) * (omega(j) / (omega(i) \
            - omega(k)) + omega(i) / (omega(j) + omega(k)) - 1.)\
            + (1./4) * m ** 2 * V(j,k,i,l) * (omega(j) / (omega(i) - omega(j))\
            + omega(k) / (omega(i) - omega(k)) - 1.)\
            - (1./4) * m ** 2 * V(k,i,j,l) * (omega(i) / (omega(i) - omega(j))\
            + omega(k) / (omega(j) + omega(k)) + 1.)\
            - (1./4) * m ** 2 * V(i,j,k,l) * (omega(i) / (omega(i) - omega(k))\
            + omega(j) / (omega(j) + omega(k)) + 1.)\
            + (1./4) * Zmin(k,j,i,l) * omega(i) / (omega(j) + omega(k))\
            - (1./4) * Zpl(i,j,k,l) * omega(k) / (omega(i) - omega(j))\
            - (1./4) * Zpl(j,k,i,l) * omega(j) / (omega(i) - omega(k))
               
        
##################################
##################################
        
  
LMAX = 1
"""  
print("\nCalculating +++ resonance terms...")
for l in range(LMAX):
    outval = 0.
    for i in range(LMAX):
        for j in range(LMAX):
            if (l - i - j - Dpl >= 0):
                #print("Omega =", Omega(i,j,l-i-j-Dpl,l), "\n")
                outval += Omega(i, j, l - i - j - Dpl, l)
            else:
               # print("Prohibited value: (%d,%d,%d,%d)\n" 
               #       % (i,j,l-i-j-Dpl,l))
               pass
    print("S(%d) = %e" % (l, outval))
"""

print("\nCalculating +-- resonance terms...")
for l in range(LMAX):
    outval = 0.
    for j in range(LMAX):
        for k in range(LMAX):
            #print("Evaluating Gamma(%d,%d,%d,%d)" % (j + k + l + Dpl, j, k, l))
            outval += Gamma(j + k + l + Dpl, j, k, l)
            
    print("S(%d) = %e" % (l, outval))
    
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
