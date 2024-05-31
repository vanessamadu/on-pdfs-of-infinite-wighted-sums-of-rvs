import numpy as np
from scipy.special import hyp1f1
import scipy.integrate as integrate

#--------------- STANDARD SINGLE RV CHARACTERISTIC FUNCTIONS --------------#
def beta_char_func(t,beta_params):
    '''
    description:    characteristic function of a symmetric beta distributed random
                    variable on the interval [-B,B] with parameter alpha.
    params:
    t:              real number > 0
    beta_params:    [alpha,B], real numbers > 0       
    '''
    # unpack
    alpha, B = beta_params

    return np.real_if_close((2*B)**(2*alpha-1)*np.exp(-1j*B*t)*hyp1f1(alpha,2*alpha,2j*B*t),tol=10E-10)

def uniform_char_func(t,uniform_params):
    '''
    description:    characteristic function of a uniform distributed random variable
                    on the interval [-B,B]
    params:
    t:              real number > 0
    uniform_params: B, real number > 0
    '''
    # unpack
    B = uniform_params

    return np.sinc(B*t/np.pi)

#-------------------- INFINITE WEIGHTED SUM CHARACTERISTIC FUNCTIONS ----------------#

def infinite_weighted_sum_RV_char_func(t,S,c_s_func,c_s_params,cf_params,cf):
    '''
    description:    finite approximation of the characteristic function of the infinite 
                    weighted sum of IID random variables
    params:
    t:              real number > 0
    S:              upper limit of product approximation. integer > 0
    c_s_func:       defines coefficients as a function of s, real function
    c_s_params:     real number(s)
    cf_params:      real numbers > 0  
    cf:             characteristic function of the RV being summed
    '''
    return np.prod([cf(c_s_func(c_s_params,s)*t,cf_params) for s in range(S)])

#------------------- WEIGHTING SCHEMES ------------------#
# geometric

def c_s_geom(theta,s):
    # -1 < theta < 1
    return theta**s
def L_geom(params):
    # -1 < theta < 1
    #unpack 
    B,theta = params
    return B/(1-np.abs(theta))
#----------------- Evaluating M ----------------#
def partial_alternating_sum(L,S,c_s_func,c_s_params,cf_params,cf,k):
    '''
    description:    finite approximation of alternating series denominator of M.

    params:
    L_func:         defines bounds of the infinite sum of random variables, real function
    L_params:       real number(s)
    S:              upper limit of product approximation. integer > 0
    c_s_func:       defines coefficients as a function of s, real function
    c_s_params:     real number(s)
    cf_params:      real numbers > 0   
    cf:             characteristic function of the RV being summed
    k:              upper limit of sum approximation. integer > 0

    '''
    return np.sum([(-1)**(n-1)*infinite_weighted_sum_RV_char_func(n*np.pi/L,S,c_s_func,c_s_params,cf_params,cf) 
                   for n in range(1,k+1)])

def partial_sum(L,S,c_s_func,c_s_params,cf_params,cf,k):
    '''
    description:    finite approximation of alternating series denominator of M.

    params:
    L_func:         defines bounds of the infinite sum of random variables, real function
    L_params:       real number(s)
    S:              upper limit of product approximation. integer > 0
    c_s_func:       defines coefficients as a function of s, real function
    c_s_params:     real number(s)
    cf_params:      real numbers > 0   
    cf:             characteristic function of the RV being summed
    k:              upper limit of sum approximation. integer > 0

    '''
    return np.sum([infinite_weighted_sum_RV_char_func(n*np.pi/L,S,c_s_func,c_s_params,cf_params,cf) 
                   for n in range(1,k+1)])

def partial_sum_remainder(sum_func,L,S,c_s_func,c_s_params,cf_params,cf,k):

    return sum_func(L,S,c_s_func,c_s_params,
                                   cf_params,cf,k) - sum_func(L,S,c_s_func,
                                                                             c_s_params,cf_params,cf,k-1)

def approx_alternating_series(L,S,c_s_func,c_s_params,cf_params,cf,tol,p):
    k = 1
    remainders = []
    partial_sums = []
    
    # start up:
    while k <= p:
        remainders.append(partial_sum_remainder(partial_alternating_sum,L,S,c_s_func,c_s_params,cf_params,cf,k))
        partial_sums.append(partial_alternating_sum(L,S,c_s_func,c_s_params,
                                   cf_params,cf,k))
        k+=1
    while (np.abs(remainders[::-1][:p]) > tol).any():
        remainders.append(partial_sum_remainder(partial_alternating_sum,L,S,c_s_func,c_s_params,cf_params,cf,k))
        partial_sums.append(partial_alternating_sum(L,S,c_s_func,c_s_params,
                                   cf_params,cf,k))
        k +=1
    return k, remainders, partial_sums, np.real_if_close(partial_alternating_sum(L,S,c_s_func,c_s_params,cf_params,cf,k),tol=10E-8)

def approx_series(L,S,c_s_func,c_s_params,cf_params,cf,tol,p):
    k = 1
    remainders = []
    partial_sums = []
    
    # start up:
    while k <= p:
        remainders.append(partial_sum_remainder(partial_sum,L,S,c_s_func,c_s_params,cf_params,cf,k))
        partial_sums.append(partial_sum(L,S,c_s_func,c_s_params,
                                   cf_params,cf,k))
        k+=1
    while (np.abs(remainders[::-1][:p]) > tol).any():
        remainders.append(partial_sum_remainder(partial_sum,L,S,c_s_func,c_s_params,cf_params,cf,k))
        partial_sums.append(partial_sum(L,S,c_s_func,c_s_params,
                                   cf_params,cf,k))
        k +=1
    return k, remainders, partial_sums, np.real_if_close(partial_sum(L,S,c_s_func,c_s_params,cf_params,cf,k),tol=10E-8)

def M(L,S,c_s_func,c_s_params,cf_params,cf,tol,p):
    vals =approx_alternating_series(L,S,c_s_func,c_s_params,cf_params,cf,tol,p)
    return np.real_if_close(np.pi/(2*vals[3]),tol=10E-8)

#----------------- Threshold of L --------------#
def L_threshold_eqn(L,S,c_s_func,c_s_params,cf_params,cf,tol,p):
    return np.real_if_close(2*L - 1 - approx_series(L,S,c_s_func, c_s_params,cf_params,cf,tol,p)[3]/approx_alternating_series(L,S,c_s_func, c_s_params,cf_params,cf,tol,p)[3],tol=10E-8)

#----------------- Probability Density Function -----------------#

def PDF(x,M_val,N,L_func,L_params,S,c_s_func,c_s_params,cf_params,cf):
    L = L_func(L_params)
    if np.abs(x)<=L:
        pdf = np.sum([infinite_weighted_sum_RV_char_func(n*np.pi/L,S,c_s_func,c_s_params,cf_params,cf)*
            np.cos(n*np.pi*x/L) for n in range(1,N+1)])
        pdf*= M_val/(np.pi*L)
        pdf += 1/(2*L)
        return np.real_if_close(pdf,tol=10E-8)
    else:
        return 0.0
        
    

def check_integral(M_val,N,L_func,L_params,S,c_s_func,c_s_params,cf_params,cf):
    L = L_func(L_params)
    result = integrate.quad(lambda x: np.real_if_close(PDF(x,M_val,N,L_func,L_params,S,c_s_func,c_s_params,cf_params,cf),tol=10E-8),-L,L)
    return result