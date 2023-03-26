### Author: Frank Salvador Ygnacio Rosas
### Email: fsyrosas@outlook.com
### Github: @frankstack

import re
import itertools
import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank

######################################## Chap 2 - NLA Book: Forward and Backward Sub. for Triangular L/U matrices.
def LU_fwd_subs(L, b):
    
    """
    Foward Substitution for Lower Triangular Nonsingular Matrices
    
    Based on: Stefanica, pag. 40. Table 2.1. LA Primer for FA (2014)
    """
    
    # define 'n'
    n = len(b)
    
    # define 'y'
    y = np.zeros(n)
    
    # initial 'y[0]'
    y[0] = b[0] / L[0][0]
    
    # outer looop
    for j in range(2, n +1):
        sumV = 0
        # inner loop
        for k in range(1, j-1 +1):
            sumV = sumV + (L[j-1][k -1] * y[k-1])
        
        # updating values of final vector
        y[j-1] = (b[j-1] - sumV)/L[j-1, j-1]
    
    # return final vector as column matrix
    return y.reshape(-1,1)

def LU_bwd_subs(U, y):
    
    """
    Backward Substitution for Upper Triangular Nonsingular Matrices
    
    Based on: Stefanica, pag. 45. Table 2.3. LA Primer for FA (2014)
    """    
    
    # define 'n'
    n = len(y)
    
    # define 'x'
    x = np.zeros(n)
    
    # last 'x[0]'
    x[n-1] = y[n-1] / U[n-1][n-1]
    
    # outer looop
    for j in range(n - 1, 0, -1):
        sumV = 0
        # inner loop
        for k in range(j +1, n +1):
            sumV = sumV + (U[j-1][k -1] * x[k-1])
        
        # updating values of final vector
        x[j-1] = (y[j-1] - sumV)/U[j-1][j-1]
    
    # return final vector as column matrix
    return x.reshape(-1,1)


################################# LU Decomp. Naive (non real) #################################
def GeneralSolver(a, b):
    return np.linalg.solve(a, b)


#######################################################################################################################################
############################################# ONE PERIOD MARKET MODEL #################################################################
#######################################################################################################################################
class ArrowDebreuModel(object):
    
    """
    Arrow Debreu Market Model
    
    Please, to know about the prcess involve here, check Chap 3. Stefanica. A LAP for FE (2014)
    """
    
    def __init__(self):
        self.errorMsg1 = "Warning! 'midp_vector' param should be different than'None'"
        
    def __baseVariables__(self, optionchain):
        # initial price vector
        St0 = optionchain["Price"].values
        
        # strike price vector
        Kvec = optionchain["Strike"].values
        
        # type options vector as 1 or 0
        typeOp = \
        [
            1 if tpo.lower() == 'call' else 0 for tpo in optionchain["Type"].to_list()
        ]
        
        return St0, Kvec, typeOp
    
    def __MidPointVectorComp__(self, vector_strikes):
        __K__ = np.unique(vector_strikes)
        return (__K__[1:] + __K__[:-1]) / 2        

    
    def __ComputePayoffMatrix__(self, 
                              strike_vector,
                              full_state_vec,                                  
                              option_type_vector):
        
        # find the payoff M tau matrix
        M_tau =  np.array(
            [
                max(w - K, 0) if option_type_vector[i] else max(K-w, 0) 
                for i, K in enumerate(strike_vector) for w in full_state_vec
            ]
        ).reshape(strike_vector.shape[0], full_state_vec.shape[0])
        return M_tau
    
    def __ADProcess__(self, 
                chain_info, 
                state1_vec,
                staten_vec,
                nonpriced_chain,
                metric,
                predefined_midpoints, 
                midp_vector,
                print_base_state_vector):
        
        # initial variables
        St0, Kvec, typeOp = self.__baseVariables__(optionchain=chain_info)

        # use predef midpoint vector
        if predefined_midpoints: 
            assert predefined_midpoints != None, self.errorMsg1
            midPoints = midp_vector
        else: 
            midPoints = self.__MidPointVectorComp__(vector_strikes=Kvec)
        
        # print base state vectors, just to check
        if print_base_state_vector: 
            print(f" >> Base Midpoints (ω^2 to ω^n-1) : {midPoints}")
        
        # state_1 and state_n possible P permutations
        state1_and_n_perm = list(
            itertools.product(
                state1_vec, 
                staten_vec
            )
        )
        
        # empty list to save the information
        M_list = []
        Qs = []
        arbitrage_free = []
        list_states = []
        
        # iteration over the P permutations 
        for tuple_states in state1_and_n_perm:
            
            # define the full vectors of state
            full_state_vec = \
            np.array(
                [tuple_states[0]] + list(midPoints) + [tuple_states[-1]]
            )              
            list_states.append(full_state_vec)
            
            # find M tau matrix
            Mtau = self.__ComputePayoffMatrix__(
                strike_vector = Kvec,
                full_state_vec = full_state_vec,
                option_type_vector = typeOp
            )
            M_list.append(Mtau)
            
            # find Q vector
            Q = np.linalg.solve(Mtau, St0)
            Qs.append(Q)
            
            # arbitrage-free?
            boolArbitrageFree = (Q>=0).all()
            
            if boolArbitrageFree:
                # find the parameters of the non priced options
                S_star, K_star, type_star = self.__baseVariables__(
                    optionchain = nonpriced_chain
                )
                
                # find the matrix payoff for the non priced options
                __Mtau2__ = self.__ComputePayoffMatrix__(
                    strike_vector = K_star,
                    full_state_vec = full_state_vec,                            
                    option_type_vector = type_star
                )
                
                # pricing nonpriced options under the AD Model (Q vals)
                pricingOptionsVector = np.dot(__Mtau2__, Q)
                
                if metric == "RMSE":
                    # finding the RMSE of the model
                    RMSE = np.sqrt(
                        np.sum(
                            (1/S_star.shape[0]) * 
                            ((pricingOptionsVector - S_star)**2 / (S_star)**2)
                        )
                    )
                    arbitrage_free.append([boolArbitrageFree, RMSE])
                    
                elif metric == "relative":
                    # finding relative error 
                    relative_error = abs(pricingOptionsVector - S_star)/S_star
                    
                    # finding the absolute error
                    mae = np.sum(abs(pricingOptionsVector - S_star))/S_star.shape[0]
                    
                    # add column of priced options and relative error w.r.t the model
                    nonpriced_chain["model_priced"] = pricingOptionsVector
                    nonpriced_chain["relative_error"] = relative_error
                    
                    # finding the RMSE of the model
                    RMSE = np.sqrt(
                        np.sum(
                            (1/S_star.shape[0]) * 
                            ((pricingOptionsVector - S_star)**2 / (S_star)**2)
                        )
                    )                    
                    
                    arbitrage_free.append([boolArbitrageFree, mae, RMSE, nonpriced_chain])
                else:
                    print(">>>>> Non Metric Selected")
                    pass
            else:
                arbitrage_free.append(boolArbitrageFree)
                            
        return M_list, Qs, arbitrage_free, list_states
    
    def compute(self, 
                tuple_events, 
                state1_vec,
                staten_vec,
                options_chain,
                metric,
                predefined_midpoints = False, 
                midp_vector = None,
                print_base_state_vector = False):
        
        # select specific events
        list_events = []
        for tpl in tuple_events:
            list_events.append(
                options_chain.query(f"Type == '{tpl[0]}' & Strike == {tpl[1]}")
            )  

        # defining option chain for the model and for the pricing 
        base_options = pd.concat(list_events)
        nonpriced_options = options_chain.drop(index=base_options.index.to_list())        

        return self.__ADProcess__(
            chain_info = base_options, 
            state1_vec = state1_vec,
            staten_vec = staten_vec,
            nonpriced_chain = nonpriced_options,
            metric = metric,
            predefined_midpoints = predefined_midpoints, 
            midp_vector = midp_vector,
            print_base_state_vector = print_base_state_vector
        )
    
################################# CUIBIC SPILINE & CHOLESKY DECOMP ################################
def zero_rates_disfactor_continous(times, dis_factors):
    """
    Find the corresponding N month zero rates given a table.
    
    The two main inputs are two lists, one with the time cashflow and another with discount factors.
    
    Normally, useful at problems like:
    
    Date Discount Factor
    1 months 0.9983
    4 months 0.9935
    10 months 0.9829
    13 months 0.9775
    20 months 0.9517
    22 months 0.9479
    """
    n = len(times)
    
    zero_rates = []
    
    for idx in range(n):
        r = -(1/times[idx]) * np.log(dis_factors[idx]) 
        zero_rates.append(r)
        
    return zero_rates

def eCubicInter(x, v):
    """
    Efficient Cubic Spline Interpolation
    
    Inputs: 
        - x: 1D np.array or list with cat values
        - v: 1D np.array or list with num values
    
    Outputs:
        M, w, a, b, c, d
        
        where: 
            - "M" and "w": matrix and vector of linear sys
            - "a", "b", "c", "d": final coeff of the linear sys
            
    Based on: Stefanica, NLA Primer for FA (2014). Table 6.8. 
    """
    # check if x and v have same length
    assert len(x) == len(v), "Your two inputs have different length!"
    
    # defining total length
    n = len(v)
    
    # z vector empty list
    z = []
    
    # empty M matrix (to fill)
    M_ = np.zeros((n-1, n-1))
    
    # loop to compute z values
    for i in range(1, n-1):
        z_ = 6 * (np.divide(v[i+1] - v[i], x[i+1]- x[i]) \
                  - np.divide(v[i] - v[i-1], x[i]-x[i-1]))
        z.append(z_)
    
    # loops to fill M
    for i in range(1, n-1):
        M_[i,i] = 2*(x[i+1] - x[i-1]) 
    for i in range(1, n-2):
        M_[i,i+1] = x[i+1] - x[i]
    for i in range(2, n-1):
        M_[i, i-1] = x[i] - x[i-1]

    # making M square matrix
    M = np.delete(M_, (0), axis=0)
    M = np.delete(M, (0), axis=1)

    # computing w
    w = np.linalg.solve(M, z)
    
    # adding w0 and wn
    w = np.insert(w, 0, 0)
    w = np.append(w, 0)
    
    # empty list of a, b, c, d
    a, b, c, d = [],[],[],[]
    
    # temporal q and r lists
    q, r = [], []
    
    # computing c and d
    for i in range(1, n):
        c_ = np.divide(w[i-1]*x[i] - w[i]*x[i-1], 2*(x[i] - x[i-1]))
        d_ = np.divide(w[i]-w[i-1], 6*(x[i]-x[i-1]))
        c.append(c_)
        d.append(d_)
        
    # computing q an r (then, a and b)
    for i in range(1, n):
        q_ = v[i-1] - c[i-1]*(x[i-1]**2)-(d[i-1]*(x[i-1])**3)
        r_ = v[i] - c[i-1]*(x[i]**2) - d[i-1]*(x[i]**3)
        q.append(q_)
        r.append(r_)
        
    # computing a and b
    for i in range(1, n):
        a_ = np.divide(q[i-1]*x[i] - r[i-1]*x[i-1], x[i] - x[i-1])
        b_ = np.divide(r[i-1] - q[i-1], x[i]-x[i-1])
        a.append(a_)
        b.append(b_)

    return M, z, w, a, b, c, d

def zeroRateECubicSpline(timeintervals, zero_rates, overnight_rate):
    """
    Zero Rates Bridge Function.
    """
    # update the time intervals and the zero rates curves
    timeInt = [0] + timeintervals
    zeroRates = [overnight_rate] + zero_rates
    # return the efficient cubic int. using updated timeInt
    return eCubicInter(x = timeInt, v = zeroRates)
    

def CholeskyDecomp(A, lower=True):
    """
    Basic function to perform a Cholesky Decomposition
    """
    # lower-triangular Cholesky factor
    L = np.linalg.cholesky(A)
    
    # return lower
    if lower: return L
    # return upper
    else: return L.T.conj()
    
############################################################################################################
#################################### OLS, Interplations and eCubic on Polynomials ########################## 
############################################################################################################

# Naive func to check belongingness of any variable to an specific interval
def check_interval(lst, value):
    """
    Function that check if "value" belongs to one interval
    in the list of lists 'lst'
    
    Output:
        - (Bool, idx): boolean if 'value' belongs to any interval on 'lst'
                       and the corresponding 'idx' of the interval.
    """
    for i, sublist in enumerate(lst):
        if value >= min(sublist) and value <= max(sublist):
            return (True, i)
    return (False, None)

# function to evalute a piecewise polynomial function of order 'n' given some intervals
def eval_polynomials(x, intervals, coefficients, zero_start = True):
    """
    Function that evaluates a polynomial of degree n given coeff for diff intervals.
    
    Inputs:
        - x: int or float that is going to be evaluated in the piecewise polynomial func
        - intervals: list of 'm' intervals
        - coefficients: list of list, where each sublist should have the same 
                        'n' elements, representing these the coeff of each term
                        in the polynomial. 
        - zero_start: boolean to consider to start from (0,intervals[0]) as the
                      first interval, or just from (interval[0], interval[1])
    Output:
        - f(x), where f was the piecewise polynomial for the intervals that corresponds to x.
    """
    
    # check if all the list of list of coefficients have same length
    assert len(set(map(len,coefficients))) == 1, \
    "'coefficients' list of list have different length" 
    
    # compute the intervals starting from 0 as list of lists
    if zero_start:
        intervals_ = [0] + intervals
    else:
        intervals_ = intervals
    fullIntervals = [[intervals_[i], intervals_[i+1]] for i in range(len(intervals_)-1)]
    
    # check if the given 'x' belongs to specific interval
    boolBelong, idX_X = check_interval(fullIntervals, x)
    assert boolBelong, "The given 'x' does not belong to any interval in the LSE."
    
    # take only the required coeff for each sublist of coeff 
    coefs_X = [sublist[idX_X] for sublist in coefficients]
    
    # finally compute the polynomial of degree n = coefs_X - 1 corresponding to 'x'
    result = 0
    for idxPower in range(len(coefs_X)):
        result+= coefs_X[idxPower] * (x**idxPower)
    return result

# General OLS Time Series Class | Useful for problem such as Q1 HW7 NLA seminar
class OLS_NLA(object):
    """
    OLS Time Series Class for dataset such as:
    
    - Timeseries dataset such as:
 
        2-year 3-year 5-year 10-year
         1.69   2.58   3.57   4.63
         1.81   2.71   3.69   4.73
         1.81   2.72   3.70   4.74
          ...    ...    ...    ...
     
     - Tabular dataset such as:
     
       Call Price   Strike   Put Price
        225.40      1175      46.60
        205.55      1200      51.55
        186.20      1225      57.15
        167.50      1250      63.30
          ...        ...       ...   

    Methods:
        - 'computeOLS': compute the OLS "y = xA" for a given 'y'.
           This method is only for time series datasets.
        - 'computeTabOLS': compute the OLS "y=xA" for an options tabular dataset.
           Only valid for 2nd type of tabular datasets to find the annualized 
           continous dividend yield and the risk-free rate; i.e.:
                           C - P = PVF - K.disc
        - 'computeCubicSplineInter': compute the eCubic Spline Inter on each row of the dataset
    """
    def __init__(self, data):
        self.data = data
        self.onevector = np.zeros((data.shape[0],1)) + 1
    
    def computeOLS(self, yCol):
        # define the y vector given the base column
        yVec = self.data[[yCol]].values
        
        # define the A non square matrix
        A = pd.concat(
            [pd.DataFrame(self.onevector),
            self.data.loc[:, self.data.columns != yCol]], 
            axis=1
        ).values
        
        # compute the coefficients of the OLS
        x = np.linalg.solve(A.T @ A, A.T @ yVec)
        
        # approximated yCol using coefficients 'x'
        yVecApprox = A @ x
        
        # approximation error
        errorLR = np.linalg.norm(yVec - yVecApprox)
        
        return x, errorLR
    
    def computeTabOLS(self):
        # compute y vector
        yVec = self.data["Call Price"] - self.data["Put Price"]
        
        # define the A non square matrix
        A = pd.concat(
            [pd.DataFrame(self.onevector),
            self.data[["Strike"]] *-1], 
            axis=1
        ).values        
        
        # compute the coefficients of the OLS
        x = np.linalg.solve(A.T @ A, A.T @ yVec)      
        
        return x
    
    def computeCubicSplineInter(self, yCol):   
        
        # define the variable that is going to be cubic interpolated 
        X_ = float(re.findall(r'\d+', ' '.join(yCol))[0])
        
        # define the updated dataframe for the cubic spline int. (without considering 'yCol')
        dfCubicSpline = \
        self.data.loc[:, self.data.columns != yCol]
        
        # define the intervals according to the column names in 'dfCubicSpline'
        intervals = \
        list(
            map(int, re.findall(r'\d+', ' '.join(dfCubicSpline.columns.to_list())))
        )
        
        # iteration to compute eCubic Spline Interpolation on each row
        cubicInterpolated_YCol = []
        for rowIdx in range(dfCubicSpline.shape[0]):
            # traditional cubic spline
            M, z, w, a, b, c, d = eCubicInter(x = intervals, v = dfCubicSpline.loc[rowIdx].values) 
            # save the coefficients in just one list
            coeffCubic = [a, b, c, d]
            # find the value of polynomial for 'X_' | desactivate the 'zero_start' parameter 
            valuePol = eval_polynomials(X_, intervals, coeffCubic, zero_start=False)
            # save the value on the initial empty list 
            cubicInterpolated_YCol.append(valuePol)
            
        # redefine the result as a np array col vector
        cubicInterpolated_YCol = np.array(cubicInterpolated_YCol).reshape(-1,1)
        
        # computing the approximation error of the Cubic Interpolation
        errorCI = np.linalg.norm(self.data[[yCol]].values - cubicInterpolated_YCol)
        
        # return the ecubic spline interpolated values as an array col vector
        return cubicInterpolated_YCol, errorCI
    
    
####################################################################################################################################
############################################ EXACT SOLUTION IMPLIED VOLATILITY #####################################################
######################### https://papers.ssrn.com/sol3/papers.cfm?abstractid=2908494 ###############################################
####################################################################################################################################

def ABC(y, R):
    """
    Belongs to: https://papers.ssrn.com/sol3/papers.cfm?abstractid=2908494 | Table 3
    """
    A = (np.exp((1 - (2/np.pi))*y) - np.exp(-(1 - (2/np.pi))*y))**2
    B = 4 * (np.exp((2/np.pi) * y) + np.exp(-(2/np.pi)*y)) - \
    2 * np.exp(-y) * (np.exp((1 - (2/np.pi))*y) + np.exp(-(1 - (2/np.pi))*y)) * (np.exp(2*y) + 1 - R**2)
    C = np.exp(-2*y) * (R**2 - (np.exp(y) - 1)**2) * ((np.exp(y) + 1)**2 - R**2)
    return A, B, C

def call_impliedVol_esol(Cm, K, T, F, r):
    """
    Belongs to: https://papers.ssrn.com/sol3/papers.cfm?abstractid=2908494 | Table 1
    """    
    y = np.log(F/K)
    alpha_C = Cm / (K * np.exp(-r * T))
    R = (2 * alpha_C) - np.exp(y) + 1
    A, B, C =  ABC(y, R)
    beta = (2*C) / (B + np.sqrt((B**2) + (4*A*C)))
    gamma = -(np.pi/2) * np.log(beta)
    
    if y >= 0:
        C0 = K * np.exp(-r * T) * (np.exp(y)*A*np.sqrt(2*y) - 0.5)
        if Cm <= C0:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) - np.sqrt(gamma - y))
        else:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    else:
        C0 = K * np.exp(-r * T) * (0.5*np.exp(y) - A*(-np.sqrt(-2*y)))
        if Cm <= C0:
            sigma = \
            (1/np.sqrt(T)) * (-np.sqrt(gamma + y) + np.sqrt(gamma - y))
        else:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    
    return sigma

def put_impliedVol_esol(Pm, K, T, F, r):
    """
    Belongs to: https://papers.ssrn.com/sol3/papers.cfm?abstractid=2908494 | Table 2
    """    
    y = np.log(F/K)
    alpha_P = Pm / (K * np.exp(-r * T))
    R = (2 * alpha_P) + np.exp(y) - 1
    A, B, C =  ABC(y, R)
    beta = (2*C) / (B + np.sqrt((B**2) + (4*A*C)))
    gamma = -(np.pi/2) * np.log(beta)
    
    if y >= 0:
        P0 = K * np.exp(-r * T) * ( 0.5 - np.exp(y)*A*(-np.sqrt(2*y)) )
        if Pm <= P0:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) - np.sqrt(gamma - y))
        else:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    else:
        P0 = K * np.exp(-r * T) * ( A*(np.sqrt(-2*y)) - (np.exp(y)/2) ) 
        if Pm <= P0:
            sigma = \
            (1/np.sqrt(T)) * (-np.sqrt(gamma + y) + np.sqrt(gamma - y))
        else:
            sigma = \
            (1/np.sqrt(T)) * (np.sqrt(gamma + y) + np.sqrt(gamma - y))
    
    return sigma

def getExactImpliedVol_StefanicaRados_OptionChain(dataset, deltaT, PVF, disc, rf):
    """
    Iterated version of the Exact Implied Volatility from Stefanica and Rados's paper.
    
    The desired input format of 'dataset' should be:
    
    Call Price 	Strike 	Put Price
    0 	260.000000 	2150 	35.250000
    1 	238.850006 	2175 	38.950001
    2 	218.149994 	2200 	43.000000
    3 	197.949997 	2225 	47.599998
    4 	178.149994 	2250 	52.650000
    ...    ...      ...     ...
    
    The final output will be a dataframe of the form:
    
            Implied Vol Call 	Implied Vol Put
    Strike 		
    2150.0 	0.173934 	0.174163
    2175.0 	0.169045 	0.169302
    2200.0 	0.164212 	0.164308
    2225.0 	0.159419 	0.159458
    2250.0 	0.154392 	0.154464
    2275.0 	0.149455 	0.149557
    ...    ...      ...     ...     
    
    """
    # empty dict to save results
    exact_impvol_sol = {}
    # iterate over each option price
    for idxPos in range(dataset.shape[0]):
        # get temporal dataset info
        chainInfo = dataset.iloc[idxPos]
        # implied volatility for call
        sigmaCall = call_impliedVol_esol(
            Cm=chainInfo["Call Price"], K=chainInfo["Strike"], T=deltaT, F=PVF/disc, r=rf
        )
        # implied volatility for put
        sigmaPut = put_impliedVol_esol(
            Pm=chainInfo["Put Price"], K=chainInfo["Strike"], T=deltaT, F=PVF/disc, r=rf
        )    
        # save implied volatilities to dictionary
        exact_impvol_sol[chainInfo["Strike"]] = [sigmaCall, sigmaPut]
    # transform final result as dataframe
    impVol = pd.DataFrame(
        exact_impvol_sol, index=["Exact Implied Vol Call", "Exact Implied Vol Put"]
    ).T.rename_axis('Strike')
    return impVol

def tangencyPortfolioComputation(mu_vector, rf, sigma, return_mu_bar = True):
    """
    Function to compute the tangency portfolio.
    
    Inputs:
        - mu_vector: np.array with expected returns, s.a. np.array([mu1, mu2, ..., muN])
        - rf: float representing the risk-free rate
        - sigma: np.array with Covariance Matrix (this should be a NxN matrix)
        
    Output:
        - 1D np.array with the weights of the tangency portfolio
        - expected return of the tangency portfolio
        - standard deviatin of the tangency portfolio
        - sharpe rati of the tangency portfolio
    """
    # reassign dim
    if mu_vector.ndim == 1:
        mu_vector = mu_vector.reshape(-1,1) 
    # create one 1 Nx1 vector
    vec1 = (np.zeros(mu_vector.shape[0]) +  1).reshape(-1,1)
    # compute mu_bar
    mu_bar = mu_vector - (rf * vec1) 
    # compute the weights of the tangency portfolio
    omegaT = (1/(vec1.T @ np.linalg.inv(sigma) @ mu_bar)) * (np.linalg.inv(sigma) @ mu_bar)
    # compute expected return, std and sharpe of tangency portfolio
    mu_T = rf + (mu_bar.T @ omegaT) 
    std_T = np.sqrt(omegaT.T @ sigma @ omegaT)
    sharpe_T = (mu_T - rf)/std_T
    
    if return_mu_bar:
        return omegaT, mu_T, std_T, sharpe_T, mu_bar
    else:
        return omegaT, mu_T, std_T, sharpe_T
    
def minimumVariancePortfolio(expected_mu, mu_bar, omega_T, rf, std_and_sharpe = False, covMatrix = None):
    """
    Function to compute the minimum variance portfolio.
    
    Useful for problem such as: 
    
    "What is the weight of the cash position in an optimal investment
    portfolio with X% expected rate of return? What are the weights of the assets in
    this portfolio?"
    
    Inputs:
        - expected_mu : float representing the expected return
        - mu_bar: adjusted expected returns array vector of before the comp. of  the tangency portfolio
        - omega_T: the array vector containing the weights of the tangency portfolio
        - rf: risk free rate
    Output:
        - float representing the cash position 
        - array vector with the weights of the minimum variance portfolio
    """
    
    weightCash = 1 - (expected_mu - rf)/(mu_bar.T @ omega_T)
    
    weightsAssets = (1 - weightCash) * omega_T
    
    if std_and_sharpe:
        sigmaMinVarPort = np.sqrt(weightsAssets.T @ covMatrix @ weightsAssets)
        sharpe = (expected_mu - rf) / sigmaMinVarPort
        return weightCash, weightsAssets, sigmaMinVarPort, sharpe
    else:
        return weightCash, weightsAssets   

def maxReturnPortfolio(expected_std, mu_bar, omega_T, rf, covMatrix, mean_and_sharpe = False):
    """
    Function to compute the maximum return portfolio.
    
    Useful for problem such as: 
    
    "What is the weight of the cash position in an optimal investment 
    portfolio with 30% standard deviation of the rate of return? What are the weights of
    the assets in this portfolio?"
    
    Inputs:
        - expected_std : float representing the expected std
        - mu_bar: adjusted expected returns array vector of before the comp. of  the tangency portfolio
        - omega_T: the array vector containing the weights of the tangency portfolio
        - rf: risk free rate
    Output:
        - float representing the cash position 
        - array vector with the weights of the minimum variance portfolio
    """
    
    vec1 = (np.zeros(mu_bar.shape[0]) + 1).reshape(-1,1)
    
    weightCash = 1 - expected_std/np.sqrt(omega_T.T @ covMatrix @ omega_T) * np.sign(vec1.T @ np.linalg.inv(covMatrix) @ mu_bar)
    
    weightsAssets = (1 - weightCash) * omega_T
    
    if mean_and_sharpe:
        meanMaxReturnPortf = rf + (mu_bar.T @ weightsAssets)
        sharpe = (meanMaxReturnPortf - rf) / expected_std
        return weightCash, weightsAssets, meanMaxReturnPortf, sharpe
    else:
        return weightCash, weightsAssets    

#####################################################################################################################################    
################################################### COMPLEMENTARY METHODS ###########################################################
#####################################################################################################################################

def returnsComputation(dataset, typeReturns = 'standard', reverse = False, idxSide = -1):
    """
    Simple function to compute returns or log returns.
    
    It includes a parameter 'reverse' to compute it on its reverse form.
    
    Notice that 'idxSide' is the lenght of the difference; i.e, return for 2 periods should be -2.
    
    The main input 'dataset' is a dataframe of prices without including dates!
    """
    if reverse:
        dataset = dataset.apply(lambda x: x.iloc[::-1]).reset_index(drop=True)
        
    if typeReturns == 'standard':
        return ((dataset.shift(idxSide) - dataset) / dataset).dropna()
    elif typeReturns == 'log':
        return dataset.apply(lambda col: np.log(col.shift(idxSide) / col) ).dropna()
    else:
        raise ValueError('Invalid Type of Returns!')
        
def sampleCovcCorr(dfReturns):
    """
    Function to compute the sample covariance and sample correlation
    from a dataset of standard or log returns.
    """
    
    Tbar = dfReturns - dfReturns.mean()
    
    # computing sample covariance matrix
    sampleSigma = pd.DataFrame((1/(Tbar .shape[0] - 1)) * np.dot(Tbar.T.values, Tbar.values))
    
    # computing diagonal covariance matrix
    diagSigma = np.linalg.inv(np.sqrt(np.diag(np.diag(sampleSigma))))
    
    # computing sample correlation matrix
    sampleOmega = pd.DataFrame(diagSigma @ sampleSigma @ diagSigma)
    
    return sampleSigma, sampleOmega        

def NLA_linearRegression(df, target_position_variable, return_approx_error = False, globalSelection = True):
    """
    Function to compute the linear regression using a base dataset.
    
    Usually, useful to solve questions like:
    > "Find the linear regression of the COLUMN IDX i 
       with respect to the OTHER COLUMNS".
       
    Inputs:
        - dataset: pd.dataframe that contains the base variables
        - target_position_variable: int which represents the 'y' vector from the idx column in 'dataset'
        - return_approx_error: bool to return the approximation error or not
        - globalSelection: bool which reflects the type of selection w.r.t. the target variable.
            if globalSelection = True:
                > the column with the target variable will be replaced with '1's.
                (activate this if ONLY IF THE QUESTION SPECIFIES TO USE A VECTOR OF 1's)
            else:
                > the column with the target variable will be dropped from the original 'df'.
    Output:
        - x: array with coefficients 
        - approxError: float representing the approximation error (optional) 
    """
    # create a copy of the input df
    dataset = df.copy()
    # select the target vector from the target column in df
    y = dataset.iloc[:, target_position_variable].values.reshape(-1,1)
    # test of the type of selection
    if globalSelection:
        # replace the column of the target variable in the org. df with 1's
        dataset.iloc[:, target_position_variable] = 1
    else:    
        # drop the column of the target variable in the org. df
        dataset = dataset.iloc[:, dataset.columns != dataset.columns[target_position_variable]]
    
    # define the A array matrix
    A = dataset.values
    # compute the values for x
    x = np.linalg.solve(A.T @ A, A.T @ y).reshape(-1,1)
    
    # if return the approximation error is required
    if return_approx_error:
        yApprox = A @ x
        approxError = np.linalg.norm(y - yApprox)
        return x, approxError
    else:
        return x
    
def correlation_from_covariance(covariance):
    """
    Simple function to compute the correlation matrix from a covariance matrix.
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation    

def diagonallyDominantTest(A, typeTest = "strict", orientation = "row", confmessage = False):
    """
    Function to test if a matrix is diagonally dominant:
        - Strictly diagonally dominant
        - Weakly diagonally dominant
        - Column Strictly diagonally dominant 
    """
    diagValues = np.diag(A)
    
    if all(i > 0 for i in np.diag(A)) and confmessage:
        print(" >>> IMPORTANT! All diag. entries are positive!")    
    
    results = {}
    
    for idx in range(0, A.shape[0]):
        
        eva = f"A({idx}, {idx})"
        
        if orientation.lower() == "row":
            absSumNonDiag = np.sum(abs(np.delete(A[idx], idx)))
        elif orientation.lower() == "column":
            absSumNonDiag = np.sum(abs(np.delete(A[:, idx], idx)))
        
        if typeTest.lower() == "strict":
            test = abs(diagValues[idx]) > absSumNonDiag  
        elif typeTest.lower() == "weak":
            test = abs(diagValues[idx]) >= absSumNonDiag 
        else:
            raise ValueError("Not recognized 'typeTest' parameter!")
            
        results[eva] = [test, absSumNonDiag]
    return results