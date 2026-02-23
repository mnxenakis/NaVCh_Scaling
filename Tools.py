import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from uncertainties import ufloat

# from uncertainties.umath import *
# from scipy.stats import kstest
from scipy.signal import savgol_filter
import os

import ModelParameters
import math
from scipy.interpolate import splrep, BSpline
from scipy.stats import gamma
from scipy.stats import norm

from collections import Counter

import statistics

from matplotlib.pyplot import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':55})
rc('text', usetex=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import gaussian_kde


# pdb code
_ , PDB = os.path.split(os.getcwd())  

'''
		Cumulative atom number modeling
'''
##	Model Richards	##
def ResidualsRichards(params, x, data, der = 0):
	
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['inv_zeta']*v['nu']*(x - v['l_i']))
	
	# Richards model
	if (der == 0):
		model = v['A']*(1 + v['nu']*phi)**(-1/v['nu'])
	if (der == 1):
		model = v['A']*v['inv_zeta']*v['nu']*phi*(v['nu']*phi + 1)**(-1./v['nu'] - 1)
	if (der == 2):
		phi_inv = np.exp(v['inv_zeta']*v['nu']*(x - v['l_i']))
		model = -v['A'] * (v['inv_zeta']**2) * (v['nu']**2) * (phi_inv - 1) / ( ((1 + v['nu']*phi)**(1/v['nu'])) * (v['nu'] + phi_inv)**2 )
	if (der == 'log0'):
		phi_inv = np.exp(v['inv_zeta']*v['nu']*(x - v['l_i']))  
		model =  x*v['inv_zeta']*v['nu'] / ( phi_inv + v['nu'] ) 
	if (der == 'log1'):
		phi_inv = np.exp(v['inv_zeta']*v['nu']*(x - v['l_i']))
		model = -x*v['inv_zeta']*v['nu'] * (phi_inv - 1) / ( phi_inv + v['nu'] )
	
	# Return residuals
	return model - data

##	Model Logistic	##
def ResidualsLogistic(params, x, data, der = 0):
    
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['inv_zeta']*(x - v['l_i']))
	
	# Logistic model
	if (der == 0):
		model = v['A']*(1 + phi)**( -1 )
	if (der == 1):
		model = v['A']*v['inv_zeta']*phi*(1 + phi)**(-2)
	if (der == 2):
		phi_inv = 1/phi
		model = - v['A']*(v['inv_zeta']**2)*phi_inv*(phi_inv - 1) / (phi_inv + 1)**3
	if (der == 'log0'):
		phi_inv = 1/phi
		model = x*v['inv_zeta'] / ( phi_inv + 1 ) 
	if (der == 'log1'):
		model = x*v['inv_zeta'] / ( ( np.exp(v['inv_zeta']*v['l_i']) - np.exp(v['inv_zeta']*x) ) /( np.exp(v['inv_zeta']*v['l_i']) + np.exp(v['inv_zeta']*x)) )
	
	# Return residuals
	return model - data
		
##	Model Gompertz ##
def ResidualsGompertz(params, x, data, der=0):
    
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['inv_xi']*(x - v['l_i']))
	
	# Gompertz model
	if (der == 0):
		model = v['A']*np.exp(- phi) 
	if (der == 1):
		model = v['inv_xi']*v['A']*np.exp(- phi - v['inv_xi']*(x - v['l_i'])) 
	if (der == 2):
		model = v['inv_xi']*v['A']*(v['inv_xi']*phi - v['inv_xi'])*np.exp(- phi - v['inv_xi']*(x - v['l_i'])) 
	if (der == 'log0'):
		model = x*v['inv_xi']*np.exp(v['inv_xi']*(v['l_i'] - x))
	if (der == 'log1'):
		model = x*v['inv_xi']*(phi - 1)
		
	# Return residuals
	return model - data
		
## Set initial guesses for model fitting procedures ##
def initGuesses(l, max_n):
	
	A_initGuess 	= max_n	
	a_initGuess 	= 0.1
	infl_initGuess 	= 0.5*(max(l) - min(l))
	nu_initGuess 	= 1.
	
	return [[A_initGuess, 		0.5*A_initGuess, 		1.5*A_initGuess], 		# roughly 1
			[a_initGuess, 		0.001*a_initGuess, 		200*a_initGuess], 		# typical values are 0.1. As the temperature of the system increases towards the hydrophobic core, this measure diverges, i.e., the correlation length vanishes. However, hydrophobic interaction decay length stays roughly the same since they are temperature-driven.
			[infl_initGuess, 	0.5*infl_initGuess, 	1.5*infl_initGuess], 	# typical values are 0.5*(L-R)
			[nu_initGuess, 		0.001*nu_initGuess, 	3*nu_initGuess]]	# typical values are 1

##	Model trace ## 
def GeomModel(x, InputParams, modelType=1, der=0):
	
	from lmfit import Parameters

	params = Parameters()
	params.add('A', value = InputParams[0])
	params.add('l_i', value = InputParams[2])
	if (modelType == 1):
		params.add('inv_zeta', value = InputParams[1])
		params.add('nu', value = InputParams[3])
		model = ResidualsRichards(params, x, np.zeros(len(x)), der)
	if (modelType == 2):
		params.add('inv_zeta', value = InputParams[1])
		params.add('nu', value = InputParams[3])
		model = ResidualsLogistic(params, x, np.zeros(len(x)), der)
	if (modelType == 3):
		params.add('inv_xi', value = InputParams[1])
		model = ResidualsGompertz(params, x, np.zeros(len(x)), der)
	
	return model

##	Model parameters ##
def StatModelParameters(x, y, RETURN_BEST = True):
	
	from lmfit import Minimizer, Parameters, fit_report

	# Initialize the solver
	initParam_val = initGuesses(x, max(y))
	
	# Init param
	params_richards = Parameters()
	params_richards.add('A', value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_richards.add('inv_zeta', value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
	params_richards.add('l_i', value = initParam_val[2][0], min = initParam_val[2][1], max = initParam_val[2][2])
	params_richards.add('nu', value = initParam_val[3][0], min = initParam_val[3][1], max = initParam_val[3][2])
	
	params_logistic = Parameters()
	params_logistic.add('A', value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_logistic.add('inv_zeta', value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
	params_logistic.add('l_i', value = initParam_val[2][0], min = initParam_val[2][1], max = initParam_val[2][2])
	
	params_gompertz = Parameters()
	params_gompertz.add('A',   value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_gompertz.add('inv_xi',   value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
	params_gompertz.add('l_i', value = initParam_val[2][0], min = initParam_val[2][1], max = initParam_val[2][2])
	
	# wrappers
	results = []
	aics = []
	
	# Create a Minimizer object (ric)
	minner = Minimizer(ResidualsRichards, params_richards, fcn_args=(x, y))	# Plug in the logged data.
	fit_richards = minner.minimize(method = 'L-BFGS-B')

	# .. a detailed fit report!
	# print(fit_report(fit_richards)) 
	
	aics_ric		= fit_richards.aic
	# Get all models that you need ..
	model	 		= ResidualsRichards(fit_richards.params, x, np.zeros(len(x)))
	# fitting error
	MAFE_model 		= FittingError(y, model)
	# pvalue
	pvalue 			= CompareDistributions(y, model)
	# additional length-scale parameters
	add_params_ric 	= GeomModelDomains(fit_richards.params, 1)

	# wrap up		# val									# unc
	results.append([fit_richards.params['A'].value, 		fit_richards.params['A'].stderr, 		# A 					[0,1]
					fit_richards.params['inv_zeta'].value, 	fit_richards.params['inv_zeta'].stderr, # a 					[2,3]
					fit_richards.params['l_i'].value, 		fit_richards.params['l_i'].stderr,		# l_i 					[4,5]						
					fit_richards.params['nu'].value, 		fit_richards.params['nu'].stderr,		# nu					[6,7]
					add_params_ric[0],						add_params_ric[1], 						# mu					[8,9]
					add_params_ric[2],						add_params_ric[3],						# tau					[10,11]	
					add_params_ric[4],						add_params_ric[5],						# l_lag					[12,13]
					add_params_ric[6],						add_params_ric[7],						# l_asy					[14,15]
					MAFE_model, 							pvalue, 								# errors on model		[16,17]
					aics_ric, fit_richards.chisqr, fit_richards.redchi, fit_richards.bic,   		# GoF scores 			[18:21]						
					1])																				# modType				[22]							
	aics = aics_ric
	
	
	# Create a Minimizer object (log)
	minner 			= Minimizer(ResidualsLogistic, params_logistic, fcn_args=(x, y))	# Plug in the logged data.
	fit_logistic	= minner.minimize(method = 'L-BFGS-B')
	aics_log 		= fit_logistic.aic
	# Get all models that you need ..
	model	 		= ResidualsLogistic(fit_logistic.params, x, np.zeros(len(x)))
	# fitting error
	MAFE_model 		= FittingError(y, model)
	# pvalue
	pvalue 			= CompareDistributions(y, model)
	# additional length-scale parameters
	add_params_log 	= GeomModelDomains(fit_logistic.params, 2)
	# wrap up		# val									# unc
	results.append([fit_logistic.params['A'].value, 		fit_logistic.params['A'].stderr, 		# A 					[0,1]
					fit_logistic.params['inv_zeta'].value, 		fit_logistic.params['inv_zeta'].stderr, 		# a 					[2,3]
					fit_logistic.params['l_i'].value, 		fit_logistic.params['l_i'].stderr, 		# l_i 					[4,5]						
					1.,										0,										# nu					[6,7]
					add_params_log[0],						add_params_log[1], 						# mu					[8,9]
					add_params_log[2],						add_params_log[3],						# tau					[10,11]	
					add_params_log[4],						add_params_log[5],						# l_lag					[12,13]
					add_params_log[6],						add_params_log[7],						# l_asy					[14,15]
					MAFE_model, 							pvalue, 								# errors on model		[16,17]
					aics_log, fit_logistic.chisqr, fit_logistic.redchi, fit_logistic.bic,   		# GoF scores 		    [18:21]																										# modType				[21]
					2])																				# modType				[22]																	# maxEnt				[23,24,25]
	aics = np.append(aics, aics_log)
	
	
	# Create a Minimizer object (gomp)
	minner 			= Minimizer(ResidualsGompertz, params_gompertz, fcn_args=(x, y))	# Plug in the logged data.
	fit_gompertz 	= minner.minimize(method = 'L-BFGS-B')
	aics_gomp 		= fit_gompertz.aic
	# get all models that you need ..
	model	 		= ResidualsGompertz(fit_gompertz.params, x, np.zeros(len(x)))
	# fitting error
	MAFE_model 		= FittingError(y, model)
	# pvalue
	pvalue 			= CompareDistributions(y, model)
	# additional length-scale parameters
	add_params_gomp = GeomModelDomains(fit_gompertz.params, 3)
	# wrap up		# val									# unc
	results.append([fit_gompertz.params['A'].value, 		fit_gompertz.params['A'].stderr, 		# A 					[0,1]
					fit_gompertz.params['inv_xi'].value, 	fit_gompertz.params['inv_xi'].stderr, 	# a 					[2,3]
					fit_gompertz.params['l_i'].value, 		fit_gompertz.params['l_i'].stderr, 		# l_i 					[4,5]						
					ModelParameters.NU_MIN,					0,									    # nu					[6,7]
					add_params_gomp[0],						add_params_gomp[1], 				    # mu					[8,9]
					add_params_gomp[2],						add_params_gomp[3],					    # tau					[10,11]	
					add_params_gomp[4],						add_params_gomp[5],					    # l_lag					[12,13]
					add_params_gomp[6],						add_params_gomp[7],					    # l_asy					[14,15]
					MAFE_model, 							pvalue,  								# errors on model		[16,17]
					aics_gomp, fit_gompertz.chisqr, fit_gompertz.redchi, fit_gompertz.bic,   		# GoF scores 			[18:21]																									
					3])                                                                             # modType				[22]	
																										
	aics = np.append(aics, aics_gomp)
	
	## Decide which mode parameters work best .. aics criterium based		
	ind_bestModel = np.where(aics == min(aics))[0][0]		
	if (RETURN_BEST != True):
		ind_modSelect = 0
	else:
		ind_modSelect = ind_bestModel

	_ = {}
	# Fill this dict with what you really need ..
	_['A'] = results[ind_modSelect][0]
	_['A_unc'] = results[ind_modSelect][1]
	_['inv_zeta'] = results[ind_modSelect][2]
	_['inv_zeta_unc'] = results[ind_modSelect][3]
	_['l_i'] = results[ind_modSelect][4]
	_['l_i_unc'] = results[ind_modSelect][5]
	_['nu'] = results[ind_modSelect][6]
	_['nu_unc'] = results[ind_modSelect][7]
	_['l_lag'] = results[ind_modSelect][12]
	_['l_lag_unc'] = results[ind_modSelect][13]
	_['l_asy'] = results[ind_modSelect][14]
	_['l_asy_unc'] = results[ind_modSelect][15]
	
	_['MAFE'] = results[ind_modSelect][16]
	_['pvalue'] = results[ind_modSelect][17]
	_['AICS'] = results[ind_modSelect][18]
	
	_['modType'] = results[ind_bestModel][22] # always return the best model index

	
	return _
	
## Additional model information ##
def GeomModelDomains(params, modType):

	if (params['A'].stderr != None):	
		A_ = ufloat(params['A'].value, params['A'].stderr)
	else:	
		A_ = ufloat(params['A'].value, 0)
		
	if (params['l_i'].stderr != None):	
		l_i_ = ufloat(params['l_i'].value, params['l_i'].stderr)
	else:
		l_i_ = ufloat(params['l_i'].value, 0)
	
	if (modType != 3):
		
		if (params['inv_zeta'].stderr != None):	
			a_ = ufloat(params['inv_zeta'].value, params['inv_zeta'].stderr)
		else:
			a_ = ufloat(params['inv_zeta'].value, 0)
		
		if (modType == 1):
			if (params['nu'].stderr != None):	
				nu_ = ufloat(params['nu'].value, params['nu'].stderr)
			else:
				nu_ = ufloat(params['nu'].value, 0)
		else:
			nu_ = ufloat(1, 0)
	
		# Calc add vars
		# l_i = l_lag + (A / mu) * (1 + nu_)**(-1/nu_) \implies
		# l_i = l_lag + \xi * (1 + \nu) \implies  \xi (1 + \nu) = \tau * (1 + nu_)**(-1/nu_) 
		# \xi / \tau = (1 + nu_)**(-1/nu_ - 1) 
		mu_ = A_ * a_ * nu_ * ((1 + nu_)**(-1 - 1/nu_))
		tau_ = A_ / mu_
		l_lag_ = l_i_ - tau_ * (1 + nu_)**(-1/nu_)
		l_asym_	= l_lag_ + tau_
	
	else: 
		
		if (params['inv_xi'].stderr != None):	
			a_gomp_ = ufloat(params['inv_xi'].value, params['inv_xi'].stderr)
		else:
			a_gomp_ = ufloat(params['inv_xi'].value, 0)
			
		# Calc add vars
		mu_ = (A_ * a_gomp_) / np.exp(1)
		tau_ = A_ / mu_
		l_lag_ = (a_gomp_ * l_i_ - 1)/ a_gomp_
		l_asym_	= l_lag_ + tau_
		
	
	return [mu_.nominal_value, mu_.std_dev,
			tau_.nominal_value,	tau_.std_dev, 
			l_lag_.nominal_value, l_lag_.std_dev, 
			l_asym_.nominal_value, l_asym_.std_dev]					


'''
		Model quality
'''
##	MAFE ##
def FittingError(y, y_fit, TYPE = 'MAFE'):
	
	if (TYPE == 'MAFE'):
		return np.mean(abs((y -  y_fit)))
	if (TYPE == 'SDAFE'):
		return np.std(abs((y -  y_fit)))

##	P-value ##
def CompareDistributions(sample1, sample2):

	'''
	if res.pvalue < ModelParameters.ALPHA_VAL:
		print('The null hypothesis is rejected. \n The samples do not have the same distribution. \n\n')
	'''
		 	
	return stats.cramervonmises_2samp(sample1, sample2, method = 'asymptotic').pvalue


'''
		Handling derivatives and zeros
'''
##	Intrapolate ##
def Intrapolator(x,y,xx):
	tck = splrep(x, y, k=1)
	yy = BSpline(*tck)(xx)
	# cruel way to deal with edge effects (important only for small l):
	if (min(xx) < min(x)):
		yy[np.argmin(xx)] = y[np.argmin(x)]
	return yy


## Smoothing ##
def Smoother(y, window):
	return savgol_filter(y, window, ModelParameters.POLYORDER)

##	Coarse derivative ##
def CoarseDifferentiation(x, y, window):

	y_smooth = Smoother(y, window)
	dy_dx = np.gradient(y_smooth, x)
        
	return dy_dx

##	N-th order radial profile ##
def RadialProfile(x, y, window, xx = [], der = 1):

	# Calculate n-th order derivative 
	for i in range(der):

		# "coarsening" acts as a primary (first-stage) low-pass flter
		y = CoarseDifferentiation(x, y, window)

	if (len(xx) != 0):

		# "interploating" acts as a secondary (second-stage) low-pass filter
		y = Intrapolator(x,y,xx)

	return y

##  Find zero crossings  ##
def ZeroCrossings(x, y):

	crossings = []
	count = 0
	for i in range(len(y)-1):
		if y[i]*y[i+1] < 0:
			x1 = x[i+1]
			x0 = x[i]
			fx1 = y[i + 1]
			fx0 = y[i]
			if (fx1 - fx0 != 0):
				crossings.append(x1 - fx1 * (x1 - x0) / (fx1 - fx0))
			count += 1

	return [crossings, count]
	
##	Get sliding window: is some * multiplier of \xi = 1/ (a * \nu) = \zeta / \nu ##
##  * here we multiply with 1/ModelParameters.HOLE_RES, but this can be mantually adjusted
def GetSlidingWindow(a, nu, modType, fac):
	
	if (modType != 3):
		window 	= math.ceil((1./(a*nu)) * fac)
	else:
		window	= math.ceil((1./(a)) * fac)	
	
	return window

## Match value with (scaler) index ##
def Match(val, vec):
	abs_diff = abs(val - np.asarray(vec))
	return np.where(abs_diff == min(abs_diff))[0].astype(int) 


'''
		Hydr moms 
'''
## Odd-order hydr mom ##
# n-order odd moment (vector), n = 2k + 1, k = 1, 2, 3, 4, ..
def OddOrderHydrMom(vecs, hvals, order):  
	
	data = []
	for i in order:
		if (i == 1):
			data.append(np.sum(vecs * hvals[:, np.newaxis], axis = 0))
		else:
			data.append(np.sum(vecs * np.sum(vecs**i,axis=1)[:, np.newaxis] * hvals[:, np.newaxis], axis = 0))
	
	return np.asarray(data)

##	Even-order hydr mom ##
 # n-order even moment (scalar), n = 2k, k = 1, 2, 3, 4, ..
def EvenOrderHydrMom(vecs, hvals, order):
	
	data = []
	for i in order:
		if (i == 0):
			data.append(sum(hvals))
		else:
			data.append(np.sum( np.sum(vecs**i, axis = 1) * hvals ))

	return np.asarray(data)

##	Get the phi-values for each residue ##
def GetPhiValuesRes(array, res_index, domain = 'lag', orderOfhydrMom = 0, coordinate = 'z', NR_OF_ENTRIES = 12):
	
	data_res = array[res_index]
	
	# select the order
	if (orderOfhydrMom % 2 == 0): 
		order_index = int(0.5*orderOfhydrMom)
		data_res = GetColumn(data_res, [order_index])
		if (domain == 'lag'):
			inds = np.arange(0, len(data_res), NR_OF_ENTRIES) 
		if (domain == 'infl'):
			inds = np.arange(1, len(data_res), NR_OF_ENTRIES) 
		if (domain == 'asy'):
			inds = np.arange(2, len(data_res), NR_OF_ENTRIES) 
	else:
		order_index = int(0.5*(orderOfhydrMom - 1))
		data_res = GetColumn(data_res, [order_index])
		if (domain == 'lag'):
			if (coordinate == 'x'):
				inds = np.arange(3, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'y'):
				inds = np.arange(4, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'z'):
				inds = np.arange(5, len(data_res), NR_OF_ENTRIES) 
		if (domain == 'infl'):
			if (coordinate == 'x'):
				inds = np.arange(6, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'y'):
				inds = np.arange(7, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'z'):
				inds = np.arange(8, len(data_res), NR_OF_ENTRIES) 
		if (domain == 'asy'):
			if (coordinate == 'x'):
				inds = np.arange(9, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'y'):
				inds = np.arange(10, len(data_res), NR_OF_ENTRIES) 
			if (coordinate == 'z'):
				inds = np.arange(11, len(data_res), NR_OF_ENTRIES) 

	return data_res[inds]
	


##	Get the hydr mom data ##
def GetHydrMom(array, pp_index, typeOfAtom = 'pathic', orderOfhydrMom = 0, coordinate = 'z'):
	
	
	# data around the pore point
	data_pp = array[pp_index]
	
	# select the category
	if (typeOfAtom == 'pathic'):
		type_index = 0
	if (typeOfAtom == 'phobic'):
		type_index = 1
	if (typeOfAtom == 'philic'):
		type_index = 2
	if (typeOfAtom == 'shell'):
		type_index = 3
		
	# select the order
	if (orderOfhydrMom % 2 == 0): 
		order_index = 0	# even
		specific_order_index = int(0.5*orderOfhydrMom) # specific order index
		# get the data you need
		data = []
		for scale_entry in data_pp:
			data.append(scale_entry[type_index][order_index][specific_order_index])
	else:				
		order_index = 1	# odd
		specific_order_index = int(0.5*(orderOfhydrMom - 1)) # specific order index
		if (coordinate == 'z'):
			coord_index = 2
		else:
			if (coordinate == 'x'):
				coord_index = 0
			if (coordinate == 'y'):
				coord_index = 1		
		# get the data you need
		data = []
		for scale_entry in data_pp:
			data.append(scale_entry[type_index][order_index][specific_order_index][coord_index])
				
	return np.asarray(data)

'''
		Scaling analysis
'''
##	Get geometric indices (i.e., identify scaling intervals)
def GetGeomIndices_other(l, data, interval_edges):
	
	l_interval = []
	data_interval = []

	for i in range(len(interval_edges)-1):

		left_edge = interval_edges[i]
		right_edge = interval_edges[i+1]
	
		left_edge_ind = np.argmin(abs(left_edge - l))
		right_edge_ind = np.argmin(abs(right_edge - l))

		if ((right_edge_ind - left_edge_ind) < ModelParameters.MIN_DOMAIN_SIZE):
			exit('\n ... EXITING SMOOTHLY ... fitting interval too small! \n')

		inds_interval = np.arange(left_edge_ind, right_edge_ind, 1)
		
		l_interval.append(l[inds_interval])
		data_interval.append(data[inds_interval])


	return [l_interval, data_interval]

'''
		Scaling analysis
'''
##	Get geometric indices (i.e., identify scaling intervals)
def GetGeomIndices(l, data, l_domains):
	
	# Find the indices for:
	# the max curve domain
	l_d1 			= 	l_domains[0]
	abs_diff 		= 	abs(l - l_d1)
	ind_d1 			= 	np.where(abs_diff == min(abs_diff))[0]
	# the inflection domain ..
	l_i 			= 	l_domains[1]
	abs_diff 		= 	abs(l - l_i)
	ind_l_i 		= 	np.where(abs_diff == min(abs_diff))[0]
	# the min curve domain
	l_d2 			=	l_domains[2]
	abs_diff 		= 	abs(l - l_d2)
	ind_d2 			= 	np.where(abs_diff == min(abs_diff))[0]
	
	# And create the corresponding data sets:
	# for the d1 domain ..
	inds_d1 		= np.arange(0,int(ind_d1 + 1) - 1, 1)
	l_d1  			= l[inds_d1]
	data_d1 		= data[inds_d1]
	
	# for the first part of the inflection domain ..
	inds_infl1 		= np.arange(int(ind_d1) + 1, int(ind_l_i), 1)
	l_infl1 		= l[inds_infl1]
	data_infl1 		= data[inds_infl1]
	# for the second part of the inflection domain ..
	inds_infl2 		= np.arange(int(ind_l_i) + 1, int(ind_d2), 1)
	l_infl2  		= l[inds_infl2]
	data_infl2 		= data[inds_infl2]
	
	# for the d2mptote domain ..
	inds_d2 		= np.arange(int(ind_d2) + 1, len(l), 1)
	l_d2			= l[inds_d2]
	data_d2			= data[inds_d2]

	# for the pre-inflection domain (as a whole) ..
	inds_pre_infl 	= np.arange(0, int(ind_l_i), 1)
	l_pre_infl	 	= l[inds_pre_infl]
	data_pre_infl 	= data[inds_pre_infl]
	# for the post-inflection domain (as a whole) ..
	inds_post_infl 	= np.arange(int(ind_l_i), len(l), 1)
	l_post_infl 	= l[inds_post_infl]
	data_post_infl 	= data[inds_post_infl]
	
	if (len(data_d1) < ModelParameters.MIN_DOMAIN_SIZE or np.std(data_d1) < 1e-08):
		exit('\n ... EXITING SMOOTHLY ... fitting interval is too small (preInfl1)! Model fails. \n')
	
	if (len(data_infl1) < ModelParameters.MIN_DOMAIN_SIZE):
		exit('\n ... EXITING SMOOTHLY ... fitting interval is too small (infl1)! Model fails. \n')
		
	if (len(data_infl2) < ModelParameters.MIN_DOMAIN_SIZE):
		exit('\n ... EXITING SMOOTHLY ... fitting interval is too small (infl2)! Model fails. \n')
	
	# The outermost shells may be empty or too few.
	# In either case, we use the last quarter of the infl2 domain.
	if (len(data_d2) < ModelParameters.MIN_DOMAIN_SIZE or np.std(data_d2) < 1e-08):
		print('\n ... WARNING ... fitting interval is too small (postInfl2)! The postInfl2 interval is replaced with the tail of infl2! \n')
		tail_length = int(0.25 * len(l_infl2))
		data_d2 = data_infl2
		l_d2 = l_infl2
		
	# To return
	l_ret = [l_d1,  		l_infl1,		l_infl2,		l_d2, 	 	l_pre_infl, 		l_post_infl]
	data_ret = [data_d1, 	data_infl1, 	data_infl2, 	data_d2, 	data_pre_infl, 		data_post_infl]
	
	return [l_ret, data_ret]

## Scaling behavior over selected interval ##
def ScalingBehavior(l, data, l_domains, PLOT = False, other = False):
			
	if other:
		data_sets = GetGeomIndices_other(l, data, l_domains)
	else:
		data_sets = GetGeomIndices(l, data, l_domains)
		
	nrOfSegments = len(data_sets[0])
	logLogFit = []

	for i in range(nrOfSegments):
		
		logLogFit.append(ScalingAnalysis(data_sets[0][i], abs(data_sets[1][i])))
			
	if PLOT:
		for i in range(nrOfSegments):
			expModelApprox = np.exp(logLogFit[i][0]*np.log(data_sets[0][i]) + logLogFit[i][1])
			if (i == 0):
				plt.title("Direct environment")
			plt.plot(data_sets[0][i], np.log(abs(data_sets[1][i])), 'bo', markersize = 10, alpha = 0.15)
			# plt.plot(l, abs(data), "bo", alpha = 0.1, markersize = 10)
			plt.plot(data_sets[0][i], np.log(expModelApprox), 'r--', linewidth = 2)
			plt.xlabel("\AA")
			plt.ylabel("Hyrd. moment")
				
			plt.show()
	

	return logLogFit

##	Scaling analysis ##
def ScalingAnalysis(x, y):

	# If there are very few atoms zeros can still occur
	# Introduce a correction
	if np.any(y == 0):
    	# Ensure min(y) is positive for scaling
		scale = np.max([np.abs(np.min(y)), 1e-8])
		correction = np.random.normal(0, scale * 0.01, size=y.shape)
		y = y + np.abs(correction)

	coeffs, cov = np.polyfit(np.log(x), np.log(y), deg=1, cov=True)
	model = np.exp(coeffs[0]*np.log(x) + coeffs[1])
	res = stats.pearsonr(np.log(y), coeffs[0]*np.log(x) + coeffs[1])

	if math.isnan(res[0]) or math.isnan(res[1]):
		exit(" \n\n .. Exiting smoothly .. You still have nans (consider increasing ModelParameters.MIN_DOMAIN_SIZE) .. \n\n")

	return [coeffs[0], coeffs[1], res[0], res[1], np.sqrt(np.diag(cov)), FittingError(y, model), FittingError(y, model, TYPE='SDAFE')]



def ScalingInfo_res(array, orderOfhydrMom = 0, component = 'z', interval =  'powerLaw_pre', infoBlock = 'exp', INTERVAL_SPECIFIC = True):
	
	# 0,1 \to 0
	# 2,3 \to 1
	# 4,5 \to 2
	# etc ..
	order_index = int(0.5*orderOfhydrMom) 
	# if 0, 2, 4, .. \to 0 (even)
	if (orderOfhydrMom % 2 == 0):
		specific_order_index = 0
	# if 1, 3, 5, .. \to 1 (odd z)
	else:
		if (component == 'x'):
			specific_order_index = 1
		if (component == 'y'):
			specific_order_index = 2
		if (component == 'z'):
			specific_order_index = 3
		
	# Specific data entries ..
	if INTERVAL_SPECIFIC:
		if (interval ==  'powerLaw_preInfl1'):
			data_index = 0
		if (interval ==  'powerLaw_infl1'):
			data_index = 1
		if (interval ==  'powerLaw_infl2'):
			data_index = 2
		if (interval ==  'powerLaw_postInfl2'):
			data_index = 3
		if (interval ==  'powerLaw_pre'):
			data_index = 4
		if (interval ==  'powerLaw_post'):
			data_index = 5
	else:
		data_index = interval
	
	if (infoBlock == 'exponent'):
		info_index = 0
	if (infoBlock == 'coeff'):
		info_index = 1
	if (infoBlock == 'PC'):
		info_index = 2
	if (infoBlock == 'p-val'):
		info_index = 3
	if (infoBlock == 'unc'):
		info_index = 4
	if (infoBlock == 'MAFE'):
		info_index = 5
	if (infoBlock == 'SDAFE'):
		info_index = 6	
	
	return np.asarray(GetColumn(array, [order_index, specific_order_index, data_index, info_index]))


##	Retrieve scaling information ##
def ScalingInfo(array, orderOfhydrMom = 0, component = 'z', interval =  'powerLaw_pre', infoBlock = 'exp', INTERVAL_SPECIFIC = True):
	
	# 0,1 \to 0
	# 2,3 \to 1
	# 4,5 \to 2
	# etc ..
	order_index = int(0.5*orderOfhydrMom) 
	# if 0, 2, 4, .. \to 0 (even)
	if (orderOfhydrMom % 2 == 0):
		specific_order_index = 0
	# if 1, 3, 5, .. \to 1 (odd z)
	else:
		if (component == 'z'):
			specific_order_index = 1
		if (component == 'abs'):
			specific_order_index = 2
		
	# Specific data entries ..
	if INTERVAL_SPECIFIC:
		if (interval ==  'powerLaw_preInfl1'):
			data_index = 0
		if (interval ==  'powerLaw_infl1'):
			data_index = 1
		if (interval ==  'powerLaw_infl2'):
			data_index = 2
		if (interval ==  'powerLaw_postInfl2'):
			data_index = 3
		if (interval ==  'powerLaw_pre'):
			data_index = 4
		if (interval ==  'powerLaw_post'):
			data_index = 5
	else:
		data_index = interval
	
	if (infoBlock == 'exponent'):
		info_index = 0
	if (infoBlock == 'coeff'):
		info_index = 1
	if (infoBlock == 'PC'):
		info_index = 2
	if (infoBlock == 'p-val'):
		info_index = 3
	if (infoBlock == 'unc'):
		info_index = 4
	if (infoBlock == 'MAFE'):
		info_index = 5
	if (infoBlock == 'SDAFE'):
		info_index = 6	
	
	return np.asarray(GetColumn(array, [order_index, specific_order_index, data_index, info_index]))


'''
		Decompose into two two components:
		h_j = h_j,+ + h_j,-
		
		In computational practice, we require that the sing of the hydropathic components remains 
		unchanges for l larger than l_cutoff, where l_cutoff is the "cutoff" scale. 

		Note that decomposition is NOT always possible. 
		In that case, we assign phi and pho to + and -, respectively. 
'''
## Decompose ##
def Decompose(h_pho, h_phi, ind_l_cutOff):

	# Successful decomposition: above the lag scale the sign does not change.
	# The part before the l_cutoff is also considered, if singularties for l < l_cutoff are not dramatic
	# In fact they are not expected to be, since we are dealing with first-order "jumps" over zero and not continuous zero-crossings.
	# Typically l_cutoff is approximately equal to l_0.	
	if (all(h_pho[ind_l_cutOff:ModelParameters.N_SCALES] < 0) and all(h_phi[ind_l_cutOff:ModelParameters.N_SCALES] > 0)):
		h_plus = h_phi
		h_minus = h_pho
	elif (all(h_pho[ind_l_cutOff:ModelParameters.N_SCALES] > 0) and all(h_phi[ind_l_cutOff:ModelParameters.N_SCALES] < 0)):
		h_plus = h_pho
		h_minus = h_phi
	# Note that it could be that the decomp Ansatz is satisfied only for a few pore points.
	# Because the transition from point-to-point is smooth, the topology is not changing abruptly.
	# Meaning, that if h_plus = h_phi, then most likely this con
	# Hence we can assign:
	else:
		print("\n Warning! Decomp. ansatz violation! \n")
		h_plus = h_phi
		h_minus = h_pho

	return [h_plus, h_minus]


'''
		Variants processing 
'''
##	Variants listing ##
def VariantsListing(column, type = 'str'):

	i = 0
	var_list = []
	for x in range(len(column)): 
		if (i == 0):
			print('\n Collecting', column[x].value, 'related variants! ')
			# continue
		else:
			if (column[x].value == None):
				break
			if	(type == 'int'):
				var_list.append(int(column[x].value))
			else:
				var_list.append(column[x].value)
		i += 1

	return var_list

##	Variants grouping ##
def GroupVariants(classificationType, groupType):
	

	'''
			Gnomad/ClinVar
	'''
	# VUS
	inds_vus = np.where(classificationType == 'Uncertain significance')[0]
	# unknown
	inds_unknown = np.where(classificationType == 'not provided')[0]
	# benign
	inds_benign = np.where(classificationType == 'Benign')[0]
	inds_benignLikelyBenign = np.where(classificationType == 'Benign/Likely benign')[0]
	inds_likelyBenign = np.where(classificationType == 'Likely benign')[0]
	# pathogenic
	inds_path = np.where(classificationType == 'Pathogenic')[0]
	inds_pathLikelyPath = np.where(classificationType == 'Pathogenic/Likely pathogenic')[0]
	inds_likelyPath = np.where(classificationType == 'Likely pathogenic')[0]
	inds_conflInterPath = np.where(classificationType == 'Conflicting interpretations of pathogenicity')[0]
	
	
	'''
			Pain Disease Phenotypes
	'''
	inds_iem = np.where(classificationType == 'IEM')[0]
	inds_sfn = np.where(classificationType == 'SFN')[0]
	inds_pepd = np.where(classificationType == 'PEPD')[0]
	inds_neutral = np.where(classificationType == 'Neutral')[0]
	inds_lof = np.where(classificationType == 'LoF')[0]
	inds_ndisc = np.where(classificationType == 'ndisc')[0]

	'''
			Classification Status
	'''
	inds_classificationStatus = np.where(classificationType == 'missclassified')[0]


	'''
			Grouping Pain Disease Phenotypes 
	'''
	'''
			GoF groupings
	'''
	if (groupType == 'PEPD'):
		inds = inds_pepd
	
	if (groupType == 'SFN'):
		inds = inds_sfn
	
	if (groupType == 'IEM'):
		inds = inds_iem

	if (groupType == 'GoF'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd))

	if (groupType == 'GoF/Pathogenic(certain)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_path, inds_conflInterPath))
		
	if (groupType == 'GoF/Pathogenic(certain/likely)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_path, inds_conflInterPath, inds_pathLikelyPath))
			
	if (groupType == 'GoF/Pathogenic(certain/likely/likely)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_path, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))

	if (groupType == 'Pathogenic(certain)'):
		inds = inds_path

	if (groupType == 'Pathogenic(confl)'):
		inds = inds_conflInterPath
	

	'''
			LoF groupings
	'''
	if (groupType == 'LoF'):
		inds = inds_lof
		
	if (groupType == 'LoF/Pathogenic(certain)'):
		inds = np.concatenate((inds_lof, inds_path, inds_conflInterPath))
		
	if (groupType == 'LoF/Pathogenic(certain/likely)'):
		inds = np.concatenate((inds_lof, inds_conflInterPath, inds_pathLikelyPath))
		
	if (groupType == 'LoF/Pathogenic(certain/likely/likely)'):
		inds = np.concatenate((inds_lof, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))
	

	'''
			GoF/LoF groupings
	'''
	if (groupType == 'GoF/LoF'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_lof))
	
	if (groupType == 'GoF/LoF/ndisc'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_lof, inds_ndisc))
		
	if (groupType == 'GoF/LoF/Pathogenic(certain)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_lof, inds_path, inds_conflInterPath))
	
	if (groupType == 'GoF/LoF/Pathogenic(certain/likely)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_lof, inds_path, inds_conflInterPath, inds_pathLikelyPath))
		
	if (groupType == 'GoF/LoF/Pathogenic(certain/likely/likely)'):
		inds = np.concatenate((inds_iem, inds_sfn, inds_pepd, inds_lof, inds_path, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))


	'''
			Pathogenic groupings
	'''	
	if (groupType == 'Pathogenic(certain)'):
		inds = np.concatenate((inds_path, inds_conflInterPath))
		
	if (groupType == 'Pathogenic(certain/likely)'):
		inds = np.concatenate((inds_path, inds_conflInterPath, inds_pathLikelyPath))
		
	if (groupType == 'Pathogenic(likely)'):
		inds = inds_pathLikelyPath
	
	if (groupType == 'Pathogenic(likely/likely)'):
		inds = np.concatenate((inds_pathLikelyPath, inds_likelyPath))
	
	if (groupType == 'Pathogenic(certain/likely/likely)'):
		inds = np.concatenate((inds_path, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))


	'''
			Benign groupings
	'''
	if (groupType == 'Neutral'):
		inds = inds_neutral 

	if (groupType == 'Benign(certain)'):
		inds = inds_benign 
	
	if (groupType == 'Benign(certain/likely)'):
		inds = np.concatenate((inds_benign, inds_benignLikelyBenign))
	
	if (groupType == 'Benign(likely)'):
		inds = inds_benignLikelyBenign
		
	if (groupType == 'Benign(certain/likely/likely)'):
		inds = np.concatenate((inds_benign, inds_benignLikelyBenign, inds_likelyBenign))
		
	if (groupType == 'Benign(likely/likely)'):
		inds = np.concatenate((inds_benignLikelyBenign, inds_likelyBenign))
		
	if (groupType == 'Neutral/Benign(certain)'):
		inds = np.concatenate((inds_benign, inds_neutral)) 
	
	if (groupType == 'Neutral/Benign(certain/likely)'):
		inds = np.concatenate((inds_benign, inds_benignLikelyBenign, inds_neutral))
		
	if (groupType == 'Neutral/Benign(certain/likely/likely)'):
		inds = np.concatenate((inds_benign, inds_benignLikelyBenign, inds_likelyBenign, inds_neutral))
		
	if (groupType == 'Benign(likely/likely)'):
		inds = np.concatenate((inds_benignLikelyBenign, inds_likelyBenign))
		
	if (groupType == 'BenignLikelyBenign'):
		inds = inds_benignLikelyBenign
		
	if (groupType == 'LikelyBenign'):
		inds = inds_likelyBenign
		

	'''
			Newly discovered
	'''
	if (groupType == 'ndisc'):
		inds = inds_ndisc 


	'''
			VUS groupings
	'''
	if (groupType == 'VUS'):
		inds = inds_vus 
	
	if (groupType == 'unknown'):
		inds = inds_unknown
	
	if (groupType == 'VUS/unknown'):
		inds = np.concatenate((inds_vus, inds_unknown))
	

	'''
			All
	'''
	if (groupType == 'All'):
		inds = np.concatenate((inds_vus, inds_unknown, 
						 		inds_benign, inds_benignLikelyBenign, inds_likelyBenign, inds_neutral,
								inds_iem, inds_sfn, inds_pepd, inds_lof, inds_path, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))
	if (groupType == 'NO VUS'):
		inds = np.concatenate((
						 		inds_benign, inds_benignLikelyBenign, inds_likelyBenign, inds_neutral,
								inds_iem, inds_sfn, inds_pepd, inds_lof, inds_path, inds_conflInterPath, inds_pathLikelyPath, inds_likelyPath))
	
	
	'''
			Classification Status
	'''
	if (groupType == 'missclassified'):
		inds = inds_classificationStatus

	
	return inds
	

"""
		Machine Learning Tools
"""
## Some simple learners ##
def MachineLearner(X_train, y_train, X_test, y_test, method = 'LogReg', kernel = None, chooseThreshold = "f1"):
	
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)
	
	# Create model
	if (method == 'LogReg'):
		model = LogisticRegression(C=ModelParameters.C, penalty='l2', max_iter=ModelParameters.MAX_ITER, random_state=ModelParameters.CLASSIFIER_SEED)

	if (method == 'SVC'): 
		if (kernel != None):
			model = SVC(kernel=kernel, probability=True, class_weight='balanced', random_state=ModelParameters.CLASSIFIER_SEED)
		else:
			exit('\n\n .. Exiting smoothly .. What kind of kernel? \n\n')

	# from sklearn.ensemble import HistGradientBoostingClassifier
	# model = HistGradientBoostingClassifier(min_samples_leaf=1,

    #                                   max_depth=,

    #                                   learning_rate=1,

    #                                   max_iter=1)

	# Fit the data 		
	model.fit(X_train, y_train)
	
	# Get probs of belonging to class '0' (disease-associated)
	p_train = model.predict_proba(X_train)
	p_test = model.predict_proba(X_test)

	# Get accuracies
	acc_train = model.score(X_train, y_train)
	acc_test = model.score(X_test, y_test)
	
	# Get f1 score
	f1_train = f1_score(y_train, model.predict(X_train), average='weighted')
	f1_test = f1_score(y_test, model.predict(X_test), average='weighted')
	
	# Get auc score
	# .. for training dataset
	auc_train = metrics.roc_auc_score(y_train, p_train[:,1])
	# .. for test dataset
	auc_test = metrics.roc_auc_score(y_test, p_test[:,1])

	# get features importance
	if (method == 'LogReg'):
		res = permutation_importance(model, X_train, y_train, n_jobs=-1, n_repeats=ModelParameters.NUM_OF_PERM_REPEATS, random_state=ModelParameters.CLASSIFIER_SEED)
		feat_imp = res.importances_mean

	if (method == 'SVC'):
		res = permutation_importance(model, X_train, y_train, n_jobs=-1, n_repeats=ModelParameters.NUM_OF_PERM_REPEATS, random_state=ModelParameters.CLASSIFIER_SEED)
		# Mean importance scores
		feat_imp = res.importances_mean

	fpr, tpr, thresholds = roc_curve(y_test, p_test[:,1])
	if (chooseThreshold == "f1"):
		f1_scores = [f1_score(y_test, p_test[:,1] >= threshold) for threshold in thresholds]
		optimal_idx = np.argmax(f1_scores)
		optimal_threshold = thresholds[optimal_idx]
	elif (chooseThreshold == "auc"):
		j_scores = tpr - fpr
		optimal_idx = np.argmax(j_scores)
		optimal_threshold = thresholds[optimal_idx]

	"""
	# Plot the predicted probabilities for the positive class
	plt.figure(figsize=(10, 6))
	plt.hist(p_test[:, 1][y_test == 0], bins=50, alpha=0.5, label='Class 0 (Negative)', color='red')
	plt.hist(p_test[:, 1][y_test == 1], bins=50, alpha=0.5, label='Class 1 (Positive)', color='green')

	# Plot the chosen threshold
	plt.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')

	# Labels and legend
	plt.xlabel('Predicted Probability')
	plt.ylabel('Frequency')
	plt.title('Predicted Probabilities with Optimal Threshold')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.show()
	"""

	return [	
				# train		# test
				acc_train, 	acc_test, 		# acc 0, 1
				auc_train, 	auc_test, 		# auc 2, 3
				f1_train, 	f1_test,		# f1  4, 5
				optimal_threshold,			# 6
				# feat importance
				feat_imp, 					# 7
				# scaler
				scaler, 					# 8
				# model
				model						# 9
			] 

##	k-fold corss validation ##
def KFoldsClassifier(X, y, X_unseen = [], method = 'LogReg', kernel = None, chooseThreshold = "f1"):
	
	# Length of data vector
	n_data = len(y)

	# Initialize
	probs_class_0 = []	
	model_class = []
	features_importance = []
	acc_auc_f1_optThres = []		
	
	if (len(X_unseen) != 0):	
		probs_class_0_unseen = []
		model_class_unseen = []

	if (ModelParameters.K > 1):
		folds = np.arange(2, ModelParameters.K + 1, 1)
	else:
		exit('\n\n .. Exiting smoothly .. At least two folds are needed! \n\n')

	# We perform NUM_OF_TRAININGS training cycles for each different fold
	for j in range(ModelParameters.NUM_OF_TRAININGS):

		# We consider k - 1 different foldings. Each folding contains k groups. Each group is considered as a training group only once.
		for k in folds:
			
			'''

					1. Shuffle the dataset randomly
					2. Split the dataset into k groups while ensuring that the ratio ind_class_0/ind_class_1 is *approximately* the same in each fold
					3. For each unique group:
						- Take the group as a hold out or test data set
						- Take the remaining groups as a training data set
			
			'''
			kf = StratifiedKFold(n_splits = k, shuffle = True, random_state = j)
			
			# Split data into k train-test sets
			probs_split = np.zeros(n_data) 
			class_split = np.zeros(n_data) 
		
			## Perform  experiments: Each group is considered as a training group only once.
			# For k = 2: two groups are formed (two differnt experiments)
			# For k = 3: three groups are formed (three differnt experiments)
			for train_indices, test_indices in kf.split(X, y):
				
				# Use the train_ and test_indices to create train_ and test_ data sets				
				X_train, y_train = X[train_indices], y[train_indices]
				X_test, y_test = X[test_indices], y[test_indices]
				
				# Good to check that there are not too little represenetatives of each class ..
				# if (len(np.where(y[test_indices] == 0)[0]) < 10 or len(np.where(y[test_indices] == 1)[0]) < 10):
				#	print('\n\n .. Warning .. your test group may contain insufficient information  .. ', len(np.where(y[test_indices] == 0)[0]), len(np.where(y[test_indices] == 1)[0]))
				
				# Train/test
				modelRes = MachineLearner(X_train, y_train, X_test, y_test, method, kernel, chooseThreshold = chooseThreshold)
			
				# Get probabilities of belonging to class '0'
				# Because of different indexing we need this intermediate array to distribute prob scores among tested cases
				probs_split[test_indices] = modelRes[-1].predict_proba(modelRes[-2].transform(X))[test_indices, 0] # test_indices: point to the tested data set, 0: zero-class probabilites
				class_split[test_indices] = modelRes[-1].predict(modelRes[-2].transform(X))[test_indices]

				acc_auc_f1_optThres.append(modelRes[0:7])
				features_importance.append(modelRes[7])
				
				# Unseen data
				if (len(X_unseen) != 0):
						probs_class_0_unseen.append(modelRes[-1].predict_proba(modelRes[-2].transform(X_unseen))[:,0])
						model_class_unseen.append(modelRes[-1].predict(modelRes[-2].transform(X_unseen)))	
			## Append 	
			# The length of these array is (K - 1) * NUM_OF_TRAININGS
			probs_class_0.append(probs_split)
			model_class.append(class_split)

	# Return: cover all different cases		
	if (len(X_unseen) != 0):
		return [probs_class_0, model_class, acc_auc_f1_optThres, features_importance, probs_class_0_unseen, model_class_unseen]
	
	return [probs_class_0, model_class, acc_auc_f1_optThres, features_importance]
	

'''
		Other useful tools ...
'''
##	Assign VdW radii ##
def VdWRad(atomName):
	
	if (atomName.split()[0][0] == 'C'):
		return 1.65
	if (atomName.split()[0][0] == 'O'):
		return 1.4 
	if (atomName.split()[0][0] == 'S'):
		return 1.8
	if (atomName.split()[0][0] == 'N'):
		return 1.55
	if 'H' in atomName: # any characters combination where 'H' appears
		return 1.00
	if (atomName.split()[0][0] == 'P'):
		return 1.80

##	Eucl distance ##
def EuclNorm(coords1, coords2, setAxis=1):
	return np.sqrt(np.sum((coords1 - coords2)**2, axis=setAxis)) 
						
##	Vector norm ##
def VecNorm(vec):
	return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

##	Rewrite B factors (purely for illustrational purposes) ##
def RewriteBFactors(input_pdb, output_pdb, vals):
	
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    new_b_factors = {str(i): vals[i-1] for i in range(1, len(vals)+1)}

    modified_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_id = line[6:11].strip()
            # Check if the atom ID is in the dictionary of new B-factors
            if atom_id in new_b_factors:
                new_b_factor = new_b_factors[atom_id]
                # Modify the B-factor value in the line
                modified_line = line[:60] + '{:6.2f}'.format(new_b_factor) + line[66:]
                modified_lines.append(modified_line)
            else:
                # If atom ID is not found in the dictionary, keep the original line
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.write(''.join(modified_lines))

	# open the structure in pymol and use: spectrum b, blue_red, selection

## Load data from .txt file ##
def LoadFile(fn, PDB_CODE = 1):

	import pickle

	if (PDB_CODE == 1):
		fn = PDB + '_' + fn + '.txt'

	with open(fn, 'rb') as data_in:
		data = pickle.load(data_in)

	return data

##	Store data to .txt file ##
def StoreFile(data, fn):
	
	import pickle

	fn = PDB + '_' + fn + '.txt'
	with open(fn, 'wb') as data_out:
		pickle.dump(data, data_out)

##	Mode calculation ##
def Mode(data):

	kde = gaussian_kde(data, bw_method='silverman')
	x = np.linspace(min(data), max(data), 1000)
	kde_values = kde(x)
	mode = x[np.argmax(kde_values)]

	return mode

##	Statistical Summary ##
def StatsCalc(data, axis, p_left):

	return [np.mean(data, axis = axis), 
		 	np.std(data, axis = axis), 
			np.median(data, axis = axis), 
			np.percentile(data, q=p_left, axis = axis), 
			np.percentile(data, q=100 -p_left,axis = axis),
			# mode and min, max values are also useful for data sampled from observables being singular
			Mode(data),
			np.min(data),
			np.max(data)]

## Merging residue name and ID ##
def ResidueNameAndID(resNAMES, resIDs, ind_offset, val_offset):

	resIDs[np.where(resIDs > ind_offset)[0]] = resIDs[np.where(resIDs > ind_offset)[0]] - val_offset
	formatted_resNAMES = [resNAME.capitalize() for resNAME in resNAMES]
	merged = [f"{resNAME}{resID}" for resNAME, resID in zip(formatted_resNAMES, resIDs)]
	
	return merged

## Percentilies ##
def LowerAndUpperPercentiles(data, n=ModelParameters.NUM_OF_PERCENTILES, percentile_min=ModelParameters.P_LEFT, percentile_max=(100 - ModelParameters.P_LEFT)):
	
	# Calculate the lower and upper percentile groups, skipping 50 percentile
	percLower = np.percentile(data, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
	percUpper = np.percentile(data, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)
	
	return [percLower, percUpper]    

## Column array ##
def GetColumn(array, jth_entry):
	
	column = []

	if (len(jth_entry) == 1):
		for row in array:
			column.append(row[jth_entry[0]])
	if (len(jth_entry) == 2):
		for row in array:
			column.append(row[jth_entry[0]][jth_entry[1]])
	if (len(jth_entry) == 3):
		for row in array:
			column.append(row[jth_entry[0]][jth_entry[1]][jth_entry[2]])
	if (len(jth_entry) == 4):
		for row in array:
			column.append(row[jth_entry[0]][jth_entry[1]][jth_entry[2]][jth_entry[3]])
	if (len(jth_entry) == 5):
		for row in array:
			column.append(row[jth_entry[0]][jth_entry[1]][jth_entry[2]][jth_entry[3]][jth_entry[4]])
	if (len(jth_entry) == 6):
		for row in array:
			column.append(row[jth_entry[0]][jth_entry[1]][jth_entry[2]][jth_entry[3]][jth_entry[4]][jth_entry[5]])

	return np.asarray(column).flatten()

##	q-Entropy ##
def qEntropy(p, q):
	if (q != 1):
		return ( 1 - sum( np.array(p)**(q) ) ) / (q - 1)
	else:
		exit('\n\n Exiting smoothly .. q index cannot be unit! \n\n')

## Exp func ##
def ExpFunc(x, a):
    return a*np.exp(-x)

## Standarize ##
def Standarization(y, l_i, l):
	ind_l_i = Match(l_i, l)
	y_norm = y/y[ind_l_i]
	return ( y_norm - min(y_norm) ) / ( max(y_norm) - min(y_norm) ) + min(abs(y_norm))

## Standarize ##
def Standarize(y):
	return (y - min(y)) / (max(y) - min(y)) + 0.000001

'''
		Plotters 
'''
## Plot cumulative atom number ##
def Plot_CumulativeAtomNumber(data_n, data_model, min_inds_l_i, max_inds_l_i, hist_data, alpha_percentile = 0.5, n_max_percentile = 5, offset_xlim = 5, 
								ytitle = r'$N$ [atom]', xtitle = r'Scale index', xticks = [], statsType = 'median'):

	# Characteristics of a single channel are summarized in terms of the median statistical descriptor:
	if (statsType == 'median'):
	
		y = np.median(data_n, axis = 0)
		y_model = np.median(data_model, axis = 0)
		
		# label_y = r'$\tilde{N}$'
		# label_model_y = r'$\tilde{n}$'
	
	if (statsType == 'mean_of_medians'):
	
		y = np.mean(data_n, axis = 0)
		y_sd = np.std(data_n, axis = 0)
		y_model = np.mean(data_model, axis = 0)
	
		# label_y = r'$\langle \tilde{N} \rangle$'
		# label_model_y = r'$\langle \tilde{n} \rangle$'
	
	if (statsType != 'median' and statsType != 'mean_of_medians'):
		exit('\n\n Exiting smoothly .. What kind of statitistical descriptor? \n\n')

	## Main figure
	fig, ax = plt.subplots()
	x = np.arange(len(y)) # scaling indices

	ax.plot(x, y, 'bo', alpha = 0.05, markersize = 15)
	
	if (statsType == 'median'):
		percentiles = LowerAndUpperPercentiles(data_n, n = n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color='b', edgecolor=None)
	if (statsType == 'mean_of_medians'):
		ax.fill_between(x, y - y_sd, y + y_sd,  color = 'b', alpha = 0.3)

	ax.plot(x, y_model, 'r--', alpha = 0.6, linewidth = 7)

	from matplotlib.lines import Line2D 
	legend_elements = 	[
						
						Line2D([0], [0], marker='o',  markeredgecolor='b', markerfacecolor='b', markersize=13, alpha=0.5, linestyle='', label=r"$N$"),
						Line2D([0], [0], linewidth = 5, alpha=0.5, color='r', linestyle='--', label=r"$n$")
						
						]
	
	ax.axvspan(min_inds_l_i, max_inds_l_i, alpha=0.1, color='m')

	ax.legend(loc = 'upper left', handles=legend_elements, fontsize = 70) 
	ax.set_title('Critical inflection regime', color = "m", fontsize = 60)
	ax.set_xticks(xticks)
	
	offset_ylim = 250
	ax.set_ylim([min(min(y), min(y_model)) - offset_ylim, max(max(y), max(y_model)) + offset_ylim])
	ax.set_xlim([0, len(y)])
	
	ax.set_ylabel(ytitle,fontsize = 65)
	ax.set_xlabel(xtitle)
	
	plt.show()
	plt.close()

	## histogram 1
	fig, ax1 = plt.subplots()
	ax1.hist(hist_data[0], color = 'm', alpha = 0.25, edgecolor = 'k', label = r'MAFE', bins = 3)
	ax1.hist(hist_data[1], color = 'k', alpha = 0.25, edgecolor = 'k', label = r'P-value', bins = 35)
	ax1.yaxis.labelpad=0.5
	ax1.xaxis.labelpad=0.5
	ax1.tick_params(axis='y')
	ax1.tick_params(axis='x')
	ax1.set_xlim([0, 1])	
	ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
	ax1.legend(loc = 'upper right', fontsize = 50)
	ax1.set_ylabel(r'Freq.', fontsize = 50)
	plt.show()

	## histogram 2
	fig, ax2 = plt.subplots() 
	ax2.hist(hist_data[2], color = 'y', alpha = 0.35, edgecolor = 'k', bins = 15, label = r'$d_{f} \big |_{l_{i}}$ (theory)')
	count, bins, _ = ax2.hist(hist_data[3], color = 'k', alpha = 0.25, edgecolor = 'k', bins = 15, label = r'$d_{f} \big |_{l_{i}}$ (empir.)')	
	params = gamma.fit(hist_data[3])
	# Get the fitted parameters
	shape, loc, scale = params
	print(f'Shape: {shape}, Loc: {loc}, Scale: {scale}')
	std = np.sqrt(shape*scale**2)
	print(f"Average d-value: {shape * scale + loc}, with std: {std}")
	x = np.linspace(min(hist_data[1]), 5, 1000)
	bin_centers = 0.5 * (bins[1:] + bins[:-1])
	fitted_pdf = gamma.pdf(x, shape, loc, scale) * np.mean(np.diff(bin_centers)) * count.sum()
	ax2.plot(x, fitted_pdf, 'k--', alpha = 0.5, lw=8, label=r'$\Gamma$-distr.')
	ax2.yaxis.labelpad=0.5
	ax2.xaxis.labelpad=0.5
	ax2.tick_params(axis='y')
	ax2.tick_params(axis='x')
	ax2.set_xlim([0,5])
	ax2.legend(loc = 'upper right', fontsize = 50)
	ax2.set_ylabel(r'Freq.', fontsize = 50)
	
	plt.show()
	
## Plot atomic hydr energy ##
def Plot_AtomicHydropathicEnergy(data_m0, data_m0_pho, data_m0_phi, data_model, min_inds_l_i, max_inds_l_i, hist_data, alpha_percentile = 0.5, n_max_percentile = 5, offset_xlim = 5, 
								ytitle = r'$| h_{0} | / N$ [kcal/atom]', xtitle = r'Scale index', xticks = [], statsType = 'median'):

	# Characteristics of a single channel are summarized in terms of the median statistical descriptor:
	if (statsType == 'median'):
		
		y = np.median(data_m0, axis = 0)
		y_pho = np.median(data_m0_pho, axis = 0)
		y_phi = np.median(data_m0_phi, axis = 0)
		y_model = np.median(data_model, axis = 0)
		
		# label_y = r'$ \tilde{M_{0}} $'
		# label_model_y = r'$\tilde{\varepsilon}$'
		# label_y_pho = r'$ \tilde{m}_{0,\mathrm{-}} $'
		# label_y_phi = r'$ \tilde{m}_{0,\mathrm{+}}  $'

	if (statsType == 'mean_of_medians'):

		y = np.mean(data_m0, axis = 0)
		y_sd = np.std(data_m0, axis = 0)
		y_pho = np.mean(data_m0_pho, axis = 0)
		y_sd_pho = np.std(data_m0_pho, axis = 0)
		y_phi = np.mean(data_m0_phi, axis = 0)
		y_sd_phi = np.std(data_m0_phi, axis = 0)
		y_model = np.mean(data_model, axis = 0)

		# label_y = r'$\langle \tilde{M_{0}}  \rangle$'
		# label_model_y = r'$\langle \tilde{\varepsilon} \rangle$'
		# label_y_pho = r'$\langle \tilde{m}_{0,\mathrm{-}} \rangle$'
		# label_y_phi = r'$\langle \tilde{m}_{0,\mathrm{+}} \rangle$'

	if (statsType != 'median' and statsType != 'mean_of_medians'):
		exit('\n\n Exiting smoothly .. What kind of statitistical descriptor? \n\n')
	
	label_y = r'$| h_{0} | / N$'
	label_model_y = r'$\varepsilon$'
	label_y_pho = r'$ h_{0,-} / N$'
	label_y_phi = r'$ h_{0,+} / N$'

	## Main figure
	fig, ax = plt.subplots()
	x = np.arange(len(y))

	ax.plot(x, y, 'k-', linewidth = 4, alpha = 0.5, label=label_y)
	ax.plot(x, y_model, '--', color = 'g', linewidth = 5, label=label_model_y)
	ax.plot(x, y_phi, 'r-', linewidth = 4, alpha = 0.5, label=label_y_phi)
	ax.plot(x, y_pho, 'b-', linewidth = 4, alpha = 0.5, label=label_y_pho)
	
	if (statsType == 'median'):
		
		percentiles = LowerAndUpperPercentiles(data_m0, n=n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color='k', edgecolor=None)
	
		percentiles = LowerAndUpperPercentiles(data_m0_phi, n=n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color='r', edgecolor=None)
	
		percentiles = LowerAndUpperPercentiles(data_m0_pho, n=n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color='b', edgecolor=None)
	
	if (statsType == 'mean_of_medians'):
		ax.fill_between(x, y - y_sd, y + y_sd,  color = 'k', alpha = 0.3)
		ax.fill_between(x, y_pho - y_sd_pho, y_pho + y_sd_pho,  color = 'b', alpha = 0.3)
		ax.fill_between(x, y_phi - y_sd_phi, y_phi + y_sd_phi,  color = 'r', alpha = 0.3)

	ax.axvspan(min_inds_l_i, max_inds_l_i, alpha=0.1, color='m')
	ax.set_xlim([0, len(y) + offset_xlim])
	ax.set_ylim([-0.8, 0.8])
	ax.set_xticks(xticks)
	ax.set_xlabel(xtitle)
	ax.set_ylabel(ytitle, fontsize = 60)
	ax.set_xlim([0,len(y)])
	ax.legend(loc = 'upper right')
	plt.show()
	plt.close()

	## histograms
	max_x = 30
	fig, ax1 = plt.subplots()	
	count, bins, _ = ax1.hist(hist_data[1], color = 'k', alpha = 0.25, edgecolor = 'k', bins=10, label = r"$\zeta$")
	count, bins, _ = ax1.hist(hist_data[0], color = 'g', alpha = 0.25, edgecolor = 'k', bins=15, label = r"$\xi$")	
	params_xi = gamma.fit(hist_data[0])
	params_zeta = gamma.fit(hist_data[1])
	# Get the fitted parameters
	shape_xi, loc_xi, scale_xi = params_xi
	shape_zeta, loc_zeta, scale_zeta = params_zeta
	std_xi = np.sqrt(shape_xi*scale_xi**2)
	std_zeta = np.sqrt(shape_zeta*scale_zeta**2)
	print(f"Average xi value: {shape_xi * scale_xi + loc_xi}, with std: {std_xi}")
	print(f"Average zeta value: {shape_zeta * scale_zeta + loc_zeta}, with std: {std_zeta}")
	x = np.linspace(min(hist_data[1]), max_x, 1000)
	bin_centers = 0.5 * (bins[1:] + bins[:-1])
	fitted_pdf_xi = gamma.pdf(x, shape_xi, loc_xi, scale_xi) * np.mean(np.diff(bin_centers)) * count.sum()
	fitted_pdf_zeta = gamma.pdf(x, shape_zeta, loc_zeta, scale_zeta) * np.mean(np.diff(bin_centers)) * count.sum()
	ax1.plot(x, fitted_pdf_xi, 'g--', alpha = 0.5, lw=8, label='$\Gamma$-distr.')
	ax1.plot(x, fitted_pdf_zeta, 'k--', alpha = 0.25, lw=8, label='$\Gamma$-distr.')
	ax1.yaxis.labelpad=0.5
	ax1.xaxis.labelpad=0.5
	
	ax1.tick_params(axis='y')
	ax1.tick_params(axis='x')

	ax1.set_xlim([-2,max_x])
	ax1.legend(loc = 'upper right', fontsize = 50)
	ax1.set_ylabel(r'Freq.', fontsize = 65)
	
	plt.show()
	
## Plot dipole moment ##
def Plot_DipoleMoment(data_z_pos, data_z_ES, data_z_neg, data_z_IS, min_inds_l_i, max_inds_l_i, exp_data, PC_data, orderOfMom = 1, alpha_percentile = 0.5, n_max_percentile = 5, 
					  ytitle = r'$h_{1,\perp}$ (norm.)', xtitle = r'Scale index', statsType = 'median', ticks_step = 6, x_min = -38, x_max = 38):


	# Characteristics of a single channel are summarized in terms of the median statistical descriptor:
	if (statsType == 'median'):

		y_pos = np.median(data_z_pos, axis = 0)
		y_ES = np.median(data_z_ES, axis = 0)
		y_neg = np.median(data_z_neg, axis = 0)
		y_IS = np.median(data_z_IS, axis = 0)

		# label_y_pos = r'$ \tilde{h}_{' + str(orderOfMom) + r',\perp, \uparrow}$'
		# label_y_neg = r'$ \tilde{h}_{' + str(orderOfMom) + r',\perp, \downarrow}$'
	
	if (statsType == 'mean_of_medians'):
		
		y_pos = np.mean(data_z_pos, axis = 0)
		y_ES = np.median(data_z_ES, axis = 0)
		y_sd_pos = np.std(data_z_pos, axis = 0)
		y_neg = np.mean(data_z_neg, axis = 0)
		y_IS = np.median(data_z_IS, axis = 0)
		y_sd_neg = np.std(data_z_neg, axis = 0)

		# label_y_pos = r'$  \tilde{h}_{' + str(orderOfMom) + r',\perp, \uparrow} \rangle $'
		# label_y_neg = r'$  \tilde{h}_{' + str(orderOfMom) + r',\perp, \downarrow} \rangle $'
		
	if (statsType != 'median' and statsType != 'mean_of_medians'):
		exit('\n\n Exiting smoothly .. What kind of statitistical descriptor? \n\n')

	label_y_pos = r'$ h_{' + str(orderOfMom) + r',\perp, \uparrow} $'
	label_y_ES = r'$ h_{' + str(orderOfMom) + r',\perp}$ (ES)'
	label_y_neg = r'$ h_{' + str(orderOfMom) + r',\perp, \downarrow}$'
	label_y_IS = r'$ h_{' + str(orderOfMom) + r',\perp}$ (IS)'
	
	## Main figure
	fig, ax = plt.subplots()

	x = np.arange(len(y_pos))
	ax.plot(x, y_pos, color="r", linewidth = 5, alpha = 0.5, label = label_y_pos)
	# ax.plot(x, y_ES, color="r", linestyle = '--', linewidth = 5, alpha = 0.75, label = label_y_ES)
	ax.plot(x, y_neg, color="b", linewidth = 5, alpha = 0.75, label = label_y_neg)
	# ax.plot(x, y_IS, color="b", linestyle = '--', linewidth = 5, alpha = 0.5, label = label_y_IS)
	

	if (statsType == "median"):
		percentiles = LowerAndUpperPercentiles(data_z_pos, n=n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color="r", edgecolor=None)
	
		percentiles = LowerAndUpperPercentiles(data_z_neg, n=n_max_percentile, percentile_min=5, percentile_max=95)
		for p1, p2 in zip(percentiles[0], percentiles[1]):
			alpha = 1./n_max_percentile
			ax.fill_between(x, p1, p2, alpha=alpha_percentile*alpha, color="b", edgecolor=None)
	
	if (statsType == "mean_of_medians"):
		ax.fill_between(x, y_neg - y_sd_neg, y_neg + y_sd_neg,  color = "b", alpha = 0.3)
		ax.fill_between(x, y_pos - y_sd_pos, y_pos + y_sd_pos,  color = "r", alpha = 0.3)

	ax.axvspan(min_inds_l_i, max_inds_l_i, alpha = 0.1, color='m')
	
	ax.set_xlim([0, len(y_pos)])
	# ax.set_ylim([-6,6])
	ax.set_ylim([-2,2])
	
	ax.axhline(y = 1,linestyle = "--", alpha = 0.25, color = "k",linewidth = 5)
	ax.axhline(y = 0,linestyle = "-", alpha = 0.25, color = "k",linewidth = 5)
	ax.axhline(y = -1, linestyle = "--", alpha = 0.25, color = "k", linewidth = 5)
	
	ax.set_xlabel(xtitle, fontsize = 65)
	ax.set_ylabel(ytitle, fontsize = 65)
	
	ax.legend(loc = 'upper left', fontsize = 50)
	
	plt.show()
	plt.close()

	# Fine-tune bins
	n_bins_PD = 25
	n_bins_PD_largePC = 30
	n_bins_VSD = 60
	n_bins_VSD_largePC = 50

	## Inset
	fig, ax = plt.subplots()

	# Threshold for large PCs
	PC_threshold = 0.85

	# PD hist
	exps_PD = np.array(exp_data[0])
	PCs_PD = np.array(PC_data[0])
	count_PD, bins_PD, _ = ax.hist(exps_PD, color = 'm', alpha = 0.15, bins = n_bins_PD, edgecolor = 'k', label = r'$\eta_{' + str(orderOfMom) + ',\perp, <}$')
	# inds to large PC values
	ind_largePC_PD = np.where(np.array(PCs_PD) > PC_threshold)[0] 
	exps_PD_largePC = exps_PD[ind_largePC_PD]
	# corresponding hist
	ax.hist(exps_PD_largePC, color = 'm', alpha = 0.4, bins = n_bins_PD_largePC, edgecolor = None, label = r'$\eta_{' + str(orderOfMom) + ',\perp,<}$ (PC$>$0.85)')

	mu, std = norm.fit(exps_PD)
	print(f"PDs: Estimated Mean: {mu}, Estimated Std Dev: {std}")
	x = np.linspace(-2, 8, 1000)
	bin_centers = 0.5 * (bins_PD[1:] + bins_PD[:-1])
	p = norm.pdf(x, mu, std) * np.mean(np.diff(bin_centers)) * count_PD.sum()
	ax.plot(x, p, 'm--', linewidth=5, label = 'Norm.-distr.', alpha = 0.5)
	
	# VSD hist (all)
	exps_VSD = np.array(exp_data[1]) 
	PCs_VSD = np.array(PC_data[1])
	ind_largePD_VSD = np.where(PCs_VSD > PC_threshold)[0]
	exps_VSD_largePC = exps_VSD[ind_largePD_VSD]
	ax.hist(exps_VSD, color = 'k', alpha = 0.13, bins = n_bins_VSD, edgecolor = 'k', label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$')
	ax.hist(exps_VSD_largePC, color = 'k', alpha = 0.25, bins = n_bins_VSD_largePC, edgecolor = None, label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (PC$>$0.85)')
	
	"""
	## VSD hist (ES)
	indices = np.where((np.array(princ_coords) >= 0))[0]
	exps_VSD_ES = exps_VSD[indices]
	PCs_VSD_ES = PCs_VSD[indices]
	ax.hist(exps_VSD_ES, color = 'b', alpha = 0.13, bins = n_bins_VSD, edgecolor = 'k', label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (ES)')
	# inds to large PC values
	ind_largePC_VSD_ES = np.where(PCs_VSD_ES > PC_threshold)[0] 
	exps_VSD_largePC = exps_VSD_ES[ind_largePC_VSD_ES]
	# corresponding hist
	ax.hist(exps_VSD_largePC, color = 'b', alpha = 0.17, bins = n_bins_VSD_largePC, edgecolor = None, label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (ES, PC$>$0.85)')

	## VSD hist (IS)
	indices = np.where((np.array(princ_coords) < 0))[0]
	exps_VSD_IS = exps_VSD[indices]
	PCs_VSD_IS = PCs_VSD[indices]
	ax.hist(exps_VSD_IS, color = 'r', alpha = 0.13, bins = n_bins_VSD, edgecolor = 'k', label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (IS)')
	# inds to large PC values
	ind_largePC_VSD_IS = np.where(PCs_VSD_IS > PC_threshold)[0] 
	exps_VSD_largePC = exps_VSD_IS[ind_largePC_VSD_IS]
	# corresponding hist
	ax.hist(exps_VSD_largePC, color = 'r', alpha = 0.17, bins = n_bins_VSD_largePC, edgecolor = None, label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (IS, PC$>$0.85)')
	"""
	
	ax.axvline(x=0, color = 'm', linestyle='-', alpha = 0.5, lw=3)
	ax.axvline(x=0, color = 'k', linestyle='-', alpha = 0.5, lw=3)

	ax.yaxis.labelpad=0.5
	ax.xaxis.labelpad=0.5
	ax.tick_params(axis='y')
	ax.tick_params(axis='x')
	x_ticks = np.arange(x_min, x_max, ticks_step)
	ax.set_xticks(x_ticks)
	ax.set_xlim([x_min, x_max])
	ax.legend(loc = 'upper left', fontsize = 40)
	ax.set_ylabel(r'Freq.', fontsize = 50)
	# ax.set_xlabel(r'Scaling exponent $\eta_{' + str(orderOfMom) + ', \perp, >}$', fontsize = 50)
	ax.set_xlabel(r'Performance (AUC)', fontsize = 50)
	
	plt.show()
	
## Clustering ##
def Clustering(X, inds_class0):
	
	import umap
	from sklearn.cluster import KMeans

	umap_model = umap.UMAP(n_components=2, random_state=42)
	X_umap = umap_model.fit_transform(X)

	# Visualize the 2D projection of the data
	plt.figure(figsize=(10, 8))
	plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
	plt.scatter(X_umap[inds_class0, 0], X_umap[inds_class0, 1], c='red', s=100, label='Class 0', marker='X')
	plt.colorbar(label='Cluster Label')
	plt.xlabel('UMAP-1', fontsize=14)
	plt.ylabel('UMAP-2', fontsize=14)
	plt.title('2D UMAP Projection of n-Dimensional Data', fontsize=16)
	plt.show()

## 3D plot ##
def Plot_3DShape(x, y, z, c, xtitle, ytitle, ztitle, ctitle, location = "right"):

	ax = plt.axes(projection ='3d')
	im = ax.scatter(x, y, z, c = c, s = 50, alpha = 0.7, cmap='coolwarm') # try also cmap = "coolwarm_r"

	ax.set_xlabel(xtitle, fontsize = 65)
	ax.set_ylabel(ytitle, fontsize = 65)
	ax.set_zlabel(ztitle, fontsize = 65)
	
	ax.yaxis.labelpad=30
	ax.zaxis.labelpad=30
	ax.xaxis.labelpad=30

	cbar = plt.colorbar(im, shrink=0.5, location=location)
	cbar_ax = cbar.ax
	cbar_ax.set_xlabel(ctitle, fontsize = 50)
	
	cbar_ax.xaxis.labelpad=10
	
	plt.show()
	plt.close()
	
## Plot exponents ##
def Plot_ExponentsTraces(princ_coord, expPre, expPre_unc, PC_pre, expPost, expPost_unc, PC_post, nu, nu_unc, poreRadius, poreRadius_sd, orderOfMom = 1):

	fig, ax = plt.subplots()
	ax.plot(princ_coord, expPre, color='g', linewidth = 3, label = r'\( \eta_{'+ str(orderOfMom) + ',\perp, <} \)')
	ax.fill_between(princ_coord, expPre + expPre_unc, expPre - expPre_unc, color='g', alpha=.35)
	ax.plot(princ_coord, expPost, color='r', linewidth = 3, label = r'\( \eta_{'+ str(orderOfMom) + ',\perp, >} \)')
	
	ax.fill_between(princ_coord, expPost + expPost_unc, expPost - expPost_unc, color='r', alpha=.35)
	ax.set_ylabel(r'\( \eta_{'+ str(orderOfMom) + ',\perp} \)', fontsize = 70)
	ax.legend(loc = 'lower left', fontsize = 40)
	ax.tick_params(axis='y')
	ax.tick_params(axis='x')
	ax.set_xlim([min(princ_coord),max(princ_coord)])
	ax.set_xlabel(r'Pore point ($\perp$-coord.)')

	ind_min_PC_PD = np.where(PC_pre == min(PC_pre))[0]
	princCoord_min_PC_PD = princ_coord[ind_min_PC_PD]

	ax.yaxis.labelpad=0.3
	ax.xaxis.labelpad=0.2
	# ax.set_yticks([-10, -5, 0, 3.5, 5, 10])
	# ax.set_xticks([-14, princCoord_min_PC_PD[0], 8])
	# ax.set_xticklabels(['AG', 'CC', 'SF'],  fontsize=70)
	ax.axhline(y = 3.5, color = "g", alpha = 0.5, linewidth = 3, linestyle = "--")
	ax.axhline(y = 0, color = "k", linestyle = "--", alpha = 0.3, linewidth = 3)
	ax.set_ylim([-12,12])
	ax.axvspan(12.3, 22.3, alpha=0.1, color='m')
	ax.axvspan(-20.2, -10.12, alpha=0.1, color='m')
	
	twin_ax = ax.twinx()
	
	poreRadius_norm = poreRadius / max(poreRadius)
	
	twin_ax.plot(princ_coord, nu, 'm-', linewidth = 3, label = r'\( \nu \)')
	# twin_ax.plot(princ_coord, RadialProfile(princ_coord, entropy, 150), 'g-', linewidth = 3, label = r'\( \partial \mathcal{S} / \partial \mathrm{p}_\perp \)')
	# Convert None values to NaN
	nu_unc = np.array(nu_unc, dtype=np.float64)  # Ensures all values are numeric
	# Filter valid indices
	inds_no_nones = np.where(~np.isnan(nu_unc))[0]
	twin_ax.fill_between(princ_coord[inds_no_nones], nu[inds_no_nones] - nu_unc[inds_no_nones], nu[inds_no_nones] + nu_unc[inds_no_nones], color = 'm', alpha = 0.2)
	twin_ax.plot(princ_coord, (PC_pre), 'g--', linewidth = 1, alpha = 0.75, label = r'\(  \mathrm{PC}_{1,\perp,<} \)')
	twin_ax.plot(princ_coord, (PC_post), 'r--', linewidth = 1, alpha = 0.75, label = r'\(  \mathrm{PC}_{1,\perp,>}  \)')
	twin_ax.plot(princ_coord, poreRadius_norm, 'k', linewidth = 3, label = r'\( R / \mathrm{max}(R) \)')
	twin_ax.fill_between(princ_coord, poreRadius_norm - poreRadius_sd / max(poreRadius), poreRadius_norm + poreRadius_sd / max(poreRadius), alpha = 0.1, color = 'k')
	# twin_ax.plot(princ_coord, PC_pre,  'g--', alpha = 0.5, label = r'PC$^{(' + str(orderOfMom) +')}_{z,\mathrm{pre}}$')
	# twin_ax.plot(princ_coord, PC_post, 'r--', alpha = 0.5, label = r'PC$^{(' + str(orderOfMom) +')}_{z,\mathrm{post}}$')
	# twin_ax.plot(princ_coord, 0.5 * (PC_post + PC_pre), 'm--', alpha = 0.5, label = r'PC$^{(' + str(orderOfMom) +')}_{z,\mathrm{pre, post}}$')
	twin_ax.set_ylabel(r'\( \mathrm{PC}  \), \( R / \mathrm{max}(R) \), \( \nu \)', fontsize = 70)
	twin_ax.legend(loc = 'lower right', fontsize = 33)
	twin_ax.tick_params(axis='y')
	
	twin_ax.axvline(x=princCoord_min_PC_PD, color = "k", linestyle = "--", alpha = 0.3, linewidth = 3)
	ax.text(princCoord_min_PC_PD - 1, -1, '(pseudo)symm. axis', rotation=90, fontsize=35, alpha = 0.5, ha='center', va='center')
	# offset_y = 0.04
	# twin_ax.text(0.46 * (princ_coord[min_SF] + princ_coord[max_SF]), offset_y, 'SF', color='k', fontsize = 38, alpha = 0.7,bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	twin_ax.set_ylim([0,1.01])
	twin_ax.yaxis.labelpad=1

	plt.show()

## Some key traces ##
def Plot_CheckTraces(princ_coord, R, L, l_i, l_half, zeta, xi, lag, asy, l_max, l_min, PDB):

	plt.title(PDB)
	plt.plot(princ_coord, R, 'k-')
	plt.plot(princ_coord, L, 'k-')
	plt.plot(princ_coord, l_i, 'm', linewidth = 5)
	plt.plot(princ_coord, l_half, 'g-', linewidth = 5)
	plt.plot(princ_coord, zeta, 'g', label = r'$\zeta$')
	plt.plot(princ_coord, xi, 'g--', label = r'$\xi$')
	plt.plot(princ_coord, lag, 'r-')
	plt.plot(princ_coord, asy, 'b-')
	plt.plot(princ_coord, l_max, 'r--')
	plt.plot(princ_coord, l_min, 'b--')
	plt.legend(fontsize = 20)
	
	plt.savefig(PDB + '_sanityTest.pdf')
	plt.close()

## 2D plot (map) of the atomic structure ##
def Plot_MoleculeViews(ind_pp_axis, axis_ori, atom_coords, cumulativeAtomProbs, l_i, PDB):

	if (ind_pp_axis == 2):
		atom_coords_ = atom_coords[:,0]
		atom_coords__ = atom_coords[:,1]
		atom_coords_pp_axis = axis_ori*atom_coords[:,ind_pp_axis]	
	if (ind_pp_axis == 1):
		atom_coords_ = atom_coords[:,0]
		atom_coords__ = atom_coords[:,2]
		atom_coords_pp_axis = axis_ori*atom_coords[:,ind_pp_axis]
	if (ind_pp_axis == 0):
		atom_coords_ = atom_coords[:,1]
		atom_coords__ = atom_coords[:,2]
		atom_coords_pp_axis = axis_ori*atom_coords[:,ind_pp_axis]
	
	
	ax = plt.gca()
	ax.set_title(PDB)
	ax.scatter(atom_coords_, atom_coords_pp_axis, edgecolor = None, c = cumulativeAtomProbs, cmap = 'coolwarm')
	plt.savefig(PDB + '_sideView.pdf')
	plt.close()

	ax = plt.gca()
	ax.set_title(PDB)
	circle_l_i = plt.Circle((0,0),np.median(l_i), fill=False, color = 'm', linewidth = 5, alpha = 0.2)
	ax.text(np.median(l_i), np.median(l_i), '$\tilde{l_{i}}$', fontsize=32, color='m', alpha = 0.8)
	ax.add_patch(circle_l_i)
	ax.scatter(atom_coords_, atom_coords__, edgecolor = None, c = cumulativeAtomProbs, cmap = 'coolwarm')
	plt.savefig(PDB + '_topView.pdf')
	plt.close()

	RewriteBFactors(PDB + '_clean_H_ori.pdb', PDB + '_probBFactors.pdb', cumulativeAtomProbs)

##	Plot Distance from the inflection point ##
def Plot_DistanceFromInflectionPoints(princ_coord, data, all_data, entr_conf_max_model, entr_conf_max_emp, poreRad, poreRad_std, title, PLOT_ENTROPY_LINE = False):
	
	mean = np.ma.masked_equal(GetColumn(data, [0]), ModelParameters.MASK_VAL)
	median = np.ma.masked_equal(GetColumn(data, [2]), ModelParameters.MASK_VAL)
	std = np.ma.masked_equal(GetColumn(data, [1]), ModelParameters.MASK_VAL)
	left_perc = np.ma.masked_equal(GetColumn(data, [3]), ModelParameters.MASK_VAL)
	right_perc = np.ma.masked_equal(GetColumn(data, [4]), ModelParameters.MASK_VAL)
	mode = np.ma.masked_equal(GetColumn(data, [5]), ModelParameters.MASK_VAL)
	minval = np.ma.masked_equal(GetColumn(data, [6]), ModelParameters.MASK_VAL)
	maxval = np.ma.masked_equal(GetColumn(data, [7]), ModelParameters.MASK_VAL)
	
	fig, ax1 = plt.subplots()

	ax1.set_title(title, fontsize = 60)
	
	ax1.plot(princ_coord, mode, 'k--', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{mode} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	ax1.plot(princ_coord, median, 'r--', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{med} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	ax1.plot(princ_coord, mean, 'b--', alpha = 0.5, linewidth = 3.5, label = r"\( \langle \{ l_{\mathrm{mut}} - l_{i} \} \rangle \)")
	# ax1.fill_between(princ_coord, mean - std, mean + std, alpha = 0.2, color = 'r')
	
	if (PLOT_ENTROPY_LINE == True):
		ax1.plot(princ_coord, entr_conf_max_model, 'g-', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (theory) ")
		ax1.plot(princ_coord, entr_conf_max_emp, 'g--', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (empir.)")
	
	# ax1.legend(loc = 'upper left', fontsize = 50)
	# ax1.set_ylabel(r'\( l_{\mathrm{mut}} - l_{i} \) [\AA]', fontsize = 65)
	# ax1.set_xlabel(r'Pore point ($\perp$-coord.)', fontsize = 55)
	ax1.axhline(y = 0, linestyle = "-", color = 'b', alpha = 0.25, linewidth = 3)
	ax1.axhline(y = 0, linestyle = "-", color = 'r', alpha = 0.25, linewidth = 3)
	ax1.set_ylim([-30, 30])
	ax1.set_xticks([])
	# ax1.set_xticks([-14, -3, 8])
	# ax1.set_xticklabels(['AG', 'CC', 'SF'],  fontsize=70)
	# print(princ_coord[np.argmin(median)]-5, princ_coord[np.argmin(median)]+5)
	
	ax1.axvspan(princ_coord[np.argmin(median)]-5, princ_coord[np.argmin(median)]+5, alpha=0.1, color='m')
	ax1.legend(loc='center left', bbox_to_anchor=(-1.0, 0.1), fontsize = 48)
	
	ax2 = ax1.twinx()
	poreRad = poreRad/max(poreRad)
	ax2.plot(princ_coord, poreRad, 'k', linewidth = 2.5, label = r'$R / \mathrm{max}(R)$')
	ax2.fill_between(princ_coord, poreRad - 0.5 * poreRad_std / max(poreRad), poreRad +  0.5 * poreRad_std / max(poreRad), alpha = 0.15, color = 'k')
	ax2.set_xlim([min(princ_coord), max(princ_coord)])

	# ax2.text(-19, 0.06, 'AG', color='k',alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(-3, 0.06, 'CC', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(8, 0.06, 'SF', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	ax2.set_ylim([0,1])
	
	# ax2.legend(loc = 'lower right', fontsize = 40)
	ax2.set_ylabel(r'Pore radius (norm.)', fontsize = 65)

	inset_ax = fig.add_axes([0.35, 0.645, 0.22, 0.22]) 
	flattened_data = [item for sublist in all_data for item in sublist]
	counts, bins, _  = inset_ax.hist(flattened_data, bins = 30, edgecolor = 'k', alpha = 0.5, color = "red", label = r"$l_{\mathrm{mut}} - l_{i}$")
	mu, std = norm.fit(flattened_data)
	print(f"Estimated Mean: {mu}, Estimated Std Dev: {std}")
	x = np.linspace(-60, 60, 1000)
	bin_centers = 0.5 * (bins[1:] + bins[:-1])
	p = norm.pdf(x, mu, std) * np.mean(np.diff(bin_centers)) * counts.sum()
	inset_ax.plot(x, p, 'k--', linewidth=5, alpha = 0.5)
	inset_ax.axvline(x = 0, linewidth = 4, linestyle = "--", color="k", alpha = 0.5)
	inset_ax.set_ylabel("Freq.")
	# inset_ax.legend(loc = 'upper left', fontsize = 28, bbox_to_anchor=(0.6, 0.98))
	inset_ax.set_xlim([-85, 85])

	plt.show()

	factor = 0.75
	plt.plot(princ_coord, mode, 'k--', alpha = 0.5, linewidth = 3.5, label = r"LoF")
	# ax1.plot(princ_coord, median_LoF, 'bo', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{med} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	# ax1.plot(princ_coord, mean_LoF, 'b*', alpha = 0.5, linewidth = 3.5, label = r"\( \langle \{ l_{\mathrm{mut}} - l_{i} \} \rangle \)")
	plt.fill_between(princ_coord, minval + factor*(mode - minval), maxval - factor*(maxval - mode), alpha = 0.2, color = 'b')

	plt.show()


##	Plot Distance from the inflection point ##
def Plot_DistanceFromInflectionPoints_SCNXA(princ_coord, data_GoF, all_data_GoF, entr_conf_max_model_GoF, entr_conf_max_emp_GoF, 
											   				data_LoF, all_data_LoF, entr_conf_max_model_LoF, entr_conf_max_emp_LoF,
															nu, d_f_i,	
											   				poreRad, poreRad_std, title, PLOT_ENTROPY_LINE = False, factor = 1):
	
	## GoF data set (color: red)
	mean_GoF = np.ma.masked_equal(GetColumn(data_GoF, [0]), ModelParameters.MASK_VAL)
	median_GoF = np.ma.masked_equal(GetColumn(data_GoF, [2]), ModelParameters.MASK_VAL)
	std_GoF = np.ma.masked_equal(GetColumn(data_GoF, [1]), ModelParameters.MASK_VAL)
	left_perc_GoF = np.ma.masked_equal(GetColumn(data_GoF, [3]), ModelParameters.MASK_VAL)
	right_perc_GoF = np.ma.masked_equal(GetColumn(data_GoF, [4]), ModelParameters.MASK_VAL)
	mode_GoF = np.ma.masked_equal(GetColumn(data_GoF, [5]), ModelParameters.MASK_VAL)
	min_GoF = np.ma.masked_equal(GetColumn(data_GoF, [6]), ModelParameters.MASK_VAL)
	max_GoF = np.ma.masked_equal(GetColumn(data_GoF, [7]), ModelParameters.MASK_VAL)
	
	fig, ax1 = plt.subplots()

	ax1.set_title(title, fontsize = 60)
	
	ax1.plot(princ_coord, mode_GoF, 'r--', alpha = 0.5, linewidth = 3.5, label = r"GoF")
	# ax1.plot(princ_coord, median_GoF, 'ro', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{med} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	# ax1.plot(princ_coord, mean_GoF, 'r*', alpha = 0.5, linewidth = 3.5, label = r"\( \langle \{ l_{\mathrm{mut}} - l_{i} \} \rangle \)")
	ax1.fill_between(princ_coord, min_GoF + factor*(mode_GoF - min_GoF), max_GoF - factor*(max_GoF - mode_GoF), alpha = 0.2, color = 'r')
	
	if (PLOT_ENTROPY_LINE == True):
		ax1.plot(princ_coord, entr_conf_max_model_GoF, 'g-', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (theory) ")
		ax1.plot(princ_coord, entr_conf_max_emp_GoF, 'g--', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (empir.)")
	
	# ax1.legend(loc = 'upper left', fontsize = 50)
	# ax1.set_ylabel(r'\( l_{\mathrm{mut}} - l_{i} \) [\AA]', fontsize = 65)
	# ax1.set_xlabel(r'Pore point ($\perp$-coord.)', fontsize = 55)
	ax1.axhline(y = 0, linestyle = "-", color = 'b', alpha = 0.25, linewidth = 3)
	ax1.axhline(y = 0, linestyle = "-", color = 'r', alpha = 0.25, linewidth = 3)
	# ax1.set_ylim([-30, 30])
	ax1.set_xticks([])
	# ax1.set_xticks([-14, -3, 8])
	# ax1.set_xticklabels(['AG', 'CC', 'SF'],  fontsize=70)
	# print(princ_coord[np.argmin(median)]-5, princ_coord[np.argmin(median)]+5)
	
	ax2 = ax1.twinx()
	poreRad = poreRad / max(poreRad)
	ax2.plot(princ_coord, poreRad, 'k', linewidth = 2.5, label = r'$R / \mathrm{max}(R)$')
	ax2.fill_between(princ_coord, poreRad - 0.5 * poreRad_std / max(poreRad), poreRad +  0.5 * poreRad_std / max(poreRad), alpha = 0.15, color = 'k')
	ax2.set_xlim([min(princ_coord), max(princ_coord)])

	# ax2.text(-19, 0.06, 'AG', color='k',alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(-3, 0.06, 'CC', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(8, 0.06, 'SF', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	ax2.set_ylim([0,1])
	
	# surfRough = d_f_i  - min(d_f_i)
	# ax2.plot(princ_coord, nu, "g")
	# ax2.plot(princ_coord, surfRough/max(surfRough), "m")

	# ax2.legend(loc = 'lower right', fontsize = 40)
	ax2.set_ylabel(r'Pore radius (norm.)', fontsize = 65)

	# inset_ax = fig.add_axes([0.15, 0.645, 0.22, 0.22]) 
	# flattened_data = [item for sublist in all_data_GoF for item in sublist]
	# counts, bins, _  = inset_ax.hist(flattened_data, bins = 30, edgecolor = 'k', alpha = 0.25, color = "r", label = r"$l_{\mathrm{mut}} - l_{i}$")
	# mu, std = norm.fit(flattened_data)
	# print(f"Estimated Mean: {mu}, Estimated Std Dev: {std}")
	# x = np.linspace(-60, 60, 1000)
	# bin_centers = 0.5 * (bins[1:] + bins[:-1])
	# p = norm.pdf(x, mu, std) * np.mean(np.diff(bin_centers)) * counts.sum()
	# inset_ax.plot(x, p, 'k--', linewidth=5, alpha = 0.5)
	# inset_ax.axvline(x = 0, linewidth = 4, linestyle = "--", color="k", alpha = 0.5)
	# # inset_ax.set_ylabel("Freq.")
	# # inset_ax.legend(loc = 'upper left', fontsize = 28, bbox_to_anchor=(0.6, 0.98))
	# inset_ax.set_xlim([-85, 85])


	## LoF data set (color: red)
	mean_LoF = np.ma.masked_equal(GetColumn(data_LoF, [0]), ModelParameters.MASK_VAL)
	median_LoF = np.ma.masked_equal(GetColumn(data_LoF, [2]), ModelParameters.MASK_VAL)
	std_LoF = np.ma.masked_equal(GetColumn(data_LoF, [1]), ModelParameters.MASK_VAL)
	left_perc_LoF = np.ma.masked_equal(GetColumn(data_LoF, [3]), ModelParameters.MASK_VAL)
	right_perc_LoF = np.ma.masked_equal(GetColumn(data_LoF, [4]), ModelParameters.MASK_VAL)
	mode_LoF = np.ma.masked_equal(GetColumn(data_LoF, [5]), ModelParameters.MASK_VAL)
	min_LoF = np.ma.masked_equal(GetColumn(data_LoF, [6]), ModelParameters.MASK_VAL)
	max_LoF = np.ma.masked_equal(GetColumn(data_LoF, [7]), ModelParameters.MASK_VAL)
	
	ax1.plot(princ_coord, mode_LoF, 'b--', alpha = 0.5, linewidth = 3.5, label = r"LoF")
	# ax1.plot(princ_coord, median_LoF, 'bo', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{med} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	# ax1.plot(princ_coord, mean_LoF, 'b*', alpha = 0.5, linewidth = 3.5, label = r"\( \langle \{ l_{\mathrm{mut}} - l_{i} \} \rangle \)")
	ax1.fill_between(princ_coord, min_LoF + factor*(mode_LoF - min_LoF), max_LoF - factor*(max_LoF - mode_LoF), alpha = 0.2, color = 'b')


	# ind_neg = np.where(princ_coord < 0)[0]
	# ax1.axvspan(princ_coord[np.argmax(abs(mode_GoF[ind_neg] - mode_LoF[ind_neg]))] - 5, princ_coord[np.argmax(abs(mode_GoF[ind_neg] - mode_LoF[ind_neg]))] + 5, alpha=0.1, color='r')
	
	# ind_pos = np.where(princ_coord >= 0)[0]
	# ax1.axvspan(princ_coord[np.argmax(abs(mode_GoF[ind_pos] - mode_LoF[ind_pos]))] - 5, princ_coord[np.argmax(abs(mode_GoF[ind_pos] - mode_LoF[ind_pos]))] + 5, alpha=0.1, color='b')
	
	
	ax1.legend(loc='upper left', fontsize = 48)
	
	# flattened_data = [item for sublist in all_data_LoF for item in sublist]
	# counts, bins, _  = inset_ax.hist(flattened_data, bins = 30, edgecolor = 'k', alpha = 0.25, color = "b", label = r"$l_{\mathrm{mut}} - l_{i}$")
	# mu, std = norm.fit(flattened_data)
	# print(f"Estimated Mean: {mu}, Estimated Std Dev: {std}")
	# x = np.linspace(-60, 60, 1000)
	# bin_centers = 0.5 * (bins[1:] + bins[:-1])
	# p = norm.pdf(x, mu, std) * np.mean(np.diff(bin_centers)) * counts.sum()
	# inset_ax.plot(x, p, 'k--', linewidth=5, alpha = 0.5)
	# inset_ax.axvline(x = 0, linewidth = 4, linestyle = "--", color="k", alpha = 0.5)
	# inset_ax.set_ylabel("Freq.")
	# # inset_ax.legend(loc = 'upper left', fontsize = 28, bbox_to_anchor=(0.6, 0.98))
	# inset_ax.set_xlim([-85, 85])




	plt.show()



## Plot the statistical summary of a feature
def Plot_FeatureStatistic(princ_coord, data_class_0, data_class_1, poreRad, poreRad_std, label_0, label_1, statMethod, index_name):

	# Entropy
	entropies = LoadFile('entropies')
	entropy = GetColumn(entropies, [0])
	entropy_norm = entropy / max(entropy)
	inds_max = np.where(entropy_norm > ModelParameters.MAX_ENT_CUT_OFF)[0]
	inds_min = np.argsort(entropy_norm)[:len(inds_max)]

	if (statMethod == "MEDIAN"):

		y0_mean = np.ma.masked_equal(GetColumn(data_class_0, [0]), ModelParameters.MASK_VAL)
		y0 = np.ma.masked_equal(GetColumn(data_class_0, [2]), ModelParameters.MASK_VAL)
		y0_mode = np.ma.masked_equal(GetColumn(data_class_0, [5]), ModelParameters.MASK_VAL)
		l0 = np.ma.masked_equal(GetColumn(data_class_0, [3]), ModelParameters.MASK_VAL)
		r0 = np.ma.masked_equal(GetColumn(data_class_0, [4]), ModelParameters.MASK_VAL)
	
		y1_mean = np.ma.masked_equal(GetColumn(data_class_1, [0]), ModelParameters.MASK_VAL)
		y1 = np.ma.masked_equal(GetColumn(data_class_1, [2]), ModelParameters.MASK_VAL)
		y1_mode = np.ma.masked_equal(GetColumn(data_class_1, [5]), ModelParameters.MASK_VAL)
		l1 = np.ma.masked_equal(GetColumn(data_class_1, [3]), ModelParameters.MASK_VAL)
		r1 = np.ma.masked_equal(GetColumn(data_class_1, [4]), ModelParameters.MASK_VAL)

	if (statMethod == "MODE"):
		
		y0 = np.ma.masked_equal(GetColumn(data_class_0, [5]), ModelParameters.MASK_VAL)
		l0 = np.ma.masked_equal(GetColumn(data_class_0, [6]), ModelParameters.MASK_VAL)
		r0 = np.ma.masked_equal(GetColumn(data_class_0, [7]), ModelParameters.MASK_VAL)
	
		y1 = np.ma.masked_equal(GetColumn(data_class_1, [5]), ModelParameters.MASK_VAL)
		l1 = np.ma.masked_equal(GetColumn(data_class_1, [6]), ModelParameters.MASK_VAL)
		r1 = np.ma.masked_equal(GetColumn(data_class_1, [7]), ModelParameters.MASK_VAL)

	fig, ax1 = plt.subplots()

	# Plot class_0
	ax1.plot(princ_coord, y0, 'r', alpha = 0.5, linewidth = 3, label = label_0)
	# ax1.plot(princ_coord, y0_mean, 'r*', alpha = 0.5, linewidth = 3, label = r"$\mathrm{m}(l - l_i)$ (GoF)") # label = label_0)
	# ax1.plot(princ_coord, y0_mode, 'r--', alpha = 0.5, linewidth = 3, label = r"$\mathrm{m}(l - l_i)$ (GoF)") # label = label_0)
	ax1.fill_between(princ_coord, l0, r0, alpha = 0.2, color = 'r')
	ax1.axvspan(princ_coord[min(inds_min)],princ_coord[max(inds_min)], alpha = 0.13, color = "g")
	ax1.axvspan(princ_coord[min(inds_max)],princ_coord[max(inds_max)], alpha = 0.13, color = "g")
	# ax1.plot(princ_coord, mean_class_0, 'r--', label = label_0)

	# Plot class_1
	ax1.plot(princ_coord, y1, 'b-', alpha = 0.5, linewidth = 3, label = label_1)
	# ax1.plot(princ_coord, y1_mean, 'b*', alpha = 0.5, linewidth = 3, label = r"$\mathrm{m}(l - l_i)$ (LoF)") # label = label_1)
	# ax1.plot(princ_coord, y1_mode, 'b--', alpha = 0.5, linewidth = 3, label = r"$\mathrm{m}(l - l_i)$ (LoF)") # label = label_1)
	ax1.fill_between(princ_coord, l1, r1, alpha = 0.2, color = 'b')
	# ax1.plot(princ_coord, mean_class_1, 'b--', label = label_0)	

	# ax1.text(-1.2, 8, 'VSDs', rotation=90, fontsize=50, alpha = 0.5, ha='center', va='center')
	# ax1.text(-1.2, -18, 'PD', rotation=90, fontsize=50, alpha = 0.5, ha='center', va='center')

	ax1.legend(loc = 'upper center', fontsize = 50)
	ax1.set_ylabel(index_name, fontsize = 50)
	ax1.set_xlabel(r'Pore point ($\perp$-coord.)', fontsize = 55)
	ax1.axhline(y = 0, linestyle = "--", color = 'grey', alpha = 0.75, linewidth = 3)
	ax1.set_xlim([min(princ_coord), max(princ_coord)])
	# ax1.set_ylim([-2.3, 1])
	ax1.axvline(x = 0, linestyle = "--", color = 'grey', alpha = 0.75, linewidth = 3)

	# ax2 = ax1.twinx()
	# poreRad = poreRad / max(poreRad)
	# ind_max = np.argmax(poreRad)
	# ax2.plot(princ_coord, poreRad, 'k', label = r'$R/\max\{R\}$', linewidth = 3)
	# ax2.fill_between(princ_coord, poreRad - 0.5 * poreRad_std / max(poreRad), poreRad +  0.5 * poreRad_std / max(poreRad), alpha = 0.15, color = 'k')
	# ax2.set_xlim([min(princ_coord), max(princ_coord)])
	# ax2.axvline(x = 0, linestyle = "--", color = 'grey', alpha = 0.75, linewidth = 3)
	# # ax2.text(-19, 0.06, 'AG', color='k',alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# # ax2.text(-3, 0.06, 'CC', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# # ax2.text(8, 0.06, 'SF', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.set_ylim([0,1])
	
	# ax2.legend(loc = 'lower right', fontsize = 50)
	# ax2.set_ylabel(r'Pore radius (norm.)', fontsize = 65)

	plt.show()

##	Plot the medians of the medians of different features for different mutation subsets ##
def Plot_MediansOfFeatureMedias(princ_coord, medians_subclasses, poreRad, poreRad_std, 
								medians_class1 = [], percentiles = [],  medians_unseen = [], medians_missclass = [], 
								label_class1 = 'Neutr.', label_unseen1 = 'Benign', label_missclass = 'missclass.', 
								y_label = 'Cluster inertia', SHOW_LEGENG = False, perp_coord_color = []):

	# If you provide your own percentiles ..
	if (len(percentiles) != 0):
		leftPerc_subclassA = percentiles[0][0]
		rightPerc_subclassA = percentiles[1][0]

		leftPerc_subclassB = percentiles[2][0]
		rightPerc_subclassB = percentiles[3][0]

		leftPerc_claas1 = percentiles[4][0]
		rightPerc_claas1 = percentiles[5][0]

		leftPerc_unseen1 = percentiles[6][0]
		rightPerc_unseen1 = percentiles[7][0]

		leftPerc_missclass = percentiles[8][0]
		rightPerc_missclass = percentiles[9][0]

	fig, ax1 = plt.subplots()
	# subclass A (GoF)
	ax1.plot(princ_coord, np.median(medians_subclasses[0], axis = 0), color = "r", linewidth = 3, label = "GoF")
	ax1.plot(princ_coord, np.mean(medians_subclasses[0], axis = 0), color = "r", linewidth = 3, linestyle = "--")
	if (len(percentiles) == 0):
		ax1.fill_between(princ_coord, np.min(medians_subclasses[0], axis = 0), np.max(medians_subclasses[0], axis = 0), color = "red", alpha = 0.2)
	else:
		ax1.fill_between(princ_coord, leftPerc_subclassA, rightPerc_subclassA, color = "red", alpha = 0.2)
	# subclass B (LoF)
	ax1.plot(princ_coord, np.median(medians_subclasses[1], axis = 0), color = "m", linewidth = 3, label = "LoF")
	ax1.plot(princ_coord, np.mean(medians_subclasses[1], axis = 0), color = "m", linewidth = 3, linestyle = "--")
	if (len(percentiles) == 0):
		ax1.fill_between(princ_coord, np.min(medians_subclasses[1], axis = 0), np.max(medians_subclasses[1], axis = 0), color = "m", alpha = 0.2)
	else:
		ax1.fill_between(princ_coord, leftPerc_subclassB, rightPerc_subclassB, color = "m", alpha = 0.2)

	# class 1
	if (len(medians_class1) != 0):
		ax1.plot(princ_coord, np.median(medians_class1, axis = 0), color = "blue", linewidth = 3, label = label_class1)
		if (len(percentiles) == 0):
			ax1.fill_between(princ_coord, np.min(medians_class1, axis = 0), np.max(medians_class1, axis = 0), color = "blue", alpha = 0.2)
		else:
			ax1.fill_between(princ_coord, leftPerc_claas1, rightPerc_claas1, color = "blue", alpha = 0.2)
	
	# unseen  
	if (len(medians_unseen) != 0):
		ax1.plot(princ_coord, np.median(medians_unseen, axis = 0), color = "blue", linewidth = 3, linestyle = "--", label = label_unseen1)
		if (len(percentiles) == 0):
			ax1.fill_between(princ_coord, np.min(medians_unseen, axis = 0), np.max(medians_unseen, axis = 0), color = "blue", alpha = 0.1)
		else:
			ax1.fill_between(princ_coord, leftPerc_unseen1, rightPerc_unseen1, color = "blue", alpha = 0.1)

	# missclassified 
	if (len(medians_missclass) != 0):
		ax1.plot(princ_coord, np.median(medians_missclass, axis = 0), color = "g", linestyle = "--", linewidth = 3, label = label_missclass)
		if (len(percentiles) == 0):
			ax1.fill_between(princ_coord, np.min(medians_missclass, axis = 0), np.max(medians_missclass, axis = 0), color = "g", alpha = 0.1)
		else:
			ax1.fill_between(princ_coord, leftPerc_missclass, rightPerc_missclass, color = "g", alpha = 0.1)

	if (SHOW_LEGENG == True):
		ax1.legend(loc='upper right', fontsize = 35)	
	ax1.set_xlim([min(princ_coord), max(princ_coord)])
	ax1.axhline(y = 0, color = 'k', alpha = 0.2, lw = 3)
	ax1.axvline(x=-3, color = 'k', alpha = 0.2, lw = 3)

	if (len(perp_coord_color) == 2):
		ax1.axvspan(perp_coord_color[0], perp_coord_color[1], color="y", alpha=0.3)
	
	ax1.set_ylabel(y_label, fontsize = 50)
	# ax1.set_ylim([y_lim_left, y_lim_right])

	ax1.set_xticks([-14, -3, 8])
	ax1.set_xticklabels(['AG', 'CC', 'SF'])
	ax1.tick_params(axis='x', labelsize=55)

	ax2 = ax1.twinx()
	
	ax2.plot(princ_coord, poreRad / max(poreRad), "k", label = r'$R / \mathrm{max}\{ R \}$')
	ax2.fill_between(princ_coord, (poreRad - poreRad_std) / max(poreRad), (poreRad + poreRad_std) / max(poreRad), alpha = 0.15, color = 'k')
	ax2.set_ylabel(r'Pore radius (norm.)', fontsize = 55)
	ax2.set_xticks([-14, -3, 8])
	ax2.set_xticklabels(['AG', 'CC', 'SF'])
	ax2.tick_params(axis='x', labelsize=55)
	if (SHOW_LEGENG == True):
		ax2.legend(loc = "lower right", fontsize = 40)

	plt.show()

## Plot Feature Importance ##
def Plot_FeaturesImportance(princ_coord, features_importance, poreRadius, labels):

	maxValues = np.zeros(len(labels))
	for i in range(len(labels)):	
		maxValues[i] = np.max(GetColumn(features_importance, [i,2])) 

	colors_list = ["blue", "red", "green", "yellow", "magenta", "black"] # ... add more colors if needed ...
	fig, ax = plt.subplots()	 
	for i in range(len(labels)):
		ax.plot(princ_coord, GetColumn(features_importance, [i,2]) / np.max(maxValues), color = colors_list[i], alpha = 0.5, linewidth = 3, label = labels[i])

	ax.legend(loc = 'upper center', fontsize = 55)

	# ax.set_xlabel(r'Pore point ($\perp$-coord.)',fontsize = 50)
	ax.set_ylabel(r'Feat. importance',fontsize = 55)
	ax.set_xlim([min(princ_coord),max(princ_coord)])
	ax.set_ylim([0,1.05])
	ax.set_xticks([-14, -3, 8])  # Set x-ticks at positions 1, 2, 3, 4, and 5
	# Optionally, set custom tick labels
	ax.set_xticklabels(['AG', 'CC', 'SF'])
	ax.tick_params(axis='x', labelsize=55)

	twin_ax = ax.twinx()
	poreRadius_norm = poreRadius / max(poreRadius)
	twin_ax.plot(princ_coord, poreRadius_norm, "k--", alpha = 0.5, linewidth = 3, label = "$R/\mathrm{max}\{R \}$")
	twin_ax.set_ylabel(r'Pore radius (norm.)', fontsize = 50)
	# twin_ax.legend(loc = 'upper right', fontsize = 50)
	twin_ax.tick_params(axis='y')
	twin_ax.set_ylim([0,1.05])
	twin_ax.set_xticks([-14, -3, 8])  # Set x-ticks at positions 1, 2, 3, 4, and 5
	# Optionally, set custom tick labels
	twin_ax.set_xticklabels(['AG', 'CC', 'SF'])
	twin_ax.tick_params(axis='x', labelsize=55)

	plt.show()

## Plot muation scoring (ensemble Learning results) ##
def Plot_MutationScores(	
		
							class0_probs, 
						
							resIDs, 
						
							n_class0, n_class1, 
							indsAppend_LoF, indsAppend_GoF, 
							
							optThres, NEUTRAL_IND = True, 
							
							class0_probs_unseen = [], 
							n_unseen = 0, 
							
						):

	data_in = []
	n_class = n_class0 + n_class1
	for i in range(n_class):
		data_in.append(GetColumn(class0_probs,[i]))
	
	if n_unseen:
		data_in_unseen = []
		for i in range(n_unseen):
			data_in_unseen.append(GetColumn(class0_probs_unseen,[i]))
	
	positions_class0 = np.arange(0, n_class0, 1) + 1	
	positions_unseen = np.arange(n_class0, n_unseen + n_class0, 1) + 1
	positions_class1 = np.arange(n_unseen + n_class0, n_unseen + n_class, 1) + 1 
	
	fig, ax = plt.subplots()

	if (len(indsAppend_LoF) > 0):
		data_LoF = []
		for i in range(len(indsAppend_LoF)):
			data_LoF.append(data_in[indsAppend_LoF[i]])
		# Pathogenic
		ax.boxplot(data_LoF, positions = positions_class0[indsAppend_LoF], 
            patch_artist=True,
            boxprops=dict(facecolor='magenta', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='magenta'),
            flierprops=dict(marker='o', color='black', markersize=8))
	
	if (len(indsAppend_GoF[0]) > 0):
		data_PEPD = []
		for i in range(len(indsAppend_GoF[0])):
			data_PEPD.append(data_in[indsAppend_GoF[0][i]])
		ax.boxplot(data_PEPD, positions = positions_class0[indsAppend_GoF[0]], 
            patch_artist=True,
            boxprops=dict(facecolor='orange', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='orange'),
            flierprops=dict(marker='o', color='black', markersize=8))
		
	if (len(indsAppend_GoF[1]) > 0):
		data_SFN = []
		for i in range(len(indsAppend_GoF[1])):
			data_SFN.append(data_in[indsAppend_GoF[1][i]])
		ax.boxplot(data_SFN, positions = positions_class0[indsAppend_GoF[1]], 
            patch_artist=True,
            boxprops=dict(facecolor='green', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='green'),
            flierprops=dict(marker='o', color='black', markersize=8))
		
	if (len(indsAppend_GoF[2]) > 0):
		data_IEM = []
		for i in range(len(indsAppend_GoF[2])):
			data_IEM.append(data_in[indsAppend_GoF[2][i]])
		print(positions_class0[indsAppend_GoF[2]])
		ax.boxplot(data_IEM, positions = positions_class0[indsAppend_GoF[2]], 
            patch_artist=True,
            boxprops=dict(facecolor='red', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red'),
            flierprops=dict(marker='o', color='black', markersize=8))

	
	if n_unseen:
		data_unseen = []
		for i in range(n_unseen):
			data_unseen.append(data_in_unseen[i])
		# Pathogenic
		print(positions_unseen)
		ax.boxplot(data_unseen, positions = positions_unseen, 
            patch_artist=True,
            boxprops=dict(facecolor='grey', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='grey'),
            flierprops=dict(marker='o', color='black', markersize=8))
		
	# Neutral/Benign
	if (NEUTRAL_IND == True):
		ax.boxplot(data_in[n_class0:n_class], positions = positions_class1, 
            patch_artist=True,
            boxprops=dict(facecolor='blue', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='blue'),
            flierprops=dict(marker='o', color='black', markersize=8))
	
	x_tickLabels = resIDs
	ax.axhspan(ymin = optThres[1], ymax = optThres[2], color = "k", alpha = 0.05)
	ax.axhline(y = np.round(optThres[0],2), color = "k", alpha = 0.3)
	
	ax.set_xticks(np.arange(1, n_class + n_unseen + 1))  # Correctly set positions only
	ax.set_xticklabels(x_tickLabels, rotation=90, fontsize=15)  # Set labels separately
	
	# Customize specific x-tick labels
	IEM_ticks = resIDs[indsAppend_GoF[2]]  
	SFN_ticks = resIDs[indsAppend_GoF[1]]  
	PEPD_ticks = resIDs[indsAppend_GoF[0]]  
	LoF_ticks = resIDs[indsAppend_LoF] 
	Neutral_ticks = resIDs[(n_class0 + n_unseen):(n_class + n_unseen)]  
	if n_unseen:
		unseen_ticks = resIDs[(n_class0):(n_class0 + n_unseen)]   

	for tick in ax.get_xticklabels():
		tick_value = tick.get_text()  
    	# Check if the current tick value is in the selected ticks
		if tick_value in IEM_ticks:
			tick.set_color("red")  # Set the color for selected ticks
		if tick_value in LoF_ticks:
			tick.set_color("m")  # Set the color for selected ticks
		if tick_value in PEPD_ticks:
			tick.set_color("orange")  # Set the color for selected ticks
		if tick_value in SFN_ticks:
			tick.set_color("g")  # Set the color for selected ticks
		if tick_value in SFN_ticks:
			tick.set_color("g")  # Set the color for selected ticks
		if tick_value in Neutral_ticks:
			tick.set_color("b")  # Set the color for selected ticks
		if n_unseen:
			if tick_value in unseen_ticks:
				tick.set_color("grey")  # Set the color for selected ticks
		if tick_value in ['Cys1719', 'Ser211', 'Leu172', 'Arg99', 'Arg1279', 'Ile739', 'Ile720', 'Thr1596', 'Trp1538']:
			tick.set_color("k")

	# print(optThres[0])
	ax.set_yticks(np.array([0.1,0.2,0.3, np.round(optThres[0],3), 0.5, 0.6,0.7,0.8,0.9,1.0]))
	ax.tick_params(axis='y', labelsize=25) 
	ax.plot([], [], color='red', label='IEM', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='green', label='SFN', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='orange', label='PEPD', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='magenta', label='LoF', marker='s', linestyle='none', linewidth = 2)
	ax.set_ylabel('Classification Summary', fontsize = 50)
	ax.legend(loc="lower left", fontsize = 32)
	
	plt.show()


def CalculateF1Threshold(y_true, prob_class_1, threshold):
	
	from sklearn.metrics import confusion_matrix
	
	y_pred = (prob_class_1 >= threshold).astype(int)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	return f1_score


def BootstrapThreshold(y_true, prob_class_1, n_iterations=1000):

	from sklearn.utils import resample		

	best_thresholds = []
	f1_scores = []
    
	for _ in range(n_iterations):
		# Generate a bootstrap sample
		X_resample, y_resample = resample(prob_class_1, y_true, random_state=None)
        
    	# Find the best threshold using the resampled data
		thresholds = np.linspace(0, 1, 101)
		f1_scores_iter = np.array([CalculateF1Threshold(y_resample, X_resample, threshold) for threshold in thresholds])
        
    	# Find the threshold with the maximum F1 score
		best_threshold_index = np.argmax(f1_scores_iter)
		best_threshold = thresholds[best_threshold_index]
        
		best_thresholds.append(best_threshold)
		f1_scores.append(f1_scores_iter[best_threshold_index])
    
	return best_thresholds, f1_scores



def SomeResidueStatistics(varInfo_res):

	from collections import Counter

	# List of amino acids to check
	res = ['ARG', 'MET', 'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
	# Count how many times each amino acid appears
	count = Counter(varInfo_res)
	# Check how many times each amino acid from the first list appears in the second list
	amino_acid_counts = {aa: count.get(aa, 0) for aa in res}
	
	# Hydropathic scores for each residue
	hydropathic_scores = {
    "PHE": -4.0, "PRO": -3.0, "ILE": -3.5, "LEU": -3.5, "TRP": -3.0,
    "VAL": -2.0, "TYR": -1.5, "MET": -1.0, "ALA": 1.0, "THR": 2.0,
    "GLY": 2.5, "CYS": 3.5, "SER": 3.5, "GLN": 5.0, "HIS": 6.0,
    "LYS": 5.0, "GLU": 6.0, "ASN": 6.5, "ASP": 7.5, "ARG": 14.5
	}

	# Normalize counts
	total_count = sum(amino_acid_counts.values())
	normalized_counts = {
    residue: count / total_count for residue, count in amino_acid_counts.items()
	}

	# Normalize hydropathic scores for coloring
	min_score = min(hydropathic_scores.values())
	max_score = max(hydropathic_scores.values())
	normalized_scores = {
    residue: (score - min_score) / (max_score - min_score)
    for residue, score in hydropathic_scores.items()
	}

	# Sort residues from most hydrophobic to most hydrophilic based on hydropathic scores
	sorted_residues = sorted(hydropathic_scores.keys(), key=lambda x: hydropathic_scores[x])

	# Prepare sorted data for plotting
	sorted_counts = [normalized_counts[residue] for residue in sorted_residues]
	sorted_colors = [
    plt.cm.coolwarm(normalized_scores[residue])
    for residue in sorted_residues
	]
	sorted_formatted_residues = [residue.capitalize() for residue in sorted_residues]

	# Plot the bar chart with sorted residues
	fig, ax = plt.subplots(figsize=(12, 8))
	bars = ax.bar(sorted_formatted_residues, sorted_counts, color=sorted_colors)
	plt.xlabel('Residue', fontsize = 50)
	plt.ylabel('Freq.', fontsize = 50)
	# plt.title('Kapcha$\&$ Rossky hydropathic scale')
	plt.xticks(rotation=90)

	# Add a color bar to indicate hydrophobic (blue) to hydrophilic (red)
	sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min_score, vmax=max_score))
	sm.set_array([])
	cbar = plt.colorbar(sm, ax = ax)
	cbar.set_label('Kapcha$\&$Rossky hydropathic scale', fontsize=35)  # Adjust fontsize here

	# Adjust font size of colorbar labels
	cbar.ax.tick_params(labelsize = 20)
	# cbar.ax.tick_params(labelsize=12)

	# Display the plot
	plt.tight_layout()
	plt.show()



'''

		Shifting contour map color pallete 

'''
def ShiftColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    
	from matplotlib import colors
    
	'''
	
		Function to offset the 'center' of a colormap. Useful for
		data with a negative min and positive max and you want the
		middle of the colormap's dynamic range to be at zero
	
		Input
		-----
		cmap : The matplotlib colormap to be altered
		start : Offset from lowest point in the colormap's range.
		Defaults to 0.0 (no lower ofset). Should be between
		0.0 and `midpoint`.
		midpoint : The new center of the colormap. Defaults to 
		0.5 (no shift). Should be between 0.0 and 1.0. In
		general, this should be  1 - vmax/(vmax + abs(vmin))
		
		For example if your data range from -15.0 to +5.0 and
		you want the center of the colormap at 0.0, `midpoint`
		should be set to  1 - 5/(5 + 15)) or 0.75 stop : Offset from highets point in the colormap's range.
		Defaults to 1.0 (no upper ofset). Should be between
		`midpoint` and 1.0.
	
	'''
	
	cdict = {'red':[],'green':[],'blue':[],'alpha':[]}
	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)
	# shifted index to match the data
	shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
	
	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)
		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))
	
	newcmap = colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap 
 

def Plot_ContourMap(princ_coord, scales, midPoint, innerRad, poreRad, outterRad, inflLength, lagLength, asyLength, field_desc, Delta_alpha = 1, Delta_z = 0.1):
	
	from matplotlib import cm
	from matplotlib import ticker

	X, Y = np.mgrid[
    1:(len(scales[0]) + 1):1,  # Δα = 1
    min(princ_coord):max(princ_coord):(len(princ_coord) * 1j)  # Ensures correct shape
	]

	print(f"X shape: {X.shape}, Y shape: {Y.shape}")
	
	for i in range(len(princ_coord)):
		X[:,i] = scales[i]
	
	field_desc = field_desc.transpose()
	

	# definitions for the axes (this needs to be adjusted to fit your screen!)
	left, width = 0.2, 0.725
	bottom, height = 0.2, 0.4

	rect_contour = [left, bottom, width, height]
	
	# start with a rectangular Figure
	plt.figure(1, figsize=(8, 8))
	axContour = plt.axes(rect_contour)
	
	# Contour plot
	fig1 = plt.figure(1) 
	
	axContour.contour(X, Y, field_desc, 1, linewidths = 0.5, colors='k') # topographic
	# color bar
	position = fig1.add_axes([0.32, 0.12, 0.38, 0.012])
	orig_cmap = cm.coolwarm
	shifted_cmap = ShiftColorMap(orig_cmap, midpoint=midPoint, name='shifted') # 0.53
	# contour map
	cc = axContour.pcolormesh(X, Y, field_desc, cmap = shifted_cmap)
	cb = fig1.colorbar(cc, cax=position, orientation='horizontal')
	ax = cb.ax
	# ax.text(4.5, -24.4, r'$\sim$ [kcal \AA /mol]', fontsize=12)
	tick_locator = ticker.MaxNLocator(nbins=5)
	cb.locator = tick_locator
	cb.update_ticks()
	ax.tick_params(labelsize=13) 
	# Geom. char of pore
	axContour.plot(innerRad, princ_coord, color='k', linestyle = '-', linewidth=1)
	axContour.plot(poreRad, princ_coord, color='k', linestyle = '-', linewidth=1)
	axContour.plot(outterRad, princ_coord, color='k', linestyle = '-', linewidth=1)
	axContour.plot(inflLength, princ_coord, color='m', linewidth=1.5, alpha = 0.25)
	axContour.plot(asyLength, princ_coord, color='y', linewidth=1.5, alpha = 0.5)
	axContour.plot(lagLength, princ_coord, color='g', linewidth=1.5, alpha = 0.5)
	
	axContour.set_ylabel(r'Pore point $\textbf{p}$ ($z$-coord.)', fontsize=16)
	# axContour.annotate(r'$D$',fontsize=16, xytext=(1.25, -22.), xy=(8.2, -23.0))
	# axContour.annotate(r'$\bar{R}(\textbf{p})$',fontsize=23, xytext=(-2, -2), xy=(5.4, 0.18), arrowprops=dict(arrowstyle='->'),)
	# axContour.annotate(r'$L$', fontsize=16, xytext=(72.5, -19), xy=(71.9, -22.))
	# axContour.annotate(r'$l$ [\AA]', fontsize=16, xytext=(2, -33), xy=(71.9, -22.))
	# axContour.annotate(r'$l_{PD}$',xy=(30.5, 1.9),color='m',fontsize=18)
	# axContour.annotate(r'$l_{asym}$',xy=(52., 1.9),color='y',fontsize=18)
	# axContour.annotate(r'$l_{lag}$',xy=(16., 1.9),color='g',fontsize=18)
	axContour.xaxis.set_label_coords(0.1, -0.075)
	axContour.yaxis.set_label_coords(-0.08, 0.6)
	# Label coords. 
	axContour.xaxis.set_label_coords(0.09, -0.078)
	axContour.yaxis.set_label_coords(-0.1, 0.6)
	# Limits
	axContour.set_ylim([-26.8, 27.1])
	# axContour.set_xlim([0,max(sysSizeSymm)])
	# Ticks
	axContour.set_yticks([0]) #([-17.9, -11.7,  -7.1,   2.7,  10.8,  22.7])
	axContour.set_xticks([10,20, 30, 40, 50, 60, 70, 80])
	axContour.tick_params(labelsize=14)
	# Mark the ES and the IS
	bbox_props = dict(boxstyle='square', fc='w', ec='0.2', alpha=0.9)
	axContour.text(-2.2,  max(princ_coord)-2, r'$ES$', ha='center', va='center', size=20, bbox=bbox_props)
	axContour.text(-2.2, min(princ_coord)+2, r'$IS$', ha='center', va='center', size=20, bbox=bbox_props)
	
	# Set some limits
	axContour.set_ylim([min(princ_coord), max(princ_coord)])
	axContour.set_xlim([0, max(outterRad)+1])
	# legend
	axContour.legend(loc = 'upper right',	fontsize=12)
	plt.savefig('figure1.pdf')
	plt.show()


"""

		weighted scores

"""
def Scoring(class0_probs, PERC = 0.5):

	# Initialize
	modes = np.zeros(len(class0_probs[0]))
	medians = np.zeros(len(class0_probs[0]))
	medians_r = np.zeros(len(class0_probs[0]))
	medians_l = np.zeros(len(class0_probs[0]))
	means = np.zeros(len(class0_probs[0]))

	# for the i-th residues compute global scores
	for i in range(len(class0_probs[0])):
	
		probs = GetColumn(class0_probs, [i])
		
		means[i] = np.mean(probs)
		modes[i] = Mode(probs)
		medians[i] = np.median(probs)
		# print(means[i],modes[i], medians[i])
		probs_sorted = np.sort(probs)
		ind_median = Match(medians[i], probs_sorted)
		# print(ind_median, probs_sorted, medians[i])
		L_r = len(np.where(probs_sorted >= medians[i])[0])
		L_l = len(np.where(probs_sorted < medians[i])[0])	
		ind_r_median = ind_median[0] + int(PERC * L_r)
		ind_l_median = ind_median[0] - int(PERC * L_l)
		medians_r[i] = probs_sorted[ind_r_median]
		medians_l[i] = probs_sorted[ind_l_median]

		# print(medians_l[i], medians[i], medians_r[i])
		
	# plt.title("Medians")
	# plt.hist(medians_l, color = "b", alpha = 0.2)	
	# plt.hist(medians, color = "r", alpha = 0.2)
	# plt.hist(medians_r, color = "g", alpha = 0.2)
	# plt.show()

	return [means, medians, modes, medians_r, medians_l]

"""

		Standarize trace

"""
def Standarization(y, l_i, l):
	ind_l_i = Match(l_i, l)
	y_norm = y/y[ind_l_i]
	return ( y_norm - min(y_norm) ) / ( max(y_norm) - min(y_norm) ) + min(abs(y_norm)) + ModelParameters.ZERO



def find_sublist_indices(big, small):
    n, m = len(big), len(small)
    return [i for i in range(n - m + 1) if big[i:i + m] == small]
"""

		Log-log representation and fitting in the interval (l, l_i)

"""
def LogLogSlopeRes(l, y, l_, ind_l_i, minNrOfPoints = 10):
	
	slope = ModelParameters.MASK_VAL 
	PC = ModelParameters.MASK_VAL  
	ind_l_ = np.argmin(abs(l - l_))

	# Window is measured in \AA: typically set to \xi
	if (l_ < l[ind_l_i]):
		y_fit = y[ind_l_:ind_l_i]
		if (len(y_fit) < int(minNrOfPoints)): # min nr of data points is 5
			ind_l_ = ind_l_ - (int(minNrOfPoints) - len(y_fit))
			y_fit = y[ind_l_:ind_l_i]
		x_fit = l[ind_l_:ind_l_i]
		coeffs, _ = np.polyfit(x_fit, y_fit, deg=1, cov=True)
		slope = coeffs[0]
		PC = stats.pearsonr(y_fit, slope*x_fit + coeffs[1])[0]
			
	else:
		y_fit = y[ind_l_i:ind_l_]
		if (len(y_fit) < int(minNrOfPoints)): # min nr of data points is 5
			ind_l_ = ind_l_ + (int(minNrOfPoints) - len(y_fit))
			y_fit = y[ind_l_i:ind_l_]
		x_fit = l[ind_l_i:ind_l_]
		coeffs, _ = np.polyfit(x_fit, y_fit, deg=1, cov=True)
		slope = coeffs[0]
		PC = stats.pearsonr(y_fit, slope*x_fit + coeffs[1])[0]
			
	# plt.plot(x_fit, y_fit, "o")
	# plt.plot(x_fit, x_fit*slope + coeffs[1], "--")
	# plt.show()

	return [slope, PC]


def Find_weights(*columns, alpha=1.0):
    # Convert inputs to numpy arrays
    columns = [np.array(col) for col in columns]
    A = np.column_stack(columns[:-1])  # All columns except the last one for A
    T = columns[-1]  # Last column is the target
    
    # Ridge regularization: A^T A + alpha * I
    I = np.eye(A.shape[1])
    weights = np.linalg.inv(A.T @ A + alpha * I) @ A.T @ T
    
    # Calculate weighted columns
    weighted_columns = A * weights
    
    # Calculate the final weighted sum
    final_weighted_column = np.sum(weighted_columns, axis=1)
    
    return weights, final_weighted_column


def ColorResidues(		residue_values, 
				  
				  		resId_class0 = [], resId_class1 = [], resId_interact = [], 
						
						SPECTRUM_COLOR = True,
						
						interval_id = None):


	import pymol
	pymol.finish_launching() 
	from pymol import cmd

	inds_noNone = np.where(residue_values[:,0] != None)[0]

	res_no = [int(float(res)) for res in residue_values[inds_noNone, 0]]
	vals = [float(val) for val in residue_values[inds_noNone, 1]]

	## If the residues are renum
	if (res_no[0] == 1 and max(res_no) == len(res_no)):
		
		fn_ending = "_clean_H_ori_renum.pdb"
		pdb_file = PDB + fn_ending

		pymol.finish_launching() 
		cmd.bg_color("white")

		cmd.load(pdb_file, "structure")
		
	else:

		fn_ending = "_clean_H_ori.pdb"
		pdb_file = PDB + fn_ending

		pymol.finish_launching() 
		cmd.bg_color("white")

		cmd.load(pdb_file, "structure")
		cmd.remove("chain B+C")

	cmd.remove("HET")
	
	
	for val in vals:
			cmd.alter("resi {}".format(res_no[vals.index(val)]), "b={}".format(val))

	if SPECTRUM_COLOR:
		cmd.spectrum("b", "blue_white_red", "name CA", minimum=min(vals), maximum=max(vals))
	else:
		cmd.spectrum("b", "lightgrey", "name CA", minimum=min(vals), maximum=max(vals))

	for i in range(len(vals)):
		if res_no[i] in resId_class0:
			cmd.show("sphere", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_scale", 0.8, f"resi {res_no[i]} and name CA")  # Adjust sphere size (small dot)
			cmd.color("green", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_transparency", 0.3, f"resi {res_no[i]} and name CA")  # Set transparency (0 = fully opaque, 1 = fully transparent)

		if res_no[i] in resId_class1:
			cmd.show("sphere", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_scale", 0.9, f"resi {res_no[i]} and name CA")  # Adjust sphere size (small dot)
			cmd.color("yellow", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_transparency", 0.3, f"resi {res_no[i]} and name CA")  # Set transparency (0 = fully opaque, 1 = fully transparent)

		if res_no[i] in resId_interact:
			cmd.show("sphere", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_scale", 0.7, f"resi {res_no[i]} and name CA")  # Adjust sphere size (small dot)
			cmd.color("cyan", f"resi {res_no[i]} and name CA")
			cmd.set("sphere_transparency", 0.1, f"resi {res_no[i]} and name CA")  # Set transparency (0 = fully opaque, 1 = fully transparent)
			
			
	## Define residue selections
	# cmd.select("res1", "chain A and resi %s" % resId_interact[0])
	# cmd.select("res2", "chain A and resi %s" % resId_interact[1])
	# cmd.select("res3", "chain A and resi %s" % resId_interact[2])
	# cmd.select("res4", "chain A and resi %s" % resId_interact[3])

	## Show sticks for clarity
	# cmd.show("sticks", "res1 or res2")
	# cmd.show("sticks", "res1 or res3")
	# cmd.show("sticks", "res2 or res3")
	# cmd.show("sticks", "res3 or res4")

	## Draw distance interactions (within 3.5 Å)
	# cmd.distance("contacts", "res1", "res2", 3.5)
	# cmd.distance("contacts", "res1", "res3", 3.5)
	# cmd.distance("contacts", "res2", "res3", 3.5)
	# cmd.distance("contacts", "res3", "res4", 3.5)

	## Show the structure in cartoon view
	cmd.show("cartoon", "structure")
	cmd.zoom("structure")
	# cmd.zoom("res1 or res2", buffer=50)

	# save the PDB file
	if (interval_id != None):
		print("saving ...")
		cmd.save(PDB + "_" + interval_id + ".pdb")

"""
	Insert pain-disease-associated variants
"""
def InsertPainVariants(source, resInfo):

	offset = source.cell(row = 4, column = 1).value 
	offsetID = source.cell(row = 6, column = 1).value 
		
	'''
		IEM
	'''			
	iem_id = np.array(VariantsListing(source['B']), 'int')
	iem_resType = VariantsListing(source['C'])
	# Introduce offset!
	inds = np.where(iem_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	iem_id[inds] = iem_id[inds.astype(int)] + offset
		
	'''
		SFN
	'''		
	sfn_id = np.array(VariantsListing(source['D']), 'int')
	sfn_resType = VariantsListing(source['E'])
	# Introduce offset!
	inds = np.where(sfn_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	sfn_id[inds] = sfn_id[inds.astype(int)] + offset

	'''
		PEPD
	'''
	pepd_id = np.array(VariantsListing(source['F']), 'int')
	pepd_resType = VariantsListing(source['G'])
	# Introduce offset!
	inds = np.where(pepd_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	pepd_id[inds] = pepd_id[inds.astype(int)] + offset
	
	'''
		NEUTRAL
	'''
	neutral_id = np.array(VariantsListing(source['H']), 'int')
	neutral_resType = VariantsListing(source['I'])
	# Introduce offset!
	inds = np.where(neutral_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset 
	neutral_id[inds] = neutral_id[inds.astype(int)] + offset
	
	'''
		LOF
	'''
	lof_id = np.array(VariantsListing(source['J']), 'int')
	lof_resType = VariantsListing(source['K'])
	# Introduce offset!
	inds = np.where(lof_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	lof_id[inds] = lof_id[inds.astype(int)] + offset

	'''
		MISSCLASS
	'''
	miss_id = np.array(VariantsListing(source['M']), 'int')
	miss_resType = VariantsListing(source['N'])
	# Introduce offset!
	inds = np.where(miss_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	miss_id[inds] = miss_id[inds.astype(int)] + offset

	'''
			NEWLY DISCOVERED
	'''
	ndisc_id = np.array(VariantsListing(source['O']), 'int')
	ndisc_resType = VariantsListing(source['P'])
	# Introduce offset!
	inds = np.where(ndisc_id > offsetID)[0]
	# normalize: if (list_id > offsetID) then pdb_id = list_id + offset
	ndisc_id[inds] = ndisc_id[inds.astype(int)] + offset

	varInfo = [[None for _ in range(4)] for _ in range(len(resInfo))] 
	i = 0
	for res in resInfo:
		resId = res[0]
		resType = res[1]
		varInfo[i][0] = resId
		varInfo[i][1] = 'not provided'
		varInfo[i][2] = resType
		varInfo[i][3] = 'classified'
		# First check if it is missclassified (or any other special category) ..
		if resId in miss_id:
			varInfo[i][3] = 'missclassified'
		# .. then, check all different phenotypes (if yes, increment i and move to the next iteration)
		if resId in iem_id:
			varInfo[i][1] = 'IEM'
			i += 1
			continue
		if resId in sfn_id:
			varInfo[i][1] = 'SFN'
			i += 1
			continue
		if resId in pepd_id:
			varInfo[i][1] = 'PEPD'
			i += 1
			continue
		if resId in lof_id:
			varInfo[i][1] = 'PEPD'
			i += 1
			continue
		if resId in ndisc_id:
			varInfo[i][1] = 'ndisc'
			i += 1
			continue

	# now save varInfo into a .txt file
	StoreFile(varInfo, 'varInfo_testing')


"""
	Insert gnomAD variants
"""
def InsertGnomADVariants(source_g,  resInfo):

	gnomad_id = np.array(VariantsListing(source_g['A']), 'int')
	element_counts = Counter(gnomad_id)
	# Filter elements that appear more than once
	duplicates = [item for item, count in element_counts.items() if count > 1]
	if (len(duplicates) > 0):
		print('.. There are list entries that appear more than once (gnomad): ', duplicates)
	gnomad_resType = VariantsListing(source_g['B'])
	gnomad_significance = VariantsListing(source_g['D']) 
	gnomad_indices = VariantsListing(source_g['E']) 


def DomainFeatures(domain, exponentsDomains_phobic, exponentsDomains_philic, intervals_phobic, intervals_philic, jth_order_even, FN_ENDING, inds_class0 = [], inds_class1 = []):


	# Interval lengths
	if (domain == "preInfl1"):
		philic_interval = GetColumn(intervals_philic, [0]) 
	if (domain == "infl1"):
		philic_interval = GetColumn(intervals_philic, [1])
	if (domain == "infl2"):
		philic_interval = GetColumn(intervals_philic, [2])
	if (domain == "postInfl2"):
		philic_interval = GetColumn(intervals_philic, [3])
	if (domain == "pre"):
		philic_interval = GetColumn(intervals_philic, [4])
	if (domain == "post"):
		philic_interval = GetColumn(intervals_philic, [5])
	
	# Interval lengths
	if (domain == "preInfl1"):
		phobic_interval = GetColumn(intervals_phobic, [0]) 
	if (domain == "infl1"):
		phobic_interval = GetColumn(intervals_phobic, [1])
	if (domain == "infl2"):
		phobic_interval = GetColumn(intervals_phobic, [2])
	if (domain == "postInfl2"):
		phobic_interval = GetColumn(intervals_phobic, [3])
	if (domain == "pre"):
		phobic_interval = GetColumn(intervals_phobic, [4])
	if (domain == "post"):
		phobic_interval = GetColumn(intervals_phobic, [5])


	## Even ##
				
	## Philic
	
	## Read
	exp_philic_even = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even, interval= 'powerLaw_' + domain, infoBlock = 'exponent')/philic_interval
	## Write
	StoreFile(np.append(exp_philic_even[inds_class0], exp_philic_even[inds_class1]), "exp_philic_" + str(jth_order_even) + "_" + domain + "_" + FN_ENDING)

	## Read
	PC_philic_even = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even, interval= 'powerLaw_' + domain, infoBlock = 'PC')
	## Write
	StoreFile(np.append(PC_philic_even[inds_class0], PC_philic_even[inds_class1]), "PC_philic_" + str(jth_order_even) + "_" + domain + "_" + FN_ENDING)

	## Phobic
	
	## Read
	exp_phobic_even = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even, interval= 'powerLaw_' + domain, infoBlock = 'exponent')/phobic_interval
	## Write
	StoreFile(np.append(exp_phobic_even[inds_class0], exp_phobic_even[inds_class1]), "exp_phobic_" + str(jth_order_even) + "_" + domain + "_" + FN_ENDING)

	## Read
	PC_phobic_even = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even, interval= 'powerLaw_' + domain, infoBlock = 'PC')
	## Write
	StoreFile(np.append(PC_phobic_even[inds_class0], PC_phobic_even[inds_class1]), "PC_phobic_" + str(jth_order_even) + "_" + domain + "_" + FN_ENDING)
	

	## Odd ##

	## Philic
	
	# Read
	exp_philic_odd_x = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "x", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/philic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_philic_odd_x[inds_class0], exp_philic_odd_x[inds_class1]), "exp_philic_" + str(jth_order_even + 1) + "_x_" + domain + "_" + FN_ENDING)

	exp_philic_odd_y = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "y", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/philic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_philic_odd_y[inds_class0], exp_philic_odd_y[inds_class1]), "exp_philic_" + str(jth_order_even + 1) + "_y_" + domain + "_" + FN_ENDING)

	exp_philic_odd_z = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "z", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/philic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_philic_odd_z[inds_class0], exp_philic_odd_z[inds_class1]), "exp_philic_" + str(jth_order_even + 1) + "_z_" + domain + "_" + FN_ENDING)

	# Read
	PC_philic_odd_x = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "x", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_philic_odd_x[inds_class0], PC_philic_odd_x[inds_class1]), "PC_philic_" + str(jth_order_even + 1) + "_x_" + domain + "_" + FN_ENDING)

	PC_philic_odd_y = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "y", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_philic_odd_y[inds_class0], PC_philic_odd_y[inds_class1]), "PC_philic_" + str(jth_order_even + 1) + "_y_" + domain + "_" + FN_ENDING)

	PC_philic_odd_z = ScalingInfo_res(exponentsDomains_philic, orderOfhydrMom = jth_order_even + 1, component = "z", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_philic_odd_z[inds_class0], PC_philic_odd_z[inds_class1]), "PC_philic_" + str(jth_order_even + 1) + "_z_" + domain + "_" + FN_ENDING)

	## Phobic

	# Read
	exp_phobic_odd_x = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "x", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/phobic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_phobic_odd_x[inds_class0], exp_phobic_odd_x[inds_class1]), "exp_phobic_" + str(jth_order_even + 1) + "_x_" + domain + "_" + FN_ENDING)

	exp_phobic_odd_y = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "y", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/phobic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_phobic_odd_y[inds_class0], exp_phobic_odd_y[inds_class1]), "exp_phobic_" + str(jth_order_even + 1) + "_y_" + domain + "_" + FN_ENDING)

	exp_phobic_odd_z = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "z", interval= 'powerLaw_' + domain, infoBlock = 'exponent')/phobic_interval
	# Write
	if len(inds_class0):
		StoreFile(np.append(exp_phobic_odd_z[inds_class0], exp_phobic_odd_z[inds_class1]), "exp_phobic_" + str(jth_order_even + 1) + "_z_" + domain + "_" + FN_ENDING)

	# Read
	PC_phobic_odd_x = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "x", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_phobic_odd_x[inds_class0], PC_phobic_odd_x[inds_class1]), "PC_phobic_" + str(jth_order_even + 1) + "_x_" + domain + "_" + FN_ENDING)

	PC_phobic_odd_y = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "y", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_phobic_odd_y[inds_class0], PC_phobic_odd_y[inds_class1]), "PC_phobic_" + str(jth_order_even + 1) + "_y_" + domain + "_" + FN_ENDING)

	PC_phobic_odd_z = ScalingInfo_res(exponentsDomains_phobic, orderOfhydrMom = jth_order_even + 1, component = "z", interval= 'powerLaw_' + domain, infoBlock = 'PC')
	# Write
	if len(inds_class0):
		StoreFile(np.append(PC_phobic_odd_z[inds_class0], PC_phobic_odd_z[inds_class1]), "PC_phobic_" + str(jth_order_even + 1) + "_z_" + domain + "_" + FN_ENDING)


	return 	[
				## Even
				exp_philic_even , 
				PC_philic_even, 
				
				exp_phobic_even , 
				PC_phobic_even, 
		 		
				## Odd
				exp_philic_odd_x, exp_philic_odd_y, exp_philic_odd_z,
				PC_philic_odd_x, PC_philic_odd_y, PC_philic_odd_z,

				exp_phobic_odd_x, exp_phobic_odd_y, exp_phobic_odd_z,
				PC_phobic_odd_x, PC_phobic_odd_y, PC_phobic_odd_z,
		 	]



def StoreFeatures(		
						exponentsDomains_phobic,			
						exponentsDomains_philic,	

						nu_phobic, 
						nu_philic,
						
						intervals_phobic,
						intervals_philic,
						
						resId, resName, 
						
						thermodynamic_variables, fractal_dimension,

						inds_class0 = [], inds_class1 = [], 

						SCNXA = False, 
						VIS_SCORE = None
						
				):

	FN_ENDING = "res_score"
	if SCNXA: 
		FN_ENDING = "res_SCNXA_score"

	"""
		Init
	"""
	"""
		Statistical summaries
	"""
	# even
	score_philic_even_preInfl1_class0 = []
	score_phobic_even_preInfl1_class0 = []
	score_philic_even_preInfl1_class1 = []
	score_phobic_even_preInfl1_class1 = []

	score_philic_even_infl1_class0 = []
	score_phobic_even_infl1_class0 = []
	score_philic_even_infl1_class1 = []
	score_phobic_even_infl1_class1 = []

	score_philic_even_infl2_class0 = []
	score_phobic_even_infl2_class0 = []
	score_philic_even_infl2_class1 = []
	score_phobic_even_infl2_class1 = []

	score_philic_even_postInfl2_class0 = []
	score_phobic_even_postInfl2_class0 = []
	score_philic_even_postInfl2_class1 = []
	score_phobic_even_postInfl2_class1 = []

	# odd
	score_philic_odd_preInfl1_class0 = []
	score_phobic_odd_preInfl1_class0 = []
	score_philic_odd_preInfl1_class1 = []
	score_phobic_odd_preInfl1_class1 = []

	score_philic_odd_infl1_class0 = []
	score_phobic_odd_infl1_class0 = []
	score_philic_odd_infl1_class1 = []
	score_phobic_odd_infl1_class1 = []

	score_philic_odd_infl2_class0 = []
	score_phobic_odd_infl2_class0 = []
	score_philic_odd_infl2_class1 = []
	score_phobic_odd_infl2_class1 = []

	score_philic_odd_postInfl2_class0 = []
	score_phobic_odd_postInfl2_class0 = []
	score_philic_odd_postInfl2_class1 = []
	score_phobic_odd_postInfl2_class1 = []

	## visual scores
	# even
	score_even_preInfl1 = 0
	score_even_infl1 = 0
	score_even_infl2 = 0
	score_even_postInfl2 = 0
	# odd
	score_odd_preInfl1 = 0
	score_odd_infl1 = 0
	score_odd_infl2 = 0
	score_odd_postInfl2 = 0

	# weighted average
	weight_phobic = (nu_phobic + 1) / (nu_phobic + nu_philic + 2)
	weight_philic = (nu_philic + 1) / (nu_phobic + nu_philic + 2)

	"""
		Scores are obtained as follows:
			1. For each domain (preInfl1, infl1, infl2, postInfl2, pre, post)
			2. For each order of the hydrophobic moment (0, 1, 2, ..., 2*ModelParameters.N_HYDR - 1)
			3. Compute the statistical summaries (even and odd) for the phobic and philic components
			4. Compute the visualization scores (even and odd) for the phobic and philic components
	"""

	for i in range(0, ModelParameters.N_HYDR*2 - 1, 2):

		"""
			preInfl1
		"""
		## 	exp_philic_even 	0 	PC_philic_even 		1 
		## 	exp_phobic_even		2 	PC_phobic_even 		3 
		## 	exp_philic_odd_x	4 	exp_philic_odd_y 	5 	exp_philic_odd_z	6 
		## 	PC_philic_odd_x		7 	PC_philic_odd_y		8 	PC_philic_odd_z		9	
		##	exp_phobic_odd_x	10 	exp_phobic_odd_y	11 	exp_phobic_odd_z	12
		## 	PC_phobic_odd_x		13 	PC_phobic_odd_y		14 	PC_phobic_odd_z 	15 
		domainScalingFeatures_preInfl1 = DomainFeatures('preInfl1', exponentsDomains_phobic, exponentsDomains_philic, intervals_phobic, intervals_philic, i, FN_ENDING, inds_class0 = inds_class0, inds_class1 = inds_class1)

		## Statistical summary (phobic)
		score_philic_even_preInfl1_class0.append(domainScalingFeatures_preInfl1[0][inds_class0] * domainScalingFeatures_preInfl1[1][inds_class0])
		score_phobic_even_preInfl1_class0.append(domainScalingFeatures_preInfl1[2][inds_class0] * domainScalingFeatures_preInfl1[3][inds_class0])
		score_philic_even_preInfl1_class1.append(domainScalingFeatures_preInfl1[0][inds_class1] * domainScalingFeatures_preInfl1[1][inds_class1])
		score_phobic_even_preInfl1_class1.append(domainScalingFeatures_preInfl1[2][inds_class1] * domainScalingFeatures_preInfl1[3][inds_class1])
		## Visualization score
		score_even_preInfl1 += weight_phobic * domainScalingFeatures_preInfl1[3] * domainScalingFeatures_preInfl1[2] + weight_philic * domainScalingFeatures_preInfl1[1] * domainScalingFeatures_preInfl1[0]

		## Statistical summary (odd)
		_score_philic = np.sqrt( 
									(domainScalingFeatures_preInfl1[4] * domainScalingFeatures_preInfl1[7])**2 +
									(domainScalingFeatures_preInfl1[5] * domainScalingFeatures_preInfl1[8])**2 +
									(domainScalingFeatures_preInfl1[6] * domainScalingFeatures_preInfl1[9])**2 
								)
		_score_phobic = np.sqrt( 
									(domainScalingFeatures_preInfl1[10] * domainScalingFeatures_preInfl1[13])**2 +
									(domainScalingFeatures_preInfl1[11] * domainScalingFeatures_preInfl1[14])**2 +
									(domainScalingFeatures_preInfl1[12] * domainScalingFeatures_preInfl1[15])**2 
								)
		
		score_philic_odd_preInfl1_class0.append(_score_philic[inds_class0])
		score_phobic_odd_preInfl1_class0.append(_score_phobic[inds_class0])
		score_philic_odd_preInfl1_class1.append(_score_philic[inds_class1])
		score_phobic_odd_preInfl1_class1.append(_score_phobic[inds_class1])	
		## Visualization score
		score_odd_preInfl1 += weight_phobic * _score_phobic + weight_philic * _score_philic 

		"""
			infl1
		"""
		## 	exp_philic_even 	0 	PC_philic_even 		1 
		## 	exp_phobic_even		2 	PC_phobic_even 		3 
		## 	exp_philic_odd_x	4 	exp_philic_odd_y 	5 	exp_philic_odd_z	6 
		## 	PC_philic_odd_x		7 	PC_philic_odd_y		8 	PC_philic_odd_z		9	
		##	exp_phobic_odd_x	10 	exp_phobic_odd_y	11 	exp_phobic_odd_z	12
		## 	PC_phobic_odd_x		13 	PC_phobic_odd_y		14 	PC_phobic_odd_z 	15 
		domainScalingFeatures_infl1 = DomainFeatures('infl1', exponentsDomains_phobic, exponentsDomains_philic, intervals_phobic, intervals_philic, i, FN_ENDING, inds_class0 = inds_class0, inds_class1 = inds_class1)

		## Statistical summary (phobic)
		score_philic_even_infl1_class0.append(domainScalingFeatures_infl1[0][inds_class0] * domainScalingFeatures_infl1[1][inds_class0])
		score_phobic_even_infl1_class0.append(domainScalingFeatures_infl1[2][inds_class0] * domainScalingFeatures_infl1[3][inds_class0])
		score_philic_even_infl1_class1.append(domainScalingFeatures_infl1[0][inds_class1] * domainScalingFeatures_infl1[1][inds_class1])
		score_phobic_even_infl1_class1.append(domainScalingFeatures_infl1[2][inds_class1] * domainScalingFeatures_infl1[3][inds_class1])
		## Visualization score
		score_even_infl1 += weight_phobic * domainScalingFeatures_infl1[3] * domainScalingFeatures_infl1[2] + weight_philic * domainScalingFeatures_infl1[1] * domainScalingFeatures_infl1[0]

		## Statistical summary (odd)
		_score_philic = np.sqrt( 
									(domainScalingFeatures_infl1[4] * domainScalingFeatures_infl1[7])**2 +
									(domainScalingFeatures_infl1[5] * domainScalingFeatures_infl1[8])**2 +
									(domainScalingFeatures_infl1[6] * domainScalingFeatures_infl1[9])**2 
								)
		_score_phobic = np.sqrt( 
									(domainScalingFeatures_infl1[10] * domainScalingFeatures_infl1[13])**2 +
									(domainScalingFeatures_infl1[11] * domainScalingFeatures_infl1[14])**2 +
									(domainScalingFeatures_infl1[12] * domainScalingFeatures_infl1[15])**2 
								)
		
		score_philic_odd_infl1_class0.append(_score_philic[inds_class0])
		score_phobic_odd_infl1_class0.append(_score_phobic[inds_class0])
		score_philic_odd_infl1_class1.append(_score_philic[inds_class1])
		score_phobic_odd_infl1_class1.append(_score_phobic[inds_class1])	
		## Visualization score
		score_odd_infl1 += weight_phobic * _score_phobic + weight_philic * _score_philic 

		"""
			infl2
		"""
		## 	exp_philic_even 	0 	PC_philic_even 		1 
		## 	exp_phobic_even		2 	PC_phobic_even 		3 
		## 	exp_philic_odd_x	4 	exp_philic_odd_y 	5 	exp_philic_odd_z	6 
		## 	PC_philic_odd_x		7 	PC_philic_odd_y		8 	PC_philic_odd_z		9	
		##	exp_phobic_odd_x	10 	exp_phobic_odd_y	11 	exp_phobic_odd_z	12
		## 	PC_phobic_odd_x		13 	PC_phobic_odd_y		14 	PC_phobic_odd_z 	15 
		domainScalingFeatures_infl2 = DomainFeatures('infl2', exponentsDomains_phobic, exponentsDomains_philic, intervals_phobic, intervals_philic, i, FN_ENDING, inds_class0 = inds_class0, inds_class1 = inds_class1)

		## Statistical summary (phobic)
		score_philic_even_infl2_class0.append(domainScalingFeatures_infl2[0][inds_class0] * domainScalingFeatures_infl2[1][inds_class0])
		score_phobic_even_infl2_class0.append(domainScalingFeatures_infl2[2][inds_class0] * domainScalingFeatures_infl2[3][inds_class0])
		score_philic_even_infl2_class1.append(domainScalingFeatures_infl2[0][inds_class1] * domainScalingFeatures_infl2[1][inds_class1])
		score_phobic_even_infl2_class1.append(domainScalingFeatures_infl2[2][inds_class1] * domainScalingFeatures_infl2[3][inds_class1])
		## Visualization score
		score_even_infl2 += weight_phobic * domainScalingFeatures_infl2[3] * domainScalingFeatures_infl2[2] + weight_philic * domainScalingFeatures_infl2[1] * domainScalingFeatures_infl2[0]

		## Statistical summary (odd)
		_score_philic = np.sqrt( 
									(domainScalingFeatures_infl2[4] * domainScalingFeatures_infl2[7])**2 +
									(domainScalingFeatures_infl2[5] * domainScalingFeatures_infl2[8])**2 +
									(domainScalingFeatures_infl2[6] * domainScalingFeatures_infl2[9])**2 
								)
		_score_phobic = np.sqrt( 
									(domainScalingFeatures_infl2[10] * domainScalingFeatures_infl2[13])**2 +
									(domainScalingFeatures_infl2[11] * domainScalingFeatures_infl2[14])**2 +
									(domainScalingFeatures_infl2[12] * domainScalingFeatures_infl2[15])**2 
								)
		
		score_philic_odd_infl2_class0.append(_score_philic[inds_class0])
		score_phobic_odd_infl2_class0.append(_score_phobic[inds_class0])
		score_philic_odd_infl2_class1.append(_score_philic[inds_class1])
		score_phobic_odd_infl2_class1.append(_score_phobic[inds_class1])	
		## Visualization score
		score_odd_infl2 += weight_phobic * _score_phobic + weight_philic * _score_philic 

		"""
			postInfl2
		"""
		## 	exp_philic_even 	0 	PC_philic_even 		1 
		## 	exp_phobic_even		2 	PC_phobic_even 		3 
		## 	exp_philic_odd_x	4 	exp_philic_odd_y 	5 	exp_philic_odd_z	6 
		## 	PC_philic_odd_x		7 	PC_philic_odd_y		8 	PC_philic_odd_z		9	
		##	exp_phobic_odd_x	10 	exp_phobic_odd_y	11 	exp_phobic_odd_z	12
		## 	PC_phobic_odd_x		13 	PC_phobic_odd_y		14 	PC_phobic_odd_z 	15 
		domainScalingFeatures_postInfl2 = DomainFeatures('postInfl2', exponentsDomains_phobic, exponentsDomains_philic, intervals_phobic, intervals_philic, i, FN_ENDING, inds_class0 = inds_class0, inds_class1 = inds_class1)

		## Statistical summary (phobic)
		score_philic_even_postInfl2_class0.append(domainScalingFeatures_postInfl2[0][inds_class0] * domainScalingFeatures_postInfl2[1][inds_class0])
		score_phobic_even_postInfl2_class0.append(domainScalingFeatures_postInfl2[2][inds_class0] * domainScalingFeatures_postInfl2[3][inds_class0])
		score_philic_even_postInfl2_class1.append(domainScalingFeatures_postInfl2[0][inds_class1] * domainScalingFeatures_postInfl2[1][inds_class1])
		score_phobic_even_postInfl2_class1.append(domainScalingFeatures_postInfl2[2][inds_class1] * domainScalingFeatures_postInfl2[3][inds_class1])
		## Visualization score
		score_even_postInfl2 += weight_phobic * domainScalingFeatures_postInfl2[3] * domainScalingFeatures_postInfl2[2] + weight_philic * domainScalingFeatures_postInfl2[1] * domainScalingFeatures_postInfl2[0]

		## Statistical summary (odd)
		_score_philic = np.sqrt( 
									(domainScalingFeatures_postInfl2[4] * domainScalingFeatures_postInfl2[7])**2 +
									(domainScalingFeatures_postInfl2[5] * domainScalingFeatures_postInfl2[8])**2 +
									(domainScalingFeatures_postInfl2[6] * domainScalingFeatures_postInfl2[9])**2 
								)
		_score_phobic = np.sqrt( 
									(domainScalingFeatures_postInfl2[10] * domainScalingFeatures_postInfl2[13])**2 +
									(domainScalingFeatures_postInfl2[11] * domainScalingFeatures_postInfl2[14])**2 +
									(domainScalingFeatures_postInfl2[12] * domainScalingFeatures_postInfl2[15])**2 
								)
		
		score_philic_odd_postInfl2_class0.append(_score_philic[inds_class0])
		score_phobic_odd_postInfl2_class0.append(_score_phobic[inds_class0])
		score_philic_odd_postInfl2_class1.append(_score_philic[inds_class1])
		score_phobic_odd_postInfl2_class1.append(_score_phobic[inds_class1])	
		## Visualization score
		score_odd_postInfl2 += weight_phobic * _score_phobic + weight_philic * _score_philic 

	## Store statistical summaries (for later on visualization)
	## even
	StoreFile(score_philic_even_preInfl1_class0, "statSum_philic_even_preInfl1_class0_" + FN_ENDING)
	StoreFile(score_phobic_even_preInfl1_class0, "statSum_phobic_even_preInfl1_class0_" + FN_ENDING)
	StoreFile(score_philic_even_preInfl1_class1, "statSum_philic_even_preInfl1_class1_" + FN_ENDING)
	StoreFile(score_phobic_even_preInfl1_class1, "statSum_phobic_even_preInfl1_class1_" + FN_ENDING)
	
	StoreFile(score_philic_even_infl1_class0, "statSum_philic_even_infl1_class0_" + FN_ENDING)
	StoreFile(score_phobic_even_infl1_class0, "statSum_phobic_even_infl1_class0_" + FN_ENDING)
	StoreFile(score_philic_even_infl1_class1, "statSum_philic_even_infl1_class1_" + FN_ENDING)
	StoreFile(score_phobic_even_infl1_class1, "statSum_phobic_even_infl1_class1_" + FN_ENDING)
	
	StoreFile(score_philic_even_infl2_class0, "statSum_philic_even_infl2_class0_" + FN_ENDING)
	StoreFile(score_phobic_even_infl2_class0, "statSum_phobic_even_infl2_class0_" + FN_ENDING)
	StoreFile(score_philic_even_infl2_class1, "statSum_philic_even_infl2_class1_" + FN_ENDING)
	StoreFile(score_phobic_even_infl2_class1, "statSum_phobic_even_infl2_class1_" + FN_ENDING)
	
	StoreFile(score_philic_even_postInfl2_class0, "statSum_philic_even_postInfl2_class0_" + FN_ENDING)
	StoreFile(score_phobic_even_postInfl2_class0, "statSum_phobic_even_postInfl2_class0_" + FN_ENDING)
	StoreFile(score_philic_even_postInfl2_class1, "statSum_philic_even_postInfl2_class1_" + FN_ENDING)
	StoreFile(score_phobic_even_postInfl2_class1, "statSum_phobic_even_postInfl2_class1_" + FN_ENDING)
	
	## odd
	StoreFile(score_philic_odd_preInfl1_class0, "statSum_philic_odd_preInfl1_class0_" + FN_ENDING)
	StoreFile(score_phobic_odd_preInfl1_class0, "statSum_phobic_odd_preInfl1_class0_" + FN_ENDING)
	StoreFile(score_philic_odd_preInfl1_class1, "statSum_philic_odd_preInfl1_class1_" + FN_ENDING)
	StoreFile(score_phobic_odd_preInfl1_class1, "statSum_phobic_odd_preInfl1_class1_" + FN_ENDING)
		
	StoreFile(score_philic_odd_infl1_class0, "statSum_philic_odd_infl1_class0_" + FN_ENDING)
	StoreFile(score_phobic_odd_infl1_class0, "statSum_phobic_odd_infl1_class0_" + FN_ENDING)
	StoreFile(score_philic_odd_infl1_class1, "statSum_philic_odd_infl1_class1_" + FN_ENDING)
	StoreFile(score_phobic_odd_infl1_class1, "statSum_phobic_odd_infl1_class1_" + FN_ENDING)
	
	StoreFile(score_philic_odd_infl2_class0, "statSum_philic_odd_infl2_class0_" + FN_ENDING)
	StoreFile(score_phobic_odd_infl2_class0, "statSum_phobic_odd_infl2_class0_" + FN_ENDING)
	StoreFile(score_philic_odd_infl2_class1, "statSum_philic_odd_infl2_class1_" + FN_ENDING)
	StoreFile(score_phobic_odd_infl2_class1, "statSum_phobic_odd_infl2_class1_" + FN_ENDING)
	
	StoreFile(score_philic_odd_postInfl2_class0, "statSum_philic_odd_postInfl2_class0_" + FN_ENDING)
	StoreFile(score_phobic_odd_postInfl2_class0, "statSum_phobic_odd_postInfl2_class0_" + FN_ENDING)
	StoreFile(score_philic_odd_postInfl2_class1, "statSum_philic_odd_postInfl2_class1_" + FN_ENDING)
	StoreFile(score_phobic_odd_postInfl2_class1, "statSum_phobic_odd_postInfl2_class1_" + FN_ENDING)

	## Store superfeature
	## even
	StoreFile(score_even_preInfl1, "superFeature_score_even_preInfl1_" + FN_ENDING)
	StoreFile(score_even_infl1, "superFeature_score_even_infl1_" + FN_ENDING)
	StoreFile(score_even_infl2, "superFeature_score_even_infl2_" + FN_ENDING)
	StoreFile(score_even_postInfl2, "superFeature_score_even_postInfl2_" + FN_ENDING)
	## odd
	StoreFile(score_odd_preInfl1, "superFeature_score_odd_preInfl1_" + FN_ENDING)
	StoreFile(score_odd_infl1, "superFeature_score_odd_infl1_" + FN_ENDING)
	StoreFile(score_odd_infl2, "superFeature_score_odd_infl2_" + FN_ENDING)
	StoreFile(score_odd_postInfl2, "superFeature_score_odd_postInfl2_" + FN_ENDING)

	S = thermodynamic_variables[0]
	F = thermodynamic_variables[1]
	T = thermodynamic_variables[2]

	## Score selection
	if VIS_SCORE != None:

		print("Visualizing: ", VIS_SCORE)
		
		if (VIS_SCORE == "preInfl1_even"):
			score_vis = score_even_preInfl1
		if (VIS_SCORE == "infl1_even"):
			score_vis = score_even_infl1
		if (VIS_SCORE == "infl2_even"):
			print("here!")
			score_vis = score_even_infl2
		if (VIS_SCORE == "postInfl2_even"):
			score_vis = score_even_postInfl2

		if (VIS_SCORE == "preInfl1_odd"):
			score_vis =  score_odd_preInfl1
		if (VIS_SCORE == "infl1_odd"):
			score_vis = score_odd_infl1
		if (VIS_SCORE == "infl2_odd"):
			score_vis = score_odd_infl2
		if (VIS_SCORE == "postInfl2_odd"):
			score_vis = score_odd_postInfl2



		X = 10
		indices = np.argsort(score_vis)[-X:][::-1] # indices to the ten largest values in descending order
		top_X_id = resId[indices]
		top_X_type = resName[indices]
		i = 0
		for res_id in top_X_id:
			print(res_id, top_X_type[i])
			i += 1
		# score_vis = S
		ColorResidues(	
	 				np.column_stack((resId, Standarize(score_vis))), 
	 			  	# resId_class0 = resId[inds_class0], 
	 				# resId_class1 = resId[inds_class1],
	 				resId_interact = top_X_id,
					
	 			)


	fig, ax = plt.subplots()

	coeffs = np.polyfit(S, fractal_dimension, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, fractal_dimension)
	
	ax.plot(S, fractal_dimension, "o", alpha = 0.2, markersize = 20)
	ax.plot(S, y_fit, color="red", alpha = 0.5, linewidth = 5, label=rf'$d_f \big|_{{i}} = {slope:.1f} \mathcal{{S}}_q + {intercept:.1f}$, PC = {r:.2f}')
	ax.set_ylim([1.7,3.5])
	ax.set_xlim([min(S) - 0.01, max(S) + 0.01])
	ax.set_xlabel(r"Entropy $(\mathcal{S}_q)$", fontsize = 70)
	ax.set_ylabel(r"Intrinsic dimension $(d_{f} \big|_{i})$", fontsize = 70)
	# ax.legend(loc = "upper left")
	ax.grid(True, axis='both', linestyle='--', alpha=0.5)  
	ax.text(0.015, 0.05, '(a)', transform=ax.transAxes, fontsize = 65, fontweight='bold')

	inset_ax = fig.add_axes([0.66, 0.73, 0.23, 0.25]) 
	coeffs = np.polyfit(T, S, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * T + intercept
	r, _ = stats.pearsonr(T, S)
	inset_ax.plot(T, S, "o", alpha = 0.2)
	inset_ax.set_xlabel(r"$\mathcal{T}$")
	inset_ax.set_ylabel(r"$\mathcal{S}_q$")
	inset_ax.plot(T[250:700], y_fit[250:700], color="red", alpha = 0.5, linewidth = 5, label=rf'$d_f \big|_{{i}} = {slope:.1f} \mathcal{{S}}_q + {intercept:.1f}$, PC = {r:.2f}')
	#inset_ax.set_ylim([0.5,2])
	# inset_ax.legend(loc = "upper left", fontsize = 25)
	inset_ax.grid(True, axis='both', linestyle='--', alpha=0.5)  
	inset_ax.text(0.78, 0.15, '(c)', transform=inset_ax.transAxes, fontsize=60, fontweight='bold')

	inset_ax1 = fig.add_axes([0.32, 0.73, 0.23, 0.25]) 
	coeffs = np.polyfit(S, F, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, F)
	inset_ax1.plot(S, F, "o", alpha = 0.2)
	inset_ax1.set_xlabel(r"$\mathcal{S}_q$")
	inset_ax1.set_ylabel(r"$\mathcal{F}$")
	inset_ax1.plot(S[250:900], y_fit[250:900], color="red", alpha = 0.5, linewidth = 5, label=rf'$d_f \big|_{{i}} = {slope:.1f} \mathcal{{S}}_q + {intercept:.1f}$, PC = {r:.2f}')
	#inset_ax.set_ylim([0.5,2])
	# inset_ax.legend(loc = "upper left", fontsize = 25)
	inset_ax1.grid(True, axis='both', linestyle='--', alpha=0.5)  
	inset_ax1.text(0.025, 0.15, '(b)', transform=inset_ax1.transAxes, fontsize=60, fontweight='bold')
	plt.show()
    
	fig, ax = plt.subplots()
	coeffs = np.polyfit(S, score_even_preInfl1, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, score_even_preInfl1)
	plt.plot(S, score_even_preInfl1, "o", alpha = 0.2, markersize = 20, label=r'rotational DoF ($j = 2k$)')
	plt.plot(S, y_fit, color="red", alpha = 0.5, linewidth = 5)

	coeffs = np.polyfit(S, score_odd_preInfl1, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, score_odd_preInfl1)
	plt.plot(S, score_odd_preInfl1, "o", alpha = 0.2, markersize = 20, label=r'translational DoF ($j = 2k + 1$)')
	plt.plot(S, y_fit, color="green", alpha = 0.5, linewidth = 5)	
	# plt.ylim([0,6.5])
	plt.xlim([min(S) - 0.01, max(S) + 0.01])
	
	plt.grid(True, axis='both', linestyle='--', alpha=0.5)  
	plt.xlabel(r"$\mathcal{S}_q$", fontsize = 70)
	plt.ylabel(r"Allostery index", fontsize = 70)
	plt.legend()
	plt.show()

	fig, ax = plt.subplots()
	coeffs = np.polyfit(S, score_even_infl2, 1)
	slope, intercept = coeffs
	# Fitted line
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, score_even_infl2)
	plt.plot(S, score_even_infl2, "o", alpha = 0.2, markersize = 20, label=r'rotational DoF ($j = 2k$)')
	plt.plot(S, y_fit, color="red", alpha = 0.5, linewidth = 5)

	coeffs = np.polyfit(S, score_odd_infl2, 1)
	slope, intercept = coeffs
	y_fit = slope * S + intercept
	r, _ = stats.pearsonr(S, score_odd_infl2)
	plt.plot(S, score_odd_infl2, "o", alpha = 0.2, markersize = 20, label=r'translational DoF ($j = 2k + 1$)')
	plt.plot(S, y_fit, color="green", alpha = 0.5, linewidth = 5)
	plt.ylim([0,6.5])
	plt.xlim([min(S) - 0.01, max(S) + 0.01])
	
	
	plt.xlabel(r"$\mathcal{S}_q$", fontsize = 70)
	plt.ylabel(r"Allostery index", fontsize = 70)
	plt.grid(True, axis='both', linestyle='--', alpha=0.5)  
	plt.legend()
	plt.show()
	
	# exit()



def Plot_DegreeOfAllostery(data_type, FN_ENDING):

	if data_type == "phobic_even":
		fig_title = "Hydrophobicity-driven rotational DoF"
	if data_type == "philic_even":
		fig_title = "Hydrophilicity-driven rotational DoF"
	if data_type == "phobic_odd":
		fig_title = "Hydrophobiciy-driven translational DoF"
	if data_type == "philic_odd":
		fig_title = "Hydrophilicity-driven translational DoF"

	data1 = [	
						np.concatenate(LoadFile("statSum_"  + data_type + "_preInfl1_class0_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_infl1_class0_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_infl2_class0_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_postInfl2_class0_" + FN_ENDING)), 
						
				]
		
	data2 = [	
						np.concatenate(LoadFile("statSum_"  + data_type +  "_preInfl1_class1_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_infl1_class1_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_infl2_class1_" + FN_ENDING)), 
						np.concatenate(LoadFile("statSum_"  + data_type + "_postInfl2_class1_" + FN_ENDING)), 
				]
	labels = [r'$\sim 30$ \AA', r'$\sim 45$ \AA', r'$\sim 60$ \AA', r'$\sim 75$ \AA']

	fig, ax = plt.subplots(figsize=(6, 4))
	ax.boxplot(	
						data1, positions=[0,1,2,3], widths=0.15,
						patch_artist=True,
						boxprops=dict(facecolor='red', color='red', alpha = 0.2),
						whiskerprops=dict(color='red'),
						capprops=dict(color='red'),
						medianprops=dict(color='red'),
						flierprops=dict(marker='o', color='red', markersize=8)
						)
					
	
	ax.boxplot(	
						data2, positions=[0.25,1.25,2.25,3.25], widths=0.15,
						patch_artist=True,
						boxprops=dict(facecolor='blue', color='blue', alpha = 0.2),
						whiskerprops=dict(color='blue'),
						capprops=dict(color='blue'),
						medianprops=dict(color='blue'),
						flierprops=dict(marker='o', color='blue', markersize=8)
					)
	
	ax.set_xticks([0.125, 1.125, 2.125, 3.125])
	ax.set_xticklabels(labels, fontsize = 35)
	
	ax.set_ylabel("Degree of allostery [1/\AA]", fontsize = 35)
	ax.tick_params(axis='y', labelsize=35) 
	ax.set_title(fig_title, fontsize = 35)
	
	ax.set_yticks(np.arange(-0.25, 2.0, 0.25))
	ax.set_yticklabels(np.arange(-0.25, 2.0, 0.25), fontsize=25)
	
	ax.grid(True, axis='y', linestyle='--', alpha=0.5)
	# if (data_type == "philic_even" or data_type == "phobic_even"):
	ax.set_ylim([-0.25,1.75])
	plt.tight_layout()
	plt.show()




def Plot_TracesExamples_SF(ind, scales, inds_l_i, l_i, fracDim_i, nrOfAtoms, statMod, orderOfMom, hydrMoments, exponentsDomains, l_min, l_max, l_min_curv, l_max_curv):
	
	l 					= scales[ind]
	ind_l_i				= inds_l_i[ind].astype(int)
	n	 				= nrOfAtoms[ind][0]
	# dn				= np.append(n[0],np.diff(n)/np.diff(l))
	modType 			= statMod[ind][0][15]
	params	 			= [statMod[ind][0][0], statMod[ind][0][2], statMod[ind][0][4], statMod[ind][0][6]]
	n_model		 		= GeomModel(l, params, modType) 
	fracDim_at_l_i      = fracDim_i[ind]

	print(params)
	print("\n Fractal dim is:", fracDim_i[ind])

	n_pho				= nrOfAtoms[ind][1]
	inds_pho 			= np.where(n_pho > 0) 
	l_pho				= l[inds_pho]
	n_pho				= n_pho[inds_pho]
	# dn_pho			= np.append(n_pho[0],np.diff(n_pho)/np.diff(l_pho))
	modType_pho 		= statMod[ind][1][15]
	params_pho 			= [statMod[ind][1][0], statMod[ind][1][2], statMod[ind][1][4], statMod[ind][1][6]]
	n_model_pho		 	= GeomModel(l_pho, params_pho, modType_pho)*max(n_pho) # equiv to "distance"
	# dn_model_pho		= Tools.GeomModel(l_pho, params_pho, der = 1)*max(n_pho) 
		
	n_phi				= nrOfAtoms[ind][2]
	inds_phi 			= np.where(n_phi > 0)
	l_phi				= l[inds_phi]
	n_phi				= n_phi[inds_phi]
	dn_phi				= np.append(n_phi[0],np.diff(n_phi)/np.diff(l_phi))
	modType_phi 		= statMod[ind][2][15]
	params_phi 			= [statMod[ind][2][0], statMod[ind][2][2], statMod[ind][2][4], statMod[ind][2][6]]
	n_model_phi		 	= GeomModel(l_phi, params_phi, modType_phi)*max(n_phi) # equiv to "distance"
	dn_model_phi		= GeomModel(l_phi, params_phi, der = 1)*max(n_phi)
	
	h1_z 				= GetHydrMom(hydrMoments, ind, 'pathic', 1, 'z')
	h2 					= GetHydrMom(hydrMoments, ind, 'pathic', 2)
	h3_z 				= GetHydrMom(hydrMoments, ind, 'pathic', 3, 'z')
	h4 					= GetHydrMom(hydrMoments, ind, 'pathic', 4)
	h5 					= GetHydrMom(hydrMoments, ind, 'pathic', 5, 'z')
	h6 					= GetHydrMom(hydrMoments, ind, 'pathic', 6)


	h1_z = h1_z/h1_z[ind_l_i]
	l_pre = l[np.arange(0, ind_l_i , 1).astype(int)] / l_i[ind]
	l_post = l[np.arange(ind_l_i , Match(l_min_curv[ind],l) + 50, 1).astype(int)] / l_i[ind]

	# for h1_z
	coeff_pre_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_pre', infoBlock = 'coeff')[ind]
	exp_pre_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_pre', infoBlock = 'exponent')[ind]
	pl_model_pre_1 = np.exp(exp_pre_1*np.log(l_pre) + coeff_pre_1) 
	
	coeff_post_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'coeff')[ind]
	exp_post_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'exponent')[ind]
	pl_model_post_1 = np.exp(exp_post_1*np.log(l_post) + coeff_post_1)

	fig, ax = plt.subplots()
	ax.plot(np.log(l/l_i[ind]), n, "bo", alpha=0.1, markersize=20) # , label = r"$N$")
	ax.plot(np.log(l/l_i[ind]), n_model*max(n), "r--", alpha = 0.7, linewidth=10, label = r"$n_{\nu \to 0}$")
	ax.set_ylabel(r'$N$ [atom]' , fontsize=60)
	ax.set_xlabel(r'$\ln(l/l_i)$' , fontsize=60)
	ax.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2)
	inset_ax = fig.add_axes([0.23, 0.42, 0.39, 0.44]) 
	inset_ax.plot(np.log(l/l_i[ind]), np.log(n), "bo", alpha=0.1, markersize=20)
	inset_ax.plot(np.log(l/l_i[ind]), np.log(l/l_i[ind])*fracDim_at_l_i + 9.7, "m--", alpha=0.5,linewidth=10, label = r"slope: $d_f	|_i \approx 2.0$" )
	inset_ax.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2)
	inset_ax.set_ylabel(r'$\ln$($N$/max($N$))', fontsize=50)
	inset_ax.set_xlabel(r'$\ln(l/l_i)$' , fontsize=50)
	# Plot tangent line at l_i with slope fracDim_at_l_i
	# inset_ax.plot(np.log(l/l_i[ind]), np.log(l/l_i[ind])*fracDim_at_l_i + 5, 'm-', linewidth=4, label=f'Tangent (slope={fracDim_at_l_i:.2f})')
	inset_ax.legend(loc ='lower right', fontsize=40)
	ax.legend(fontsize=60, loc = "lower right")
	plt.show()

	plt.figure(figsize=(16, 5))
	# ln-ln plot (natural logarithm)
	plt.subplots_adjust(top=0.85)  # Increase the gap at the top of the figure
	plt.plot(np.log(l/l_i[ind]), np.log(abs(h1_z)), "bo", alpha=0.1, markersize=20) # , label = r'$\ln(|h_{' + str(orderOfMom) + ',z}/h_{' + str(orderOfMom) + ',z}|_i|)$')
	if len(l_pre) > 1:
		plt.plot(np.log(l_pre), np.log(pl_model_pre_1) + 6.3, 'r--', linewidth=10, alpha=0.7, label=r'slope: $\eta_{1,\perp,<} \approx 4.3$')
	# Post-inflection
	if len(l_post) > 1:		
		plt.plot(np.log(l_post), np.log(pl_model_post_1) - 18.3, 'g--', linewidth=10, alpha=0.7, label=r'slope: $\eta_{1,\perp,>} \approx -2.4$')
	plt.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2 )
	plt.axvline(x=np.log(l_min_curv[ind]/l_i[ind]), color='y', linestyle='-', linewidth=10, alpha=0.2)
	plt.text(np.log(l_min_curv[ind]/l_i[ind])-0.08, np.min(np.log(abs(h1_z))) + 5.6, r'$\ln(l_{>}/l_i$)', color='y', fontsize=40, rotation=90, va='bottom', ha='center')
	plt.xlabel(r'$\ln(l/l_i)$', fontsize=60)
	plt.ylabel(r'$\ln(|h_{' + str(orderOfMom) + ',z}/h_{' + str(orderOfMom) + ',z}|_i|)$', fontsize=60)
	# plt.ylim([np.min(np.log(abs(h1_z))) - 1, np.max(np.log(abs(h1_z))) + 1])
	plt.legend(fontsize=50, loc = "lower right")
	plt.show()
	

	'''
	
	'''
	# plt.plot(l, dn_phi * np.mean(np.diff(l_phi)))
	# plt.plot(l, dn_model_phi * np.mean(np.diff(l_phi)))
	# plt.plot(l, fracDim)
	# plt.axvline(x = l[ind_l_i])
	# plt.show()
	# exit()
	'''

	fig, ax = plt.subplots()
	ax.set_title(PDB)
	ax.plot(l, n, 'ko', alpha = 0.015, markersize = 15, label = r'$\tilde{n}')
	ax.plot(l, n_model, 'k-', linewidth = 4, label = r'$n$')
	ax.plot(l_pho, n_pho, 'bo', alpha = 0.015, markersize = 15, label = r'$\tilde{n}_{\mathrm{-}}$')
	ax.plot(l_pho, n_model_pho, 'b-', linewidth = 4, label = r'$n_{-}$')
	ax.plot(l_phi, n_phi, 'ro', alpha = 0.015, markersize = 15, label = r'$\tilde{n}_{\mathrm{+}}$')
	ax.plot(l_phi, n_model_phi, 'r-', linewidth = 4, label = r'$n_{+}$')
	
	ax.set_xlabel(r'Scale $l$ [\AA]')
	ax.set_ylabel(r'Cumulative atomic number')
	
	from matplotlib.lines import Line2D # type: ignore
	legend_elements = 	[
						
						Line2D([0], [0], marker='o',  markeredgecolor='k', label=r'$n,$', markerfacecolor='k', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}$',linewidth=5, alpha=0.5, color = 'k', linestyle='-'),
						Line2D([0], [0], marker='o',  markeredgecolor='b', label=r'$n_{\mathrm{-}}$', markerfacecolor='b', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}_{\mathrm{-}}$',linewidth=5, alpha=0.5, color = 'b', linestyle='-'),
						Line2D([0], [0], marker='o',  markeredgecolor='r', label=r'$n_{\mathrm{+}}$', markerfacecolor='r', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}_{\mathrm{+}}$',linewidth=5, alpha=0.5, color = 'r', linestyle='-')
						
						]

	ax.set_xlim([0, max(l)+10])				
	ax.legend(loc = 'upper left', handles=legend_elements, fontsize = 32)
	
	twin_ax = ax.twinx()
	
	l_pre = l[np.arange(0, ind_l_i - 15, 1).astype(int)]
	coeff_pre = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_pre', infoBlock = 'coeff')[ind]
	exp_pre = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_pre', infoBlock = 'exponent')[ind]
	pl_model_pre = np.exp(exp_pre*np.log(l_pre) + coeff_pre) 
	
	l_post = l[np.arange(ind_l_i + 15, Match(l_min_curv[ind],l) + 1, 1).astype(int)]
	coeff_post = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'coeff')[ind]
	exp_post = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'exponent')[ind]
	pl_model_post = np.exp(exp_post*np.log(l_post) + coeff_post) 
	
	twin_ax.plot(l, abs(h_z), 'yo', alpha = 0.15, markersize = 20, linestyle = None)
	twin_ax.plot(l_pre, pl_model_pre, 'g--', linewidth = 8, alpha = 0.5)
	twin_ax.plot(l_post, pl_model_post, 'r--', linewidth = 8, alpha = 0.5)
	ax.axvline(x = l_i[ind], color = 'm', linestyle = '--', linewidth = 5, alpha = 0.2)
	
	existing_ticks = [l_min[ind], l_max_curv[ind], l_i[ind], l_min_curv[ind], l_max[ind]] # ax.get_xticks().tolist()
	existing_labels = ['$R$', '$l_{min}$', '$l_{i}$', '$l_{max}$', '$L$']

	# Set the modified ticks and labels
	ax.set_xticks(existing_ticks)
	ax.set_xticklabels(existing_labels)
	ax.set_xlim([0,max(l)+1])
	twin_ax.set_ylabel(r'$|h_{' + str(orderOfMom) + ',z}|$ [kcal $\cdot$ \AA]')
	
	legend_elements = [Line2D([0], [0], marker='o',  markeredgecolor='y', label=r'$|h_{' + str(orderOfMom) + ',z}|$', markerfacecolor='y', markersize=13, alpha=0.5, linestyle='')]
	twin_ax.legend(loc = 'center right', handles=legend_elements, fontsize = 35)
	
	plt.show()
	'''




def Plot_TracesExamples_AG(ind, scales, inds_l_i, l_i, fracDim_i, nrOfAtoms, statMod, orderOfMom, hydrMoments, exponentsDomains, l_min, l_max, l_min_curv, l_max_curv):
	
	l 					= scales[ind]
	ind_l_i				= inds_l_i[ind].astype(int)
	n	 				= nrOfAtoms[ind][0]
	# dn				= np.append(n[0],np.diff(n)/np.diff(l))
	modType 			= statMod[ind][0][15]
	params	 			= [statMod[ind][0][0], statMod[ind][0][2], statMod[ind][0][4], statMod[ind][0][6]]
	print(params)
	n_model		 		= GeomModel(l, params, modType) 
	fracDim_at_l_i      = fracDim_i[ind]


	print("\n Fractal dim is:", fracDim_i[ind])

	n_pho				= nrOfAtoms[ind][1]
	inds_pho 			= np.where(n_pho > 0) 
	l_pho				= l[inds_pho]
	n_pho				= n_pho[inds_pho]
	# dn_pho			= np.append(n_pho[0],np.diff(n_pho)/np.diff(l_pho))
	modType_pho 		= statMod[ind][1][15]
	params_pho 			= [statMod[ind][1][0], statMod[ind][1][2], statMod[ind][1][4], statMod[ind][1][6]]
	n_model_pho		 	= GeomModel(l_pho, params_pho, modType_pho)*max(n_pho) # equiv to "distance"
	# dn_model_pho		= Tools.GeomModel(l_pho, params_pho, der = 1)*max(n_pho) 
		
	n_phi				= nrOfAtoms[ind][2]
	inds_phi 			= np.where(n_phi > 0)
	l_phi				= l[inds_phi]
	n_phi				= n_phi[inds_phi]
	dn_phi				= np.append(n_phi[0],np.diff(n_phi)/np.diff(l_phi))
	modType_phi 		= statMod[ind][2][15]
	params_phi 			= [statMod[ind][2][0], statMod[ind][2][2], statMod[ind][2][4], statMod[ind][2][6]]
	n_model_phi		 	= GeomModel(l_phi, params_phi, modType_phi)*max(n_phi) # equiv to "distance"
	dn_model_phi		= GeomModel(l_phi, params_phi, der = 1)*max(n_phi)
	
	h1_z 				= GetHydrMom(hydrMoments, ind, 'pathic', 1, 'z')
	h2 					= GetHydrMom(hydrMoments, ind, 'pathic', 2)
	h3_z 				= GetHydrMom(hydrMoments, ind, 'pathic', 3, 'z')
	h4 					= GetHydrMom(hydrMoments, ind, 'pathic', 4)
	h5 					= GetHydrMom(hydrMoments, ind, 'pathic', 5, 'z')
	h6 					= GetHydrMom(hydrMoments, ind, 'pathic', 6)


	h1_z = h1_z/h1_z[ind_l_i]
	l_pre = l[np.arange(0, ind_l_i , 1).astype(int)] / l_i[ind]
	l_post = l[np.arange(ind_l_i , Match(l_min_curv[ind],l) + 50, 1).astype(int)] / l_i[ind]

	# for h1_z
	coeff_pre_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_pre', infoBlock = 'coeff')[ind]
	exp_pre_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_pre', infoBlock = 'exponent')[ind]
	pl_model_pre_1 = np.exp(exp_pre_1*np.log(l_pre) + coeff_pre_1) 
	
	coeff_post_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'coeff')[ind]
	exp_post_1 = ScalingInfo(exponentsDomains, orderOfhydrMom = 1, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'exponent')[ind]
	pl_model_post_1 = np.exp(exp_post_1*np.log(l_post) + coeff_post_1)

	fig, ax = plt.subplots()
	ax.plot(np.log(l/l_i[ind]), n, "bo", alpha=0.1, markersize=20, label = r"$N$")
	ax.plot(np.log(l/l_i[ind]), n_model*max(n), "r--", alpha = 0.7, linewidth=10, label = r"$n_{\nu \approx 1}$")
	ax.set_ylabel(r'$N$ [atom]' , fontsize=60)
	ax.set_xlabel(r'$\ln(l/l_i)$' , fontsize=60)
	ax.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2)
	inset_ax = fig.add_axes([0.23, 0.42, 0.39, 0.44]) 
	inset_ax.plot(np.log(l/l_i[ind]), np.log(n), "bo", alpha=0.1, markersize=20)
	inset_ax.plot(np.log(l/l_i[ind]), np.log(l/l_i[ind])*fracDim_at_l_i + 10, "m--", alpha=0.5,linewidth=10, label = r"slope: $d_f	|_i \approx 2.1$" )
	inset_ax.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2)
	inset_ax.set_ylabel(r'$\ln$($N$/max($N$))', fontsize=50)
	inset_ax.set_xlabel(r'$\ln(l/l_i)$' , fontsize=50)
	# Plot tangent line at l_i with slope fracDim_at_l_i
	# inset_ax.plot(np.log(l/l_i[ind]), np.log(l/l_i[ind])*fracDim_at_l_i + 5, 'm-', linewidth=4, label=f'Tangent (slope={fracDim_at_l_i:.2f})')
	inset_ax.legend(loc ='lower right', fontsize=40)
	ax.legend(fontsize=60, loc = "lower right")
	plt.show()

	plt.figure(figsize=(16, 5))
	# ln-ln plot (natural logarithm)
	plt.subplots_adjust(top=0.85)  # Increase the gap at the top of the figure
	plt.plot(np.log(l/l_i[ind]), np.log(abs(h1_z)), "bo", alpha=0.1, markersize=20, label = r'$\ln(|h_{' + str(orderOfMom) + ',\perp}/h_{' + str(orderOfMom) + ',\perp}|_i|)$')
	if len(l_pre) > 1:
		plt.plot(np.log(l_pre), np.log(pl_model_pre_1) + 4.8, 'r--', linewidth=10, alpha=0.7, label=r'slope: $\eta_{1,\perp,<} \approx 3.6$')
	# Post-inflection
	if len(l_post) > 1:		
		plt.plot(np.log(l_post), np.log(pl_model_post_1) + 6.42, 'g--', linewidth=10, alpha=0.7, label=r'slope: $\eta_{1,\perp,>} \approx 3.9$')
	plt.axvline(x=0, color='m', linestyle='-', linewidth=10, alpha=0.2 )
	plt.axvline(x=np.log(l_min_curv[ind]/l_i[ind]), color='y', linestyle='-', linewidth=10, alpha=0.2)
	plt.text(np.log(l_min_curv[ind]/l_i[ind])-0.08, np.min(np.log(abs(h1_z))) + 6.4, r'$\ln(l_{>}/l_i$)', color='y', fontsize=40, rotation=90, va='bottom', ha='center')
	plt.xlabel(r'$\ln(l/l_i)$', fontsize=60)
	plt.ylabel(r'$\ln(|h_{' + str(orderOfMom) + ',\perp}/h_{' + str(orderOfMom) + ',\perp}|_i|)$', fontsize=60)
	
	# plt.ylim([np.min(np.log(abs(h1_z))) - 1, np.max(np.log(abs(h1_z))) + 1])
	plt.legend(fontsize=50, loc = "lower right")
	plt.show()
	

	'''
	
	'''
	# plt.plot(l, dn_phi * np.mean(np.diff(l_phi)))
	# plt.plot(l, dn_model_phi * np.mean(np.diff(l_phi)))
	# plt.plot(l, fracDim)
	# plt.axvline(x = l[ind_l_i])
	# plt.show()
	# exit()
	'''

	fig, ax = plt.subplots()
	ax.set_title(PDB)
	ax.plot(l, n, 'ko', alpha = 0.015, markersize = 15, label = r'$\tilde{n}')
	ax.plot(l, n_model, 'k-', linewidth = 4, label = r'$n$')
	ax.plot(l_pho, n_pho, 'bo', alpha = 0.015, markersize = 15, label = r'$\tilde{n}_{\mathrm{-}}$')
	ax.plot(l_pho, n_model_pho, 'b-', linewidth = 4, label = r'$n_{-}$')
	ax.plot(l_phi, n_phi, 'ro', alpha = 0.015, markersize = 15, label = r'$\tilde{n}_{\mathrm{+}}$')
	ax.plot(l_phi, n_model_phi, 'r-', linewidth = 4, label = r'$n_{+}$')
	
	ax.set_xlabel(r'Scale $l$ [\AA]')
	ax.set_ylabel(r'Cumulative atomic number')
	
	from matplotlib.lines import Line2D # type: ignore
	legend_elements = 	[
						
						Line2D([0], [0], marker='o',  markeredgecolor='k', label=r'$n,$', markerfacecolor='k', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}$',linewidth=5, alpha=0.5, color = 'k', linestyle='-'),
						Line2D([0], [0], marker='o',  markeredgecolor='b', label=r'$n_{\mathrm{-}}$', markerfacecolor='b', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}_{\mathrm{-}}$',linewidth=5, alpha=0.5, color = 'b', linestyle='-'),
						Line2D([0], [0], marker='o',  markeredgecolor='r', label=r'$n_{\mathrm{+}}$', markerfacecolor='r', markersize=13, alpha=0.5, linestyle=''),
						Line2D([0], [0], label=r'$\tilde{n}_{\mathrm{+}}$',linewidth=5, alpha=0.5, color = 'r', linestyle='-')
						
						]

	ax.set_xlim([0, max(l)+10])				
	ax.legend(loc = 'upper left', handles=legend_elements, fontsize = 32)
	
	twin_ax = ax.twinx()
	
	l_pre = l[np.arange(0, ind_l_i - 15, 1).astype(int)]
	coeff_pre = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_pre', infoBlock = 'coeff')[ind]
	exp_pre = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_pre', infoBlock = 'exponent')[ind]
	pl_model_pre = np.exp(exp_pre*np.log(l_pre) + coeff_pre) 
	
	l_post = l[np.arange(ind_l_i + 15, Match(l_min_curv[ind],l) + 1, 1).astype(int)]
	coeff_post = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'coeff')[ind]
	exp_post = ScalingInfo(exponentsDomains, orderOfhydrMom = orderOfMom, component = 'z', interval = 'powerLaw_infl2', infoBlock = 'exponent')[ind]
	pl_model_post = np.exp(exp_post*np.log(l_post) + coeff_post) 
	
	twin_ax.plot(l, abs(h_z), 'yo', alpha = 0.15, markersize = 20, linestyle = None)
	twin_ax.plot(l_pre, pl_model_pre, 'g--', linewidth = 8, alpha = 0.5)
	twin_ax.plot(l_post, pl_model_post, 'r--', linewidth = 8, alpha = 0.5)
	ax.axvline(x = l_i[ind], color = 'm', linestyle = '--', linewidth = 5, alpha = 0.2)
	
	existing_ticks = [l_min[ind], l_max_curv[ind], l_i[ind], l_min_curv[ind], l_max[ind]] # ax.get_xticks().tolist()
	existing_labels = ['$R$', '$l_{min}$', '$l_{i}$', '$l_{max}$', '$L$']

	# Set the modified ticks and labels
	ax.set_xticks(existing_ticks)
	ax.set_xticklabels(existing_labels)
	ax.set_xlim([0,max(l)+1])
	twin_ax.set_ylabel(r'$|h_{' + str(orderOfMom) + ',z}|$ [kcal $\cdot$ \AA]')
	
	legend_elements = [Line2D([0], [0], marker='o',  markeredgecolor='y', label=r'$|h_{' + str(orderOfMom) + ',z}|$', markerfacecolor='y', markersize=13, alpha=0.5, linestyle='')]
	twin_ax.legend(loc = 'center right', handles=legend_elements, fontsize = 35)
	
	plt.show()
	'''