import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from uncertainties import ufloat

from uncertainties.umath import *
from scipy.signal import savgol_filter

import os
PDB = os.getcwd()[-4:]

import ModelParameters
import math
from scipy.interpolate import splrep, BSpline
from scipy.stats import gamma
from scipy.stats import norm

from matplotlib.pyplot import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':16})
rc('text', usetex=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, f1_score

from scipy.stats import gaussian_kde

'''
		Cumulative atom number modeling
'''
##	Model Richards	##
def ResidualsRichards(params, x, data, der = 0):
	
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['zeta']*v['nu']*(x - v['l_i']))
	
	# Richards model
	if (der == 0):
		model = v['K']*(1 + v['nu']*phi)**(-1/v['nu'])
	if (der == 1):
		model = v['K']*v['zeta']*v['nu']*phi*(v['nu']*phi + 1)**(-1./v['nu'] - 1)
	if (der == 2):
		phi_inv = np.exp(v['zeta']*v['nu']*(x - v['l_i']))
		model = -v['K'] * (v['zeta']**2) * (v['nu']**2) * (phi_inv - 1) / ( ((1 + v['nu']*phi)**(1/v['nu'])) * (v['nu'] + phi_inv)**2 )
	if (der == 'log0'):
		phi_inv = np.exp(v['zeta']*v['nu']*(x - v['l_i']))  
		model =  x*v['zeta']*v['nu'] / ( phi_inv + v['nu'] ) 
	if (der == 'log1'):
		phi_inv = np.exp(v['zeta']*v['nu']*(x - v['l_i']))
		model = -x*v['zeta']*v['nu'] * (phi_inv - 1) / ( phi_inv + v['nu'] )
	
	# Return residuals
	return model - data

##	Model Logistic	##
def ResidualsLogistic(params, x, data, der = 0):
    
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['zeta']*(x - v['l_i']))
	
	# Logistic model
	if (der == 0):
		model = v['K']*(1 + phi)**( -1 )
	if (der == 1):
		model = v['K']*v['zeta']*phi*(1 + phi)**(-2)
	if (der == 2):
		phi_inv = 1/phi
		model = - v['K']*(v['zeta']**2)*phi_inv*(phi_inv - 1) / (phi_inv + 1)**3
	if (der == 'log0'):
		phi_inv = 1/phi
		model = x*v['zeta'] / ( phi_inv + 1 ) 
	if (der == 'log1'):
		model = x*v['zeta'] / ( ( np.exp(v['zeta']*v['l_i']) - np.exp(v['zeta']*x) ) /( np.exp(v['zeta']*v['l_i']) + np.exp(v['zeta']*x)) )
	
	# Return residuals
	return model - data
		
##	Model Gompertz ##
def ResidualsGompertz(params, x, data, der=0):
    
	# Get an ordered dictionary of parameter values
	v = params.valuesdict()
	phi = np.exp(-v['a_gomp']*(x - v['l_i']))
	
	# Gompertz model
	if (der == 0):
		model = v['K']*np.exp(- phi) 
	if (der == 1):
		model = v['a_gomp']*v['K']*np.exp(- phi - v['a_gomp']*(x - v['l_i'])) 
	if (der == 2):
		model = v['a_gomp']*v['K']*(v['a_gomp']*phi - v['a_gomp'])*np.exp(- phi - v['a_gomp']*(x - v['l_i'])) 
	if (der == 'log0'):
		model = x*v['a_gomp']*np.exp(v['a_gomp']*(v['l_i'] - x))
	if (der == 'log1'):
		model = x*v['a_gomp']*(phi - 1)
		
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
			[nu_initGuess, 		0.001*nu_initGuess, 	10*nu_initGuess]]		# typical values are 1

##	Model trace ## 
def GeomModel(x, InputParams, modelType=1, der=0):
	
	from lmfit import Parameters

	params = Parameters()
	params.add('K', value = InputParams[0])
	params.add('l_i', value = InputParams[2])
	if (modelType == 1):
		params.add('zeta', value = InputParams[1])
		params.add('nu', value = InputParams[3])
		model = ResidualsRichards(params, x, np.zeros(len(x)), der)
	if (modelType == 2):
		params.add('zeta', value = InputParams[1])
		params.add('nu', value = InputParams[3])
		model = ResidualsLogistic(params, x, np.zeros(len(x)), der)
	if (modelType == 3):
		params.add('a_gomp', value = InputParams[1])
		model = ResidualsGompertz(params, x, np.zeros(len(x)), der)
	
	return model

##	Model parameters ##
def StatModelParameters(x, y):
	
	from lmfit import Minimizer, Parameters, fit_report

	# Initialize the solver
	initParam_val = initGuesses(x, max(y))
	
	# Init param
	params_richards = Parameters()
	params_richards.add('K', value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_richards.add('zeta', value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
	params_richards.add('l_i', value = initParam_val[2][0], min = initParam_val[2][1], max = initParam_val[2][2])
	params_richards.add('nu', value = initParam_val[3][0], min = initParam_val[3][1], max = initParam_val[3][2])
	
	params_logistic = Parameters()
	params_logistic.add('K', value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_logistic.add('zeta', value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
	params_logistic.add('l_i', value = initParam_val[2][0], min = initParam_val[2][1], max = initParam_val[2][2])
	
	params_gompertz = Parameters()
	params_gompertz.add('K',   value = initParam_val[0][0], min = initParam_val[0][1], max = initParam_val[0][2])
	params_gompertz.add('a_gomp',   value = initParam_val[1][0], min = initParam_val[1][1], max = initParam_val[1][2])
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
	results.append([fit_richards.params['K'].value, 		fit_richards.params['K'].stderr, 		# A 					[0,1]
					fit_richards.params['zeta'].value, 		fit_richards.params['zeta'].stderr, 		# a 					[2,3]
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
	results.append([fit_logistic.params['K'].value, 		fit_logistic.params['K'].stderr, 		# A 					[0,1]
					fit_logistic.params['zeta'].value, 		fit_logistic.params['zeta'].stderr, 		# a 					[2,3]
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
	results.append([fit_gompertz.params['K'].value, 		fit_gompertz.params['K'].stderr, 		# A 					[0,1]
					fit_gompertz.params['a_gomp'].value, 	fit_gompertz.params['a_gomp'].stderr, 	# a 					[2,3]
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
	ind_modSelect = np.where(aics == min(aics))[0][0]		

	_ = {}
	# Fill this dict with what you really need ..
	_['K'] = results[ind_modSelect][0]
	_['A_unc'] = results[ind_modSelect][1]
	_['zeta'] = results[ind_modSelect][2]
	_['a_unc'] = results[ind_modSelect][3]
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
	
	_['modType'] = results[ind_modSelect][22]

	
	return _
	
## Additional model information ##
def GeomModelDomains(params, modType):

	
	if (params['K'].stderr != None):	
		A_ = ufloat(params['K'].value, params['K'].stderr)
	else:	
		A_ = ufloat(params['K'].value, 0)
		
	if (params['l_i'].stderr != None):	
		l_i_ = ufloat(params['l_i'].value, params['l_i'].stderr)
	else:
		l_i_ = ufloat(params['l_i'].value, 0)
	
	if (modType != 3):
		
		if (params['zeta'].stderr != None):	
			a_ = ufloat(params['zeta'].value, params['zeta'].stderr)
		else:
			a_ = ufloat(params['zeta'].value, 0)
		
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
		
		if (params['a_gomp'].stderr != None):	
			a_gomp_ = ufloat(params['a_gomp'].value, params['a_gomp'].stderr)
		else:
			a_gomp_ = ufloat(params['a_gomp'].value, 0)
			
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
	tck = splrep(x, y)
	return BSpline(*tck)(xx)

##	Coarse derivative ##
def CoarseDifferentiation(x, y, window):

	y_smooth = savgol_filter(y, window, ModelParameters.POLYORDER)
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
	
##	Get sliding window ##
def GetSlidingWindow(a, nu, modType):
	
	if (modType != 3):
		window 	= math.ceil((1./(a*nu)) / ModelParameters.HOLE_RES)
	else:
		window	= math.ceil((1./(a)) / ModelParameters.HOLE_RES)	
	
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

##	Get the hydr mom data ##
def getHydrMom(array, pp_index, typeOfAtom = 'pathic', orderOfhydrMom = 0, coordinate = 'z'):
	
	
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
	
	if (len(data_d1) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: d1 domain disapears! \n') 
	
	if (len(data_infl1) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: infl1 domain disapears! \n')
		
	if (len(data_infl2) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: infl2 domain disapears! \n')
	
	if (len(data_d2) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: d2 domain disapears! \n')
	
	if (len(data_pre_infl) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: pre domain disapears! \n')
	
	if (len(data_post_infl) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n WARNING: post domain disapears! \n') 
			
	# To return
	indices_ret 	= [ind_d1, 		ind_l_i, 		ind_d2]
	l_ret 			= [l_d1,  		l_infl1,		l_infl2,		l_d2, 	 	l_pre_infl, 		l_post_infl]
	data_ret	 	= [data_d1, 	data_infl1, 	data_infl2, 	data_d2, 	data_pre_infl, 		data_post_infl]
	
	return [indices_ret, l_ret, data_ret]

## Scaling behavior over selected interval ##
def ScalingBehavior(l, data, l_domains):
			
	data_sets = GetGeomIndices(l, data, l_domains)
	nrOfSegments = len(data_sets[2])
	logLogFit = []

	for i in range(nrOfSegments):
		logLogFit.append(ScalingAnalysis(data_sets[1][i], abs(data_sets[2][i])))
		
	'''
	for i in range(nrOfSegments):
	
		expModelApprox = np.exp(logLogFit[i][0]*np.log(data_sets[1][i]) + logLogFit[i][1])
		plt.plot(data_sets[1][i], abs(data_sets[2][i]), 'bo', alpha = 0.05)
		plt.plot(l, abs(data), "bo", alpha = 0.1, markersize = 4)
		plt.plot(data_sets[1][i], expModelApprox, '--', linewidth = 2)
		
		plt.show()
	'''

	return logLogFit

##	Scaling analysis ##
def ScalingAnalysis(x, y):
	
	if (any(y) < 0):
		exit('\n\n Exiting smoothly .. you got zeros or negative numbers in the wrong place .. \n\n')
	
	if (len(y) < ModelParameters.MIN_DOMAIN_SIZE):
		print('\n Domain vanishingly small: ', len(y), len(x))
		return [None]

	coeffs, cov = np.polyfit(np.log(x), np.log(y), deg=1, cov=True)
	model = np.exp(coeffs[0]*np.log(x) + coeffs[1])
	res = stats.pearsonr(np.log(y), coeffs[0]*np.log(x) + coeffs[1])

	return [coeffs[0], coeffs[1], res[0], res[1], np.sqrt(np.diag(cov)), FittingError(y, model), FittingError(y, model, TYPE='SDAFE')]

##	Retrieve scaling information ##
def ScalingInfo(array, orderOfhydrMom = 0, component = 'z', dataType =  'powerLaw_pre', infoBlock = 'exp'):
	
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
	if (dataType ==  'powerLaw_preInfl1'):
		data_index = 0
	if (dataType ==  'powerLaw_infl1'):
		data_index = 1
	if (dataType ==  'powerLaw_infl2'):
		data_index = 2
	if (dataType ==  'powerLaw_postInfl2'):
		data_index = 3
	if (dataType ==  'powerLaw_pre'):
		data_index = 4
	if (dataType ==  'powerLaw_post'):
		data_index = 5
	
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
		unchanges for l larger than l_lag, where l_lag is the "lag" scale. 

		Note that decomposition is NOT always possible. 
		In that case, we assign phi and pho to + and -, respectively. 
'''
## Decompose ##
def Decompose(h_pho, h_phi, ind_l_cutOff):

	# Successful decomposition: above the lag scale the sign does not change.
	# The part before the l_lag is also considered, if singularties for l<l_lag are not dramatic
	# In fact they are not expected to be, since we are dealing with first-order "jumps" over zero and not continuous zero-crossings.	
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
		print("\n\n\n Decomp. Ansatz Violation! \n\n\n")
		h_plus = h_phi
		h_minus = h_pho

	return [h_plus, h_minus]


'''
		Variants processing 
'''
##	Variants listing ##
def VariantsListing(column):

	i = 0
	var_list = []
	for x in range(len(column)): 
		if (i == 0):
			print(' Collecting', column[x].value, 'related variants! ')
			# continue
		else:
			if (column[x].value == None):
				break
			var_list.append(column[x].value)
		i += 1

	return var_list

##	Variants grouping ##
def GroupVariants(varInfo_type, groupType):
	
	'''
	
			Gnomad/ClinVar
	
	'''
	# VUS
	inds_vus = np.where(varInfo_type == 'Uncertain significance')[0]
	# unknown
	inds_unknown = np.where(varInfo_type == 'not provided')[0]
	# benign
	inds_benign = np.where(varInfo_type == 'Benign')[0]
	inds_benignLikelyBenign = np.where(varInfo_type == 'Benign/Likely benign')[0]
	inds_likelyBenign = np.where(varInfo_type == 'Likely benign')[0]
	# pathogenic
	inds_path = np.where(varInfo_type == 'Pathogenic')[0]
	inds_pathLikelyPath = np.where(varInfo_type == 'Pathogenic/Likely pathogenic')[0]
	inds_likelyPath = np.where(varInfo_type == 'Likely pathogenic')[0]
	inds_conflInterPath = np.where(varInfo_type == 'Conflicting interpretations of pathogenicity')[0]
	
	
	'''
	
			Pain Disease Phenotypes
	
	'''
	inds_iem = np.where(varInfo_type == 'IEM')[0]
	inds_sfn = np.where(varInfo_type == 'SFN')[0]
	inds_pepd = np.where(varInfo_type == 'PEPD')[0]
	inds_neutral = np.where(varInfo_type == 'Neutral')[0]
	inds_lof = np.where(varInfo_type == 'LoF')[0]
	

	'''
	
			Classification Status
	
	'''
	inds_classificationStatus = np.where(varInfo_type == 'missclassified')[0]

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

	# model = RandomForestClassifier(n_estimators=100, random_state=42)

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
def KFoldsClassifier(X, y, X_other = [], X_unseen = [], y_unseen = [], method = 'LogReg', kernel = None, chooseThreshold = "f1"):
	
	# Length of data vector
	n_data = len(y)

	# Initialize
	probs_class_0 = []	
	model_class = []
	features_importance = []
	acc_auc_f1_optThres = []		
	
	if (len(X_other) != 0):	
		probs_class_0_other = [[] for _ in range(len(X_other))]
	
	if (len(X_unseen) != 0 and len(y_unseen) != 0): 
		if (len(y_unseen) == len(X_unseen)):
			acc_auc_f1_optThres_unseen = []	

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
				
				# 'Other' data
				if (len(X_other) != 0):
					for i in range(len(X_other)):
						_ = modelRes[-1].predict_proba(modelRes[-2].transform(X_other[i]))[:,0]
						probs_class_0_other[i].append(_) # get probs that "others" belong to class_0 
				
				# 'Unseen' data
				if (len(X_unseen) != 0):
					
					# Trasform
					X_unseen_trans = modelRes[-2].transform(X_unseen)
					# Apply the model on the unseen data:
					# get the acc
					acc_unseen = modelRes[-1].score(X_unseen_trans, y_unseen)
					# get the auc
					y_unseen_probs = modelRes[-1].predict_proba(X_unseen_trans) # get probs that "unseen" belong to class_0
					auc_unseen = metrics.roc_auc_score(y_unseen, y_unseen_probs[::,1])
					# f1
					f1_unseen = f1_score(y_unseen, modelRes[-1].predict(X_unseen_trans), average='weighted')

					fpr, tpr, thresholds = roc_curve(y_unseen, y_unseen_probs[:,1])
					f1_scores = [f1_score(y_unseen, y_unseen_probs[:,1] >= threshold) for threshold in thresholds]
					optimal_idx = np.argmax(f1_scores)
					optimal_threshold = thresholds[optimal_idx]

					## Append
					acc_auc_f1_optThres_unseen.append([acc_unseen, auc_unseen, f1_unseen, optimal_threshold])
			
			## Append 	
			# The length of these array is (K - 1) * NUM_OF_TRAININGS
			probs_class_0.append(probs_split)
			model_class.append(class_split)

	# Return: cover all different cases
	if (len(X_unseen) != 0 and len(X_other) != 0):
		return [probs_class_0, model_class, acc_auc_f1_optThres, features_importance, probs_class_0_other, acc_auc_f1_optThres_unseen]
		
	if (len(X_other) != 0):
		return [probs_class_0, model_class, acc_auc_f1_optThres, features_importance, probs_class_0_other]
		
	if (len(X_unseen) != 0):
		return [probs_class_0, model_class, acc_auc_f1_optThres, features_importance, acc_auc_f1_optThres_unseen]
	
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

	mode = Mode(data)

	return [np.mean(data, axis = axis), 
		 	np.std(data, axis = axis), 
			np.median(data, axis = axis), 
			np.percentile(data, q=p_left, axis = axis), 
			np.percentile(data, q=100 -p_left,axis = axis),
			# mode and min, max values are also useful for data sampled from observables being singular
			mode,
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



'''
		Plotters 
'''
## Plot cumulative atom number ##
def Plot_CumulativeAtomNumber(data_n, data_model, min_inds_l_i, max_inds_l_i, hist_data, alpha_percentile = 0.5, n_max_percentile = 5, offset_xlim = 5, ytitle = r'$N$ [atom]', xtitle = r'Scale index', xticks = [], statsType = 'median'):

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

	label_y = r'$N$'
	label_model_y = r'$n$'
	
	## Main figure
	fig, ax = plt.subplots()
	x = np.arange(ModelParameters.N_SCALES)

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
						
						Line2D([0], [0], marker='o',  markeredgecolor='b', markerfacecolor='b', markersize=13, alpha=0.5, linestyle='', label=label_y),
						Line2D([0], [0], linewidth = 5, alpha=0.5, color='r', linestyle='--', label=label_model_y)
						
						]
	
	ax.axvspan(min_inds_l_i, max_inds_l_i, alpha=0.1, color='m')

	offset_ylim = 250
	ax.set_ylim([min(min(y), min(y_model)) - offset_ylim, max(max(y), max(y_model)) + offset_ylim])
	ax.set_xlim([0, ModelParameters.N_SCALES + offset_xlim])
	ax.legend(loc = 'upper left', handles=legend_elements, fontsize = 70) 
	ax.set_title('Critical inflection regime', color = "m", fontsize = 60)
	ax.set_xticks(xticks)
	ax.set_xlim([0,ModelParameters.N_SCALES])
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
def Plot_AtomicHydropathicEnergy(data_m0, data_m0_pho, data_m0_phi, data_model, min_inds_l_i, max_inds_l_i, hist_data, alpha_percentile = 0.5, n_max_percentile = 5, offset_xlim = 5, ytitle = r'$| h_{0} | / N$ [kcal/atom]', xtitle = r'Scale index', xticks = [], statsType = 'median'):

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
	x = np.arange(ModelParameters.N_SCALES)

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
	ax.set_xlim([0, ModelParameters.N_SCALES + offset_xlim])
	ax.set_ylim([-0.8, 0.8])
	ax.set_xticks(xticks)
	ax.set_xlabel(xtitle)
	ax.set_ylabel(ytitle, fontsize = 60)
	ax.set_xlim([0,ModelParameters.N_SCALES])
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
def Plot_DipoleMoment(data_z_pos, data_z_ES, data_z_neg, data_z_IS, min_inds_l_i, max_inds_l_i, exp_data, PC_data, orderOfMom = 1, alpha_percentile = 0.5, n_max_percentile = 5, ytitle = r'$h_{1,\perp}$ (norm.)', xtitle = r'Scale index', statsType = 'median', ticks_step = 6, x_min = -38, x_max = 38):


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

	label_y_pos = r'$  h_{' + str(orderOfMom) + r',\perp, \uparrow} $'
	label_y_ES = r'$  h_{' + str(orderOfMom) + r',\perp}$ (ES)'
	label_y_neg = r'$  h_{' + str(orderOfMom) + r',\perp, \downarrow}$'
	label_y_IS = r'$  h_{' + str(orderOfMom) + r',\perp}$ (IS)'
	
	## Main figure
	fig, ax = plt.subplots()

	x = np.arange(ModelParameters.N_SCALES)
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
	ax.set_xlim([0, ModelParameters.N_SCALES])
	ax.set_ylim([-2,2])
	ax.set_xlim([0,ModelParameters.N_SCALES])
	# ax.set_ylim([-6,6])
	ax.axhline(y = 1,linestyle = "--", alpha = 0.25, color = "k",linewidth = 5)
	ax.axhline(y = 0,linestyle = "-", alpha = 0.25, color = "k",linewidth = 5)
	ax.axhline(y = -1, linestyle = "--", alpha = 0.25, color = "k", linewidth = 5)
	ax.set_xlabel(xtitle, fontsize = 65)
	ax.set_ylabel(ytitle, fontsize = 65)
	ax.legend(loc = 'upper left', fontsize = 50)
	plt.show()
	plt.close()

	# VSD
	
	# for prokaryotes summary try:
	n_bins_PD = 25
	n_bins_PD_largePC = 30
	n_bins_VSD = 60
	n_bins_VSD_largePC = 50

	# for eukaryotes try:
	# n_bins_PD = 10
	# n_bins_PD_largePC = 12
	# n_bins_VSD = 15
	# n_bins_VSD_largePC = 18

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
	# # VSD hist (ES)
	indices = np.where((np.array(princ_coords) >= 0))[0]
	exps_VSD_ES = exps_VSD[indices]
	PCs_VSD_ES = PCs_VSD[indices]
	ax.hist(exps_VSD_ES, color = 'b', alpha = 0.13, bins = n_bins_VSD, edgecolor = 'k', label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (ES)')
	# inds to large PC values
	ind_largePC_VSD_ES = np.where(PCs_VSD_ES > PC_threshold)[0] 
	exps_VSD_largePC = exps_VSD_ES[ind_largePC_VSD_ES]
	# corresponding hist
	ax.hist(exps_VSD_largePC, color = 'b', alpha = 0.17, bins = n_bins_VSD_largePC, edgecolor = None, label = r'$\eta_{' + str(orderOfMom) + ',\perp,>}$ (ES, PC$>$0.85)')

	# # VSD hist (IS)
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
	ax.set_xlabel(r'Scaling exponent $\eta_{' + str(orderOfMom) + ', \perp, >}$', fontsize = 50)
	
	plt.show()
	
## 3D plot ##
def Plot_3DShape(x, y, z, c, xtitle, ytitle, ztitle, ctitle, location = "right"):

	ax = plt.axes(projection ='3d')
	im = ax.scatter(x, y, z, c = c, cmap='coolwarm')

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
	twin_ax.fill_between(princ_coord, nu - nu_unc, nu + nu_unc, color = 'm', alpha = 0.2)
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
	
	fig, ax1 = plt.subplots()

	ax1.set_title(title, fontsize = 60)
	
	ax1.plot(princ_coord, mode, 'k--', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{mode} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	ax1.plot(princ_coord, median, 'r--', alpha = 0.5, linewidth = 3.5, label = r"\( \mathrm{med} ( \{ l_{\mathrm{mut}} - l_{i} \} )\)")
	ax1.plot(princ_coord, mean, 'b--', alpha = 0.5, linewidth = 3.5, label = r"\( \langle \{ l_{\mathrm{mut}} - l_{i} \} \rangle \)")
	ax1.fill_between(princ_coord, mean - std, mean + std, alpha = 0.2, color = 'r')
	
	if (PLOT_ENTROPY_LINE == True):
		ax1.plot(princ_coord, entr_conf_max_model, 'g-', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (theoy) ")
		ax1.plot(princ_coord, entr_conf_max_emp, 'g--', alpha = 0.5, linewidth = 3.5, label = r"\( l_* - l_i\) (empir.)")
	
	# ax1.legend(loc = 'upper left', fontsize = 50)
	ax1.set_ylabel(r'\( l_{\mathrm{mut}} - l_{i} \) [\AA]', fontsize = 65)
	# ax1.set_xlabel(r'Pore point ($\perp$-coord.)', fontsize = 55)
	ax1.axhline(y = 0, linestyle = "-", color = 'b', alpha = 0.25, linewidth = 3)
	ax1.axhline(y = 0, linestyle = "-", color = 'r', alpha = 0.25, linewidth = 3)
	ax1.set_ylim([-30, 30])
	ax1.set_xticks([])
	# ax1.set_xticks([-14, -3, 8])
	# ax1.set_xticklabels(['AG', 'CC', 'SF'],  fontsize=70)
	# print(princ_coord[np.argmin(median)]-5, princ_coord[np.argmin(median)]+5)
	
	ax1.axvspan(princ_coord[np.argmin(median)]-5, princ_coord[np.argmin(median)]+5, alpha=0.1, color='m')
	# ax1.legend(loc='center left', bbox_to_anchor=(0, 0.1), fontsize = 32)
	
	ax2 = ax1.twinx()
	poreRad = poreRad / max(poreRad)
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

## Plot the statistical summary of a feature
def Plot_FeatureStatistic(princ_coord, data_class_0, data_class_1, poreRad, poreRad_std, label_0, label_1, statMethod, index_name):

	if (statMethod == "MEDIAN"):

		y0 = np.ma.masked_equal(GetColumn(data_class_0, [2]), ModelParameters.MASK_VAL)
		l0 = np.ma.masked_equal(GetColumn(data_class_0, [3]), ModelParameters.MASK_VAL)
		r0 = np.ma.masked_equal(GetColumn(data_class_0, [4]), ModelParameters.MASK_VAL)
	
		y1 = np.ma.masked_equal(GetColumn(data_class_1, [2]), ModelParameters.MASK_VAL)
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
	ax1.plot(princ_coord, y0, 'r', alpha = 0.5, linewidth = 2.5, label = label_0)
	ax1.fill_between(princ_coord, l0, r0, alpha = 0.2, color = 'r')
	# ax1.plot(princ_coord, mean_class_0, 'r--', label = label_0)

	# Plot class_1
	ax1.plot(princ_coord, y1, 'b-', alpha = 0.5, linewidth = 2.5, label = label_1)
	ax1.fill_between(princ_coord, l1, r1, alpha = 0.2, color = 'b')
	# ax1.plot(princ_coord, mean_class_1, 'b--', label = label_0)	

	ax1.legend(loc = 'upper left', fontsize = 50)
	ax1.set_ylabel(index_name, fontsize = 65)
	ax1.set_xlabel(r'Pore point ($\perp$-coord.)', fontsize = 55)
	ax1.axhline(y = 0, linestyle = "-", color = 'b', alpha = 0.25, linewidth = 3)
	ax1.axhline(y = 0, linestyle = "-", color = 'r', alpha = 0.25, linewidth = 3)

	ax2 = ax1.twinx()
	poreRad = poreRad / max(poreRad)
	ax2.plot(princ_coord, poreRad, 'k', label = r'$R / \mathrm{max}(R)$')
	ax2.fill_between(princ_coord, poreRad - 0.5 * poreRad_std / max(poreRad), poreRad +  0.5 * poreRad_std / max(poreRad), alpha = 0.15, color = 'k')
	ax2.set_xlim([min(princ_coord), max(princ_coord)])
	# ax2.text(-19, 0.06, 'AG', color='k',alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(-3, 0.06, 'CC', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	# ax2.text(8, 0.06, 'SF', color='k', alpha = 0.7, fontsize = 55, bbox=dict(facecolor='none', edgecolor='k', boxstyle='round'))
	ax2.set_ylim([0,1])
	
	ax2.legend(loc = 'upper right', fontsize = 50)
	ax2.set_ylabel(r'Pore radius (norm.)', fontsize = 65)

	plt.show()

##	Plot the medians of the medians of different features for different mutation subsets ##
def Plot_MediansOfFeatureMedias(princ_coord, medians_subclasses, medians_class1, poreRad, poreRad_std, 
								percentiles = [],  medians_unseen = [], medians_missclass = [], 
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
	if (len(percentiles) == 0):
		ax1.fill_between(princ_coord, np.min(medians_subclasses[0], axis = 0), np.max(medians_subclasses[0], axis = 0), color = "red", alpha = 0.2)
	else:
		ax1.fill_between(princ_coord, leftPerc_subclassA, rightPerc_subclassA, color = "red", alpha = 0.2)
	# subclass B (LoF)
	ax1.plot(princ_coord, np.median(medians_subclasses[1], axis = 0), color = "m", linewidth = 3, label = "LoF")
	if (len(percentiles) == 0):
		ax1.fill_between(princ_coord, np.min(medians_subclasses[1], axis = 0), np.max(medians_subclasses[1], axis = 0), color = "m", alpha = 0.2)
	else:
		ax1.fill_between(princ_coord, leftPerc_subclassB, rightPerc_subclassB, color = "m", alpha = 0.2)

	# class 1
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

	# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize = 35)	
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

	ax.legend(loc = 'upper left', fontsize = 50)

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
def Plot_MutationScores(class0_probs, resIDs, n_class0, n_class1, indsAppend_LoF, indsAppend_GoF, optThres, NEUTRAL_IND = True):

	data_in = []

	for i in range(len(resIDs)):
		data_in.append(GetColumn(class0_probs,[i]))
	

	n_class = n_class0 + n_class1
	inds_class0 = np.arange(0, n_class0, 1) 
	inds_class1 = np.arange(n_class0, n_class, 1) 

	fig, ax = plt.subplots()

	if (len(indsAppend_LoF) > 0):
		data_LoF = []
		for i in range(len(indsAppend_LoF)):
			data_LoF.append(data_in[indsAppend_LoF[i]])
		# Pathogenic
		ax.boxplot(data_LoF, positions = (inds_class0 + 1)[indsAppend_LoF], 
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
		ax.boxplot(data_PEPD, positions = (inds_class0 + 1)[indsAppend_GoF[0]], 
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
		ax.boxplot(data_SFN, positions = (inds_class0 + 1)[indsAppend_GoF[1]], 
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
		ax.boxplot(data_IEM, positions = (inds_class0 + 1)[indsAppend_GoF[2]], 
            patch_artist=True,
            boxprops=dict(facecolor='red', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red'),
            flierprops=dict(marker='o', color='black', markersize=8))

	# Neutral
	if (NEUTRAL_IND == True):
		ax.boxplot(data_in[n_class0:n_class], positions = inds_class1 + 1, 
            patch_artist=True,
            boxprops=dict(facecolor='blue', color='black', alpha = 0.2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='blue'),
            flierprops=dict(marker='o', color='black', markersize=8))
	
	ax.axhspan(ymin = optThres[1], ymax = optThres[2], color = "k", alpha = 0.05)
	ax.axhline(y = np.round(optThres[0],2), color = "k", alpha = 0.3)
	ax.set_xticks(np.arange(1, n_class + 1))  # Correctly set positions only
	ax.set_xticklabels(resIDs, rotation=90, fontsize=15)  # Set labels separately
	

	# Customize specific x-tick labels
	IEM_ticks = resIDs[indsAppend_GoF[2]]  
	SFN_ticks = resIDs[indsAppend_GoF[1]]  
	PEPD_ticks = resIDs[indsAppend_GoF[0]]  
	LoF_ticks = resIDs[indsAppend_LoF] 
	Neutral_ticks = resIDs[n_class0:n_class]  

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
		if tick_value in ['Cys1719', 'Leu172', 'Arg99', 'Arg1279', 'Ile739', 'Ile720', 'Arg185', 'Trp1538']:
			tick.set_color("k")

	# print(optThres[0])
	ax.set_yticks(np.array([0.1,0.2,0.3,0.4, np.round(optThres[0],3), 0.5, 0.6,0.7,0.8,0.9,1.0]))
	ax.tick_params(axis='y', labelsize=25) 
	ax.plot([], [], color='red', label='IEM', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='green', label='SFN', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='orange', label='PEPD', marker='s', linestyle='none', linewidth = 2)
	ax.plot([], [], color='magenta', label='LoF', marker='s', linestyle='none', linewidth = 2)
	ax.set_ylabel('Classification Summary', fontsize = 50)
	ax.legend(loc="lower left", fontsize = 32)
	
	plt.show()



