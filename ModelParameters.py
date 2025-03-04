'''

		Parameter values

'''
## RG parameters
N_SCALES = 800 					# Nr of scales (this is also the sampling resolution along the radial direction: \alpha = 1,2,..,N_SCALES)
N_HYDR = 6 						# ODD and EVEN nr of exp free energy obs (hydr mom), nr of moments is: 2*Key_Parameters.N_HYDR

## Hole parameters
N_RUNS_HOLE = 50 				# HOLE runs
ROUND_DIGIT_PP = 2 				# Pore points coordinates will be rounded to ROUND_DIGIT_PP decimal digit
HOLE_RES = 1e-1 				# SAMPLE parameter of the hole.inp file

## Additive noise parameters
STD_NOISE = 1e-3 				# *Very* weak noise profile ("smooths" hydropathic profiles mimicking the effect of room temperature fluctuations)
ZERO = 1e-04 					# Numerial zero 

## Numerical comparisons of pore point coordinates
ACC	= 1e-2 

## Numerical differentiation and smoothing  
POLYORDER = 2 					# Smoothing parameter (degree of the polynomial used to fit the data within each window)

## Window factor (for computing derivatives)
WINDOW_FAC = 1 					# Adjsuts the window size (the smaller, the more detailed the scaling picture)

## Machine Learning Parameters

# K-fold cross-validation 
CLASSIFIER_SEED	= 42 			# Classifier seed	
K = 3 							# Maximum nr of folds (determines size of validation test: 1/K) (note that we consider K - 1 different folds: 2, 3, .. , K) 
NUM_OF_TRAININGS = 20 			# Nr of k-fold trainings # run with: 20 (local learning) and 50 (ensemble learning)
NUM_OF_IMBALANCES_REMOVAL = 10 	# Nr of balancing attempts for the "unseen" data set

# Logistic Regression classifier parameters
MAX_ITER = 1000					
C = 1 							

# SVM classifier parameters
KERNEL_NONLINEAR = 'rbf'
KERNEL_LINEAR = 'linear'
NUM_OF_PERM_REPEATS	= 10 		# Nr of permuations (feature importance evaluation during SVM) 

## Percentiles plotting parameter
NUM_OF_PERCENTILES = 20
P_LEFT = 25

## Minimum size of domain (measured in terms of radial shells)
MIN_DOMAIN_SIZE	= 5

## Masking 
MASK_VAL = 10000