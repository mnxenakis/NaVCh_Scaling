'''
		Parameter list
'''

## RG parameters
N_SCALES = 800 # nr of scales (this is also the sampling resolution along the radial direction: \alpha = 1,2,..,N_SCALES)
N_HYDR = 6 # ODD and EVEN nr of exp free energy obs (hydr mom), nr of moments is: 2*Key_Parameters.N_HYDR

## Hole parameters
N_RUNS_HOLE = 50 # HOLE runs
ROUND_DIGIT_PP = 2 # pore points coordinates will be rounded to ROUND_DIGIT_PP decimal digit
HOLE_RES = 1e-1 # SAMPLE parameter of the hole.inp file

## Additive noise parameters
STD_NOISE = 1e-3 # *very* weak noise profile ("smooths" hydropathic profiles mimicking the effect of room temperature fluctuations)
ZERO = 1e-04 # numerial correction for the hydropathic energy of empty initial shells

## Geom model parameters
NU_MIN = 1e-3 # smallest value of \nu (\nu \to 0)

## Numerical comparisons of pore point coordinates
ACC	= 1e-2 

## Numerical differentiation and smoothing  
POLYORDER = 2 # smoothing parameter (degree of the polynomial used to fit the data within each window)
# window factor
WINDOW_FAC = 1.0 # adjsuts the window size (the smaller, the more detailed the scaling picture)

## Machine Learning Parameters
# K-fold cross-validation 
CLASSIFIER_SEED	= 42 # classifier seed	
K = 2 # maximum nr of folds (determines size of validation test: 1/K) (note that we consider K - 1 different folds: 2, 3, .. , K) run with: 3
NUM_OF_TRAININGS = 20 # number of k-fold trainings # run with: 20 (local learning) and 50 (ensemble learning)
NUM_OF_IMBALANCES_REMOVAL = 10 # number of balancing attempts for the "unseen" data set
# Log Reg parameters
MAX_ITER = 1000	# Max Iter of Log Reg
C = 1 # C param of Log Reg

# SVC parameters
KERNEL_NONLINEAR = 'rbf'
KERNEL_LINEAR = 'linear'
NUM_OF_PERM_REPEATS	= 10 # nr of permuations (feature importance evaluation for SVC) # run with 10

## Percentiles plotting parameter
NUM_OF_PERCENTILES = 20
P_LEFT = 25

## Minimum size of domain (measured in terms of radial shells)
MIN_DOMAIN_SIZE	= 5

## Masking 
MASK_VAL = 10000