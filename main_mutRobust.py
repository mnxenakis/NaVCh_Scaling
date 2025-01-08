"""

	This is your main function for Mutational Robustness Analysis.

	It essentially orchestrates and executes all necessary tasks locally. 
	It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.

"""

import sys
sys.path.insert(1, '/home/xxx/hydroscale')

import os
pdb_code = os.getcwd()[-4:]
print("\n \n .. Starting working with molecule:", pdb_code, "found in: \n", os.getcwd())

import Methods
import ModelParameters

"""
		Insert variants
"""
Methods.InsertVariants()

"""
		
        Assess mutational robustness:
       		
            Generally, mutations are expected to cluster somewhere around \( \partial B^{*} \) (where the entropy of the evolutionary tossing-coin game maximizes)

       	 	Consider two main classes: 
        		-	pathogenic events (class 0)
            	-	control (non-path.) events (class 1)
               
"""
pathogenic = "GoF/LoF" 
control = "Neutral"
pathogenic_unseen = "Pathogenic(certain/likely/likely)"
control_unseen = "Benign(certain/likely/likely)"
missclassified = 'missclassified'

"""
		Investigate the distribution of mutations inside the channel around \( \partial B_i \)
"""
Methods.MutationDistribution(["GoF/LoF", "All"])
 
"""
	Extract features:
        
        - Consider orders of moments j = 0,1,2,..,(2*N_HYDR - 1).
        - Two groups, j = 2k and j = 2k + 1, corresponding to rotational and translational dynamics.
        - Obtain two main features: 

            (a), cluster's sensitivity to perturbations (denoted as \phi_{j})
            (b), cluster-shell's sensitivity to perturbations (i.e., interfacial coupling strength (denoted as \mathcal{I}_{j}))
                
        	Note that \mathcal{I}_{j} is the first-order derivative of \phi_{j}. 
            Hence, it can be interpreted as the cluster shell's sensitivity to perturbations
            Accrodingly, higher order derivatives of \phi_{j} could also be used as features. 
        	However, accurate estimation of higher order derivaties and their interpretation becomes increasingly difficult.
                
"""
for j in range(2*ModelParameters.N_HYDR):
	Methods.FeaturesExtraction(		
									[pathogenic, control], 	
									[pathogenic_unseen, control_unseen], 
									[missclassified], j, subClasses = ["GoF", "LoF"]
								) 


"""
		Plot features summary (medians of the medians)
"""
even_contributions = [num for num in range(2*ModelParameters.N_HYDR) if num % 2 == 0]
odd_contributions = [num for num in range(2*ModelParameters.N_HYDR) if num % 2 != 0 and num != 1] # exclude first-order contributions since they do not satisfy the Decomp Ansatz everywhere
Methods.FeaturesSummary(even_contributions, odd_contributions, unseen = 'unseen', missclassified = 'missclassified') 
Methods.FeaturesSummary([0], [1], unseen = 'unseen', missclassified = 'missclassified') 


"""
	Perform machine learning experiments:
		- At each pore point (locally) classify based on two main features: \mathcal{I}_{j} and \phi_{j}, j = 2k and j = 2k + 1
		- Across pore points (globally) classify based on pore-point "learnings" (i.e., medians of class_0_probabilities):
				
"""
features_absPhi_evenOrder = ['0-order_absPhi', '2-order_absPhi', '4-order_absPhi', '6-order_absPhi', '8-order_absPhi', '10-order_absPhi']
features_absPhi_oddOrder = ['1-order_absPhi', '3-order_absPhi', '5-order_absPhi', '7-order_absPhi', '9-order_absPhi', '11-order_absPhi']
features_phi_evenOrder = ['0-order_phi', '2-order_phi', '4-order_phi', '6-order_phi', '8-order_phi', '10-order_phi']
features_phi_oddOrder = ['1-order_phi', '3-order_phi', '5-order_phi', '7-order_phi', '9-order_phi', '11-order_phi']

features_absDerPhi_evenOrder = ['0-order_absDerPhi', '2-order_absDerPhi', '4-order_absDerPhi', '6-order_absDerPhi', '8-order_absDerPhi', '10-order_absDerPhi']
features_absDerPhi_oddOrder = ['1-order_absDerPhi', '3-order_absDerPhi', '5-order_absDerPhi', '7-order_absDerPhi', '9-order_absDerPhi', '11-order_absDerPhi']
features_derPhi_evenOrder = ['0-order_derPhi', '2-order_derPhi', '4-order_derPhi', '6-order_derPhi', '8-order_derPhi', '10-order_derPhi']
features_derPhi_oddOrder = ['1-order_derPhi', '3-order_derPhi', '5-order_derPhi', '7-order_derPhi', '9-order_derPhi', '11-order_derPhi']

features_pertPot_evenOrder = ['0-order_pertPot', '2-order_pertPot', '4-order_pertPot', '6-order_pertPot', '8-order_pertPot', '10-order_pertPot']
features_pertPot_oddOrder = ['1-order_pertPot', '3-order_pertPot', '5-order_pertPot', '7-order_pertPot', '9-order_pertPot', '11-order_pertPot']


"""
		Local learning (pore-point-specific)
"""
##
##	Even contributions (j = 2k): rotational dynamical effects ##
##
learnings_absPhi_evenOrder = ['learnings_absPhi_evenOrder']
Methods.PorePointLearning(features_absPhi_evenOrder, pathogenic, control, learnings_absPhi_evenOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_even =  ["$|\phi_{0}|$", "$|\phi_{2}|$", "$|\phi_{4}|$", "$|\phi_{6}|$", "$|\phi_{8}|$", "$|\phi_{10}|$"]
Methods.Plot_Learnings(learnings_absPhi_evenOrder, featLabels_even)

learnings_phi_evenOrder = ['learnings_phi_evenOrder']
Methods.PorePointLearning(features_phi_evenOrder, pathogenic, control, learnings_phi_evenOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_even =  ["$\phi_{0}$", "$\phi_{2}$", "$\phi_{4}$", "$\phi_{6}$", "$\phi_{8}$", "$\phi_{10}$"]
Methods.Plot_Learnings(learnings_phi_evenOrder, featLabels_even)

learnings_derPhi_evenOrder = ['learnings_derPhi_evenOrder']
Methods.PorePointLearning(features_derPhi_evenOrder, pathogenic, control, learnings_derPhi_evenOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_even =  ["$\mathcal{I}_{0}$", "$\mathcal{I}_{2}$", "$\mathcal{I}_{4}$", "$\mathcal{I}_{6}$", "$\mathcal{I}_{8}$", "$\mathcal{I}_{10}$"]
Methods.Plot_Learnings(learnings_derPhi_evenOrder, featLabels_even)


learnings_absDerPhi_evenOrder = ['learnings_absDerPhi_evenOrder']
Methods.PorePointLearning(features_absDerPhi_evenOrder, pathogenic, control, learnings_absDerPhi_evenOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_even =  ["$|\mathcal{I}_{0}|$", "$|\mathcal{I}_{2}|$", "$|\mathcal{I}_{4}|$", "$|\mathcal{I}_{6}|$", "$|\mathcal{I}_{8}|$", "$|\mathcal{I}_{10}|$"]
Methods.Plot_Learnings(learnings_absDerPhi_evenOrder, featLabels_even)

# ##
# ##	Odd contributions (j = 2k + 1): translational dynamical effects ##
# ##
learnings_absPhi_oddOrder = ['learnings_absPhi_oddOrder']
Methods.PorePointLearning(features_absPhi_oddOrder, pathogenic, control, learnings_absPhi_oddOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_odd =  ["$|\phi_{1,\perp}|$", "$|\phi_{3,\perp}|$", "$|\phi_{5,\perp}|$", "$|\phi_{7,\perp}|$", "$|\phi_{9,\perp}|$", "$|\phi_{11,\perp}|$"]
Methods.Plot_Learnings(learnings_absPhi_oddOrder, featLabels_odd)

learnings_phi_oddOrder = ['learnings_phi_oddOrder']
Methods.PorePointLearning(features_phi_oddOrder, pathogenic, control, learnings_phi_oddOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_odd =  ["$\phi_{1,\perp}$", "$\phi_{3,\perp}$", "$\phi_{5,\perp}$", "$\phi_{7,\perp}$", "$\phi_{9,\perp}$", "$\phi_{11,\perp}$"]
Methods.Plot_Learnings(learnings_phi_oddOrder, featLabels_odd)

learnings_derPhi_oddOrder = ['learnings_derPhi_oddOrder']
Methods.PorePointLearning(features_derPhi_oddOrder, pathogenic, control, learnings_derPhi_oddOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_odd =  ["$\mathcal{I}_{1,\perp}$", "$\mathcal{I}_{3,\perp}$", "$\mathcal{I}_{5,\perp}$", "$\mathcal{I}_{7,\perp}$", "$\mathcal{I}_{9,\perp}$", "$\mathcal{I}_{11,\perp}$"]
Methods.Plot_Learnings(learnings_derPhi_oddOrder, featLabels_odd)

learnings_absDerPhi_oddOrder = ['learnings_absDerPhi_oddOrder']
Methods.PorePointLearning(features_absDerPhi_oddOrder, pathogenic, control, learnings_absDerPhi_oddOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
featLabels_odd =  ["$|\mathcal{I}_{1,\perp}|$", "$|\mathcal{I}_{3,\perp}|$", "$|\mathcal{I}_{5,\perp}|$", "$|\mathcal{I}_{7,\perp}|$", "$|\mathcal{I}_{9,\perp}|$", "$|\mathcal{I}_{11,\perp}|$"]
Methods.Plot_Learnings(learnings_absDerPhi_oddOrder, featLabels_odd)

"""
		Ensemble (non-local) learning (acros pore points)
"""
## Take directionality into account ..
features_ensemble = [	
						# even
						'learnings_phi_evenOrder', 
						'learnings_derPhi_evenOrder', 	
						# odd
						'learnings_phi_oddOrder', 
						'learnings_derPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
## even
features_ensemble = [	
						# even
						# 'learnings_phi_evenOrder', 
						# 'learnings_derPhi_evenOrder', 	
						# odd
						'learnings_phi_oddOrder', 
						'learnings_derPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
## odd
features_ensemble = [	
						# even
						'learnings_phi_evenOrder', 
						'learnings_derPhi_evenOrder', 	
						# odd
						# 'learnings_phi_oddOrder', 
						# 'learnings_derPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)

## Do not take directionality into account (unsigned features) ..
features_ensemble = [	
						# even
						'learnings_absPhi_evenOrder', 
						'learnings_absDerPhi_evenOrder', 	
						# odd
						'learnings_absPhi_oddOrder', 
						'learnings_absDerPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
## even
features_ensemble = [	
						# even
						# 'learnings_absPhi_evenOrder', 
						# 'learnings_absDerPhi_evenOrder', 	
						# odd
						'learnings_absPhi_oddOrder', 
						'learnings_absDerPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
## odd
features_ensemble = [	
						# even
						'learnings_absPhi_evenOrder', 
						'learnings_absDerPhi_evenOrder', 	
						# odd
						# 'learnings_absPhi_oddOrder', 
						# 'learnings_absDerPhi_oddOrder',
					]
Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)



	
