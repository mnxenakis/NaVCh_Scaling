"""
This is your main function for Mutational Robustness Analysis.

It essentially orchestrates and executes all necessary tasks locally. 
It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.
"""

import os
import sys

# Get the PDB code from the current directory name
pdb_code = os.getcwd()[-4:]
print(f"\n\n.. Starting working with molecule: {pdb_code}")
print(f"Working directory: {os.getcwd()}")

# Add the hydroscale directory to the system path for module imports
# Note: Update the path below to match your actual hydroscale directory
hydroscale_path = '/home/xxx/hydroscale'  # TODO: Update this path
sys.path.insert(0, hydroscale_path)

try:
    import Methods
    import ModelParameters
    
    print("Starting mutational robustness analysis...")
    
    # Insert variants
    print("1. Inserting variants...")
    Methods.InsertVariants()
    
    # Define mutation classes for analysis
    pathogenic = "GoF/LoF" 
    control = "Neutral"
    pathogenic_unseen = "Pathogenic(certain/likely/likely)"
    control_unseen = "Benign(certain/likely/likely)"
    missclassified = 'missclassified'
    
    print("2. Analyzing mutation distribution...")
    Methods.MutationDistribution(["GoF/LoF", "All"])
    
    print("3. Extracting features for all hydrophobic moment orders...")
    # Extract features for different orders of moments (j = 0,1,2,...,(2*N_HYDR - 1))
    # Two groups: j = 2k (rotational dynamics) and j = 2k + 1 (translational dynamics)
    for j in range(2 * ModelParameters.N_HYDR):
        print(f"   Processing order {j}...")
        Methods.FeaturesExtraction(
            [pathogenic, control], 	
            [pathogenic_unseen, control_unseen], 
            [missclassified], 
            j, 
            subClasses=["GoF", "LoF"]
        ) 

    # Plot features summary (medians of the medians)
    print("4. Plotting features summary...")
    even_contributions = [num for num in range(2*ModelParameters.N_HYDR) if num % 2 == 0]
    odd_contributions = [num for num in range(2*ModelParameters.N_HYDR) if num % 2 != 0 and num != 1] 
    # exclude first-order contributions since they do not satisfy the Decomp Ansatz everywhere
    
    Methods.FeaturesSummary(even_contributions, odd_contributions, unseen='unseen', missclassified='missclassified') 
    Methods.FeaturesSummary([0], [1], unseen='unseen', missclassified='missclassified') 

    # Define feature lists for machine learning experiments
    print("5. Setting up machine learning features...")
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

    # Local learning (pore-point-specific)
    print("6. Performing local learning experiments...")
    
    # Even contributions (j = 2k): rotational dynamical effects
    print("   6a. Processing even contributions (rotational dynamics)...")
    learnings_absPhi_evenOrder = ['learnings_absPhi_evenOrder']
    Methods.PorePointLearning(features_absPhi_evenOrder, pathogenic, control, learnings_absPhi_evenOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_even = ["$|\phi_{0}|$", "$|\phi_{2}|$", "$|\phi_{4}|$", "$|\phi_{6}|$", "$|\phi_{8}|$", "$|\phi_{10}|$"]
    Methods.Plot_Learnings(learnings_absPhi_evenOrder, featLabels_even)

    learnings_phi_evenOrder = ['learnings_phi_evenOrder']
    Methods.PorePointLearning(features_phi_evenOrder, pathogenic, control, learnings_phi_evenOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_even = ["$\phi_{0}$", "$\phi_{2}$", "$\phi_{4}$", "$\phi_{6}$", "$\phi_{8}$", "$\phi_{10}$"]
    Methods.Plot_Learnings(learnings_phi_evenOrder, featLabels_even)

    learnings_derPhi_evenOrder = ['learnings_derPhi_evenOrder']
    Methods.PorePointLearning(features_derPhi_evenOrder, pathogenic, control, learnings_derPhi_evenOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_even = ["$\mathcal{I}_{0}$", "$\mathcal{I}_{2}$", "$\mathcal{I}_{4}$", "$\mathcal{I}_{6}$", "$\mathcal{I}_{8}$", "$\mathcal{I}_{10}$"]
    Methods.Plot_Learnings(learnings_derPhi_evenOrder, featLabels_even)

    learnings_absDerPhi_evenOrder = ['learnings_absDerPhi_evenOrder']
    Methods.PorePointLearning(features_absDerPhi_evenOrder, pathogenic, control, learnings_absDerPhi_evenOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_even = ["$|\mathcal{I}_{0}|$", "$|\mathcal{I}_{2}|$", "$|\mathcal{I}_{4}|$", "$|\mathcal{I}_{6}|$", "$|\mathcal{I}_{8}|$", "$|\mathcal{I}_{10}|$"]
    Methods.Plot_Learnings(learnings_absDerPhi_evenOrder, featLabels_even)

    # Odd contributions (j = 2k + 1): translational dynamical effects
    print("   6b. Processing odd contributions (translational dynamics)...")
    learnings_absPhi_oddOrder = ['learnings_absPhi_oddOrder']
    Methods.PorePointLearning(features_absPhi_oddOrder, pathogenic, control, learnings_absPhi_oddOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_odd = ["$|\phi_{1,\perp}|$", "$|\phi_{3,\perp}|$", "$|\phi_{5,\perp}|$", "$|\phi_{7,\perp}|$", "$|\phi_{9,\perp}|$", "$|\phi_{11,\perp}|$"]
    Methods.Plot_Learnings(learnings_absPhi_oddOrder, featLabels_odd)

    learnings_phi_oddOrder = ['learnings_phi_oddOrder']
    Methods.PorePointLearning(features_phi_oddOrder, pathogenic, control, learnings_phi_oddOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_odd = ["$\phi_{1,\perp}$", "$\phi_{3,\perp}$", "$\phi_{5,\perp}$", "$\phi_{7,\perp}$", "$\phi_{9,\perp}$", "$\phi_{11,\perp}$"]
    Methods.Plot_Learnings(learnings_phi_oddOrder, featLabels_odd)

    learnings_derPhi_oddOrder = ['learnings_derPhi_oddOrder']
    Methods.PorePointLearning(features_derPhi_oddOrder, pathogenic, control, learnings_derPhi_oddOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_odd = ["$\mathcal{I}_{1,\perp}$", "$\mathcal{I}_{3,\perp}$", "$\mathcal{I}_{5,\perp}$", "$\mathcal{I}_{7,\perp}$", "$\mathcal{I}_{9,\perp}$", "$\mathcal{I}_{11,\perp}$"]
    Methods.Plot_Learnings(learnings_derPhi_oddOrder, featLabels_odd)

    learnings_absDerPhi_oddOrder = ['learnings_absDerPhi_oddOrder']
    Methods.PorePointLearning(features_absDerPhi_oddOrder, pathogenic, control, learnings_absDerPhi_oddOrder, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)
    featLabels_odd = ["$|\mathcal{I}_{1,\perp}|$", "$|\mathcal{I}_{3,\perp}|$", "$|\mathcal{I}_{5,\perp}|$", "$|\mathcal{I}_{7,\perp}|$", "$|\mathcal{I}_{9,\perp}|$", "$|\mathcal{I}_{11,\perp}|$"]
    Methods.Plot_Learnings(learnings_absDerPhi_oddOrder, featLabels_odd)

    # Ensemble (non-local) learning (across pore points)
    print("7. Performing ensemble learning experiments...")
    
    # Taking directionality into account
    print("   7a. With directional features...")
    features_ensemble_combinations = [
        # All features
        ['learnings_phi_evenOrder', 'learnings_derPhi_evenOrder', 'learnings_phi_oddOrder', 'learnings_derPhi_oddOrder'],
        # Only odd contributions  
        ['learnings_phi_oddOrder', 'learnings_derPhi_oddOrder'],
        # Only even contributions
        ['learnings_phi_evenOrder', 'learnings_derPhi_evenOrder']
    ]
    
    for i, features_ensemble in enumerate(features_ensemble_combinations):
        print(f"      Combination {i+1}: {features_ensemble}")
        Methods.EnsembleLearning(features_ensemble, pathogenic, control, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)

    # Without directionality (unsigned features)
    print("   7b. With unsigned features...")
    unsigned_features_combinations = [
        # All unsigned features
        ['learnings_absPhi_evenOrder', 'learnings_absDerPhi_evenOrder', 'learnings_absPhi_oddOrder', 'learnings_absDerPhi_oddOrder'],
        # Only odd unsigned contributions
        ['learnings_absPhi_oddOrder', 'learnings_absDerPhi_oddOrder'],
        # Only even unsigned contributions  
        ['learnings_absPhi_evenOrder', 'learnings_absDerPhi_evenOrder']
    ]
    
    for i, features_ensemble in enumerate(unsigned_features_combinations):
        print(f"      Unsigned combination {i+1}: {features_ensemble}")
        Methods.EnsembleLearning(features_ensemble, pathogenic, control, method='SVC', kernel=ModelParameters.KERNEL_NONLINEAR)

    print("\n✓ Mutational robustness analysis completed successfully!")
    
except ImportError as e:
    print(f"Error: Could not import required modules. Please check the hydroscale path: {hydroscale_path}")
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during mutational robustness analysis: {e}")
    sys.exit(1)



	
