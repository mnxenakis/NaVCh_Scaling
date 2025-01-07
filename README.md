# Voltage-gated Sodium Channel Protein Molecule Scaling and Mutational Robustness Analysis

# Preparation

Welcome to the voltage-gated sodium channels (NaVChs) scaling analysis project!

\textit{Step 1} To begin, create a directory in your home directory by executing the following command in a terminal:

	mkdir ~/hydroscale

This creates the directory /home/xxx/hydroscale, where xxx is your username.

Step 2. Place the following files in the newly created hydroscale directory:

    Methods.py
    Tools.py
    reduce.rad
    ModelParameters.py
    KapchaRosskyScale.py

You can use the mv or cp command to move or copy these files into /home/xxx/hydroscale:

	mv Methods.py ModelParameters.py KapchaRosskyScale.py Tools.py reduce.rad ~/hydroscale/

Step 3. Download a PDB file (Protein Data Bank file) of interest. 

Create a directory named after the PDB code: 

    	mkdir -p ~/NaVCh_ScalingAnalysis/NaVAb/3rvy
    
This ensures that PDB-related files are stored in the NaVCh_ScalingAnalysis directory instead of hydroscale. To exeplify, we use here the 3rvy PDB code corresponding to a prototype bacterial NaVCh, namely, the NaVAb protein molecule captured at a pre-open state (https://www.wwpdb.org/pdb?id=pdb_00003rvy)

Step 4. To proceed with the full analysis cycle, you will need to perform the following procedures: 
- "Clean" the PDB file (remove waters, toxins, HETATM, and other non-standard atoms)
- Protonate the structure (add hydrogens)
- Align the principal pore axis of the structure with the z-axis

Unfortunately, "cleaning" of the PDB file cannot be fully automated. We chose to "clean" PDB files in the yasara software (https://www.yasara.org/).

Protonation of the "clean" structure is performed by the reduce software (https://github.com/rlabduke/reduce). 

For the 3rvy molecule, we use the reduce command: 

    reduce -noadj 3rvy_clean.pdb > 3rvy_clean_H.pdb

For any other molecule, we use the reduce command:

    reduce -BUILD -NOHETh 3rvy_clean.pdb > 3rvy_clean_H.pdb 

To align the principal pore axis of the structure with the z-axis we use the VMD software (https://www.ks.uiuc.edu/Research/vmd/).
Ensure the Orient package is available in your VMD installation. Utilize the following .tcl script:
    
    package require Orient
    namespace import Orient::orient

    # Load the cleaned and protonated .pdb file
    mol load pdb 3rvy_clean_H.pdb

    # Select all atoms
    set sel [atomselect top "all"]

    # Calculate the principal axes
    set I [draw principalaxes $sel]

    # Align the principal axis to the z-axis
    set A [orient $sel [lindex $I 2] {0 0 1}]
    
    # Save the aligned structure
    set sel [atomselect top "all"]
    $sel writepdb 3rvy_clean_H_ori.pdb 
    
    # Save the geom center of the structure in the geom_center.dat file
    # We need the mol center of the clean, protonated, and oriented structure to initiate the HOLE routine
    set file [open "geom_center.dat" w]
    puts $file [ geom_center $sel ] 
    close $file
    quit

Save the script as align.tcl (or a name of your choice).
Run it in VMD with the following command:

    vmd -dispdev text 3rvy_clean_H.pdb < align.tcl

Step 5. We renumber the residue entries in the PDB file, ensuring they follow a sequential and consistent order. This is done by utilizing the pdb tool (http://www.bonvinlab.org/pdb-tools/) command:

    pdb_reres -1 3rvy_clean_H_ori.pdb > 3rvy_clean_H_ori_renum.pdb 

Step 6. Navigate through the NaVCh pore environment. We use the HOLE software (https://www.holeprogram.org/).
We call the HOLE rountine $N_{\text{HOLE}} = 50$ times. We use the following script to change the HOLE seed and extract pore radius results:

    #!/bin/bash

    # Number of runs (e.g., N_HOLE = 50)
    n=50             
    # Initial seed value
    seed=1000        
    # Seed increment between runs
    incr_seed=1000   

    # Loop for the number of runs
    for (( i=0; i<$n; i++ )); do

        # Print message about the current run and seed
        echo "Calling HOLE for the $i-th time with seed $seed, which will be incremented to $((seed+incr_seed))"
    
        # Run HOLE program and save output to hole_out.txt
        echo "Run HOLE"
        hole < hole.inp > hole_out.txt 

        # Insert space between numbers in the output for easier parsing
        printf "%s\n" "${string}" | sed 's/-/ -/g' hole_out.txt > hole_out_spaced.txt

        # Extract lines containing "mid-" or "sampled" and save to poreRadius.tsv
        egrep "mid-|sampled" hole_out_spaced.txt > poreRadius.tsv 

        # Format the poreRadius.tsv file and save it to ppr_$i.dat
        cat poreRadius.tsv | awk '{print $1,$2}' > ppr_$i.dat 

        # Extract the highest radius point found and save to porePoints.dat
        sed -n '/ highest radius point found:/{n;p;}' hole_out_spaced.txt > porePoints.dat

        # Clean up the porePoints.dat file by removing the first two columns and save it as pp_$i.dat
        awk '{ $1=""; $2=""; print $0}' porePoints.dat > pp_$i.dat 

        # Remove the temporary porePoints.dat file
        rm porePoints.dat

        # Update the seed in hole.inp by replacing the old seed with the incremented seed
        sed -i 's/RASEED '$seed'/RASEED '$((seed+incr_seed))'/g' hole.inp

        # Increment the seed for the next run
        ((seed=seed+incr_seed))

    done

    # After all runs, reset the seed in hole.inp back to the initial value (1000)
    sed -i 's/RASEED '$seed'/RASEED '1000'/g' hole.inp

    # Return to the original directory
    cd -

Step 7. Finally, we organize the generated data. In the current subdirectory, we execute the following commands:

    # Create a directory to save results
    mkdir 3rvy_prad 

    # Move all ppr_$i.dat files (generated by HOLE) to the newly created directory
    mv *pp* 3rvy_prad 

    # Clean up temporary files while keeping the hole.inp file as a precaution
    rm hole_out.txt
    rm hole_out_spaced.txt
    rm hole_out.sph
    rm poreRadius.tsv
    rm hole_out.sph.old

This step ensures the results are neatly stored in a dedicated directory and removes unnecessary temporary files, leaving the workspace organized and ready for further analysis.

# Single Voltage-gated Sodium Channel Protein Molecule Scaling Analysis

Now we are prepared to execute the `main_scaling.py` function locally, i.e., within the `3rvy` directory:

	"""
	This is your main function for Scaling Analysis.
 
 	It essentially orchestrates and executes all necessary tasks locally. 
	It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.
	"""

	import os

	# Get the PDB code from the current directory name
	pdb_code = os.getcwd()[-4:]
	print("\n\n.. Starting working with molecule:", pdb_code, "found in: \n", os.getcwd())
	
	import sys

	# Add the hydroscale directory to the system path for module imports
	sys.path.insert(1, '/home/xxx/hydroscale')

	import time
	import Methods

	# Record the start time for performance measurement
	start_time = time.time()

	# Call the sequence of methods required for the analysis
	Methods.HOLEOutputAnalysis()       # Analyze the output of HOLE
	Methods.PDBStructurePreperation()  # Prepare the PDB structure
	Methods.CollectObservables()       # Collect observables for the analysis
	Methods.InformationProfile()       # Generate the information profile

	# Display the elapsed time for the full process
	print("--- %s seconds ---" % (time.time() - start_time))

	# Exit the program
	exit()


Once the program has exit, the following files have appeared in the `3rvy` directory:

	3rvy_atomProbs.txt
	3rvy_dists.txt
	3rvy_entropies.txt
	3rvy_exponentsDomains.txt
	3rvy_hydrMoments.txt
	3rvy_mol.txt
	3rvy_nrOfAtoms.txt
	3rvy_resProbs.txt
	3rvy_scales.txt
	3rvy_statMod.txt
	3rvy_SummaryInfo.txt
	3rvy_topologies.txt
	3rvy_Preparation_Report.txt

These files contain all relevant information and may be used for further analysis.

# Single Voltage-gated Sodium Channel Protein Molecule Mutational Robustness Analysis

We focus on human NaVCh molecules because mutations in these channels are linked to various diseases, known as channelopathies. Specifically, we are interested in mutations related to pain disorders that affect the human NaV1.7 channel. To exeplify, we use here the 7w9k PDB code corresponding to a human NaV1.7 protein molecule captured at a potentially inactivated state (https://www.rcsb.org/structure/7W9K).												

Our fundamental assumption is that the scaling properties of the atomic environment along the principal pore axis encode information about the robustness of a structural location (i.e., residue geometric center) to mutation-induced perturbations. Similar to how the behavior of a self-organized critical system is determined by its critical scaling exponents. 

Now we are prepared to execute the `main_mutRobust.py` function locally, i.e., within the `7w9k` directory:

	"""
	
		This is your main function for Mutational Robustness Analysis.
	
		It essentially orchestrates and executes all necessary tasks locally. 
		It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.
	
	"""
	
	import os
	
	pdb_code = os.getcwd()[-4:]
	
	print("\n \n .. Starting working with molecule:", pdb_code, "found in: \n", os.getcwd())
	
	import sys
	sys.path.insert(1, '/home/xxx/hydroscale')
	
	import numpy as np
	
	import time
	import Methods
	import ModelParameters
	
	
	start_time = time
	
	"""
			Insert variants
	"""
	Methods.InsertVariants()
	
	"""
			
	        Assess mutational robustness:
	       		
	            Generally, mutations are expected to cluster around \( \partial B_i \)
	        	Specifically, pathogenic events exploit local "weaknesses" to induge chagnes easily
	
	       	 	Always consider two main classes: 
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
	# Methods.MutationDistribution(["GoF/LoF", "All"])
	 
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
	
	learnings_pertPot_evenOrder = ['learnings_pertPot_evenOrder']
	# Methods.PorePointLearning(features_pertPot_evenOrder, pathogenic, control, learnings_pertPot_evenOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
	featLabels_even =  ["$\omega_{0}$", "$\omega_{2}$", "$\omega_{4}$", "$\omega_{6}$", "$\omega_{8}$", "$\omega_{10}$"]
	Methods.Plot_Learnings(learnings_pertPot_evenOrder, featLabels_even)
	
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
	
	learnings_pertPot_oddOrder = ['learnings_pertPot_oddOrder']
	Methods.PorePointLearning(features_pertPot_oddOrder, pathogenic, control, learnings_pertPot_oddOrder, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)
	featLabels_odd =  ["$\omega_{1,\perp}$", "$\omega_{3,\perp}$", "$\omega_{5,\perp}$", "$\omega_{7,\perp}$", "$\omega_{9,\perp}$", "$\omega_{11,\perp}$"]
	Methods.Plot_Learnings(learnings_pertPot_oddOrder, featLabels_odd)
	
	
	"""
			Ensemble (non-local) learning (acros pore points)
	"""
	features_ensemble = [	
							# even
							'learnings_absPhi_evenOrder', 
							'learnings_absDerPhi_evenOrder', 	
							# odd
							'learnings_absPhi_oddOrder', 
							'learnings_absDerPhi_oddOrder',
						]
	Methods.EnsembleLearning(features_ensemble, pathogenic, control, method = 'SVC', kernel = ModelParameters.KERNEL_NONLINEAR)


This call will generate a lot of data files!



