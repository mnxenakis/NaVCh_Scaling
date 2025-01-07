# Voltage-gated Sodium Channel Protein Molecule Scaling and Mutational Robustness Analysis

# Preparation

Welcome to the voltage-gated sodium channels (NaVChs) scaling analysis project!

Step 1. To begin, create a directory in your home directory by executing the following command in a terminal:

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



