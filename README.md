# NaVCh_Scaling

Welcome to the voltage-gated sodium channels (NaVChs) scaling analysis project!

Step 1. To begin, create a directory in your home directory by executing the following command in a terminal:

mkdir ~/hydroscale

This creates the directory /home/xxx/hydroscale, where xxx is your username.

Step 2. Place the following files in the newly created hydroscale directory:

    Methods.py
    ModelParameters.py
    Tools.py
    reduce.rad

You can use the mv or cp command to move or copy these files into /home/xxx/hydroscale:

mv Methods.py ModelParameters.py Tools.py reduce.rad ~/hydroscale/

Replace mv with cp if you want to copy the files instead of moving them.

Step 3. Download a .pdb file (Protein Data Bank file) of interest. For example, you can download it from the RCSB PDB website. The pdb file encodes atomic coordinates of a NaVCh protein molecule.

Create a directory named after the PDB code inside a separate directory called work, not in the hydroscale directory:

mkdir -p ~/NaVCh_ScalingAnalysis/<PDB_code>

Replace <PDB_code> with the actual code of the .pdb file. For instance:

    mkdir -p ~/NaVCh_ScalingAnalysis/3rvy
    
This ensures that PDB-related files are stored in the NaVCh_ScalingAnalysis directory instead of hydroscale. Note that 3rvy is the pdb code corresponding to the prototype bacterial NaVCh, namely, the NaVAb protein molecule, captured at a pre-open state (https://www.wwpdb.org/pdb?id=pdb_00003rvy)

Step 4. To proceed with the full analysis cycle, you will need to perform the following procedures: 
- "Clean" the .pdb file (remove waters, toxins, HETATM, etc ..)
- Protonate the structure (add hydrogens)
- Align the principal pore axis of the structure with the z-axis

Unfortunately, "cleaning" of the .pdb file cannot be fully automated. We chose to "clean" .pdb files in the yasara software (https://www.yasara.org/).

Protonation of the "clean" structure is performed by the reduce software (https://github.com/rlabduke/reduce). 

For the 3rvy molecule, we use the reduce command: 

    reduce -noadj 3rvy_clean.pdb > 3rvy_clean_H.pdb

For any other molecule, we use the reduce command:

    reduce -BUILD -NOHETh <PDB_code>_clean.pdb > <PDB_code>_clean_H.pdb 

To align the principal pore axis of the structure with the z-axis we use the VMD software (https://www.ks.uiuc.edu/Research/vmd/).
Ensure the Orient package is available in your VMD installation. Utilize the following .tcl script:
    
    package require Orient
    namespace import Orient::orient

    # Load the cleaned and protonated .pdb file
    mol load pdb <PDB_code>_clean_H.pdb

    # Select all atoms
    set sel [atomselect top "all"]

    # Calculate the principal axes
    set I [draw principalaxes $sel]

    # Align the principal axis to the z-axis
    set A [orient $sel [lindex $I 2] {0 0 1}]
    
    # Save the aligned structure
    set sel [atomselect top "all"]
    $sel writepdb <PDB_code>_clean_H_ori.pdb 
    quit

Save the script as align.tcl (or a name of your choice).
Run it in VMD with the following command:

    vmd -dispdev text <PDB_code>_clean_H.pdb < align.tcl

Step 5. Navigate through the NaVCh pore environment. We use the HOLE software (https://www.holeprogram.org/).
We call the HOLE rountine $N_{HOLE} = 50$ times. We use the following script to change the HOLE seed and extract pore radius results:

    n=50 			# nr of runs (N_HOLE = 50)
    seed=1000 			# initial seed
    incr_seed=1000 		# seed increment

    for (( i=0 ; i<$n ; i++ )); 
	
    do

    echo calling hole .. for the $i-th time with seed $seed which will be incremented to $((seed+incr_seed))
    echo "run hole"
    hole < hole.inp > hole_out.txt # run hole

    printf "%s\n" "${string}" | sed 's/-/ -/g' hole_out.txt > hole_out_spaced.txt

    egrep "mid-|sampled" hole_out_spaced.txt > poreRadius.tsv 

    cat poreRadius.tsv|awk '{print $1,$2}' > ppr_$i.dat 

    sed -n '/ highest radius point found:/{n;p;}' hole_out_spaced.txt > porePoints.dat

    awk '{ $1=""; $2=""; print $0}' porePoints.dat > pp_$i.dat 

    rm porePoints.dat

    sed -i 's/RASEED '$seed'/RASEED '$((seed+incr_seed))'/g' hole.inp # replace current seed with its increment 

    ((seed=seed+incr_seed)) # increment

    done

    sed -i 's/RASEED '$seed'/RASEED '1000'/g' hole.inp # remember to re-set the seed after you are done!

    cd -

    done







