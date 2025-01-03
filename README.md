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
Specifically, we utilize the following .tcl script:
    
    # Load the cleaned and protonated .pdb file
    mol load pdb "$subdir"_clean_H.pdb

    # Calculate the principal axis of the selected atoms (e.g., protein backbone)
    set sel [atomselect top "protein"]
    set eigvec [measure inertia [\$sel get {x y z}]]

    # Align the principal axis with the z-axis
    set zaxis {0 0 1}
    set rotation_matrix [transvec \$eigvec \$zaxis]
    \$sel move \$rotation_matrix

    # Save the aligned structure
    \$sel writepdb "$subdir"_aligned.pdb
    quit






