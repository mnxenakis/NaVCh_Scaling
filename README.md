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

    mkdir -p ~/NaVCh_ScalingAnalysis/1ABC

This ensures that PDB-related files are stored in the NaVCh_ScalingAnalysis directory instead of hydroscale.





