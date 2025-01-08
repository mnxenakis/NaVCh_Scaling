"""

  	This is your main function for Scaling Analysis.

  	It essentially orchestrates and executes all necessary tasks locally. 
  	It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.

"""

import sys
# Add the hydroscale directory to the system path for module imports
sys.path.insert(1, '/home/xxx/hydroscale')

import os
# Get the PDB code from the current directory name
pdb_code = os.getcwd()[-4:]
print("\n\n.. Starting working with molecule:", pdb_code, "found in: \n", os.getcwd())

import Methods

import time
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