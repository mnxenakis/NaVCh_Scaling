"""
This is your main function for Scaling Analysis.

It essentially orchestrates and executes all necessary tasks locally. 
It assumes you are working within the <PDB_code> directory and utilizes imported methods to process the data.
"""

import os
import sys
import time

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
    
    # Record the start time for performance measurement
    start_time = time.time()
    
    print("Starting analysis pipeline...")
    
    # Call the sequence of methods required for the analysis
    print("1. Analyzing HOLE output...")
    Methods.HOLEOutputAnalysis()       # Analyze the output of HOLE
    
    print("2. Preparing PDB structure...")
    Methods.PDBStructurePreperation()  # Prepare the PDB structure
    
    print("3. Collecting observables...")
    Methods.CollectObservables()       # Collect observables for the analysis
    
    print("4. Generating information profile...")
    Methods.InformationProfile()       # Generate the information profile
    
    # Display the elapsed time for the full process
    elapsed_time = time.time() - start_time
    print(f"\n✓ Analysis completed successfully!")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
except ImportError as e:
    print(f"Error: Could not import Methods module. Please check the hydroscale path: {hydroscale_path}")
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during analysis: {e}")
    sys.exit(1)