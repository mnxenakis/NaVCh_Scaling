"""
This is your main function for Statistical Summary of NaVCh scaling characteristics.
    
It essentially orchestrates and executes all necessary tasks locally. 
It assumes you are working within the <SUBTYPE/SPECIES_summary> directory and utilizes imported methods to process the data.
"""

import os
import sys
import time

# Get the current directory and extract the name
current_dir = os.getcwd()
dir_name = os.path.basename(current_dir)
name = dir_name.rsplit("_", 1)[0]

print(f"Working with: {name}")
print(f"Working directory: {current_dir}")

# Add the hydroscale directory to the system path for module imports
# Note: Update the path below to match your actual hydroscale directory
hydroscale_path = '/home/xxx/hydroscale'  # TODO: Update this path
sys.path.insert(0, hydroscale_path)

try:
    import Methods
    
    # Record the start time for performance measurement
    start_time = time.time()
    
    print("Starting statistical summary analysis...")
    
    # Generate summary with default range [-25, 25]
    Methods.Summary(name, [-25, 25])
    
    # Display the elapsed time for the full process
    elapsed_time = time.time() - start_time
    print(f"\n✓ Summary analysis completed successfully!")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
except ImportError as e:
    print(f"Error: Could not import Methods module. Please check the hydroscale path: {hydroscale_path}")
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during summary analysis: {e}")
    sys.exit(1)
