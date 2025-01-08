"""

	This is your main function for Statistical Summarizy of NaVCh scaling characteristics.
    
    It essentially orchestrates and executes all necessary tasks locally. 
	It assumes you are working within the <SUBTYPE/SPECIES_summary> directory and utilizes imported methods to process the data.
	
"""

import sys
sys.path.insert(1, '/home/xxx/hydroscale')

import os
current_dir = os.getcwd()
dir_name = os.path.basename(current_dir)
name = dir_name.rsplit("_", 1)[0]

print(f"Working with: {name}")

import Methods

import time
start_time = time.time()
 
Methods.Summary(name, [-25, 25])
print("--- %s seconds ---" % (time.time() - start_time))

exit()
