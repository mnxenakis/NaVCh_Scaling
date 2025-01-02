import numpy as np
import pandas as pd
from statistics import mode
from matplotlib import *
import matplotlib.pyplot as plt


"""
    An atomc level hydropathic scale (unweighted)
    Note that: a HE2 atom is added in the histidine. Its value is -1.
    
"""


alpha 	= 0.5
beta 	= 2*alpha
gama 	= 2*beta


hScale = {

    'PHE':	pd.Series([beta,  		-alpha,  		beta,  			beta,  		-alpha,   		
					-alpha,  		-alpha,  		-alpha,  		-alpha,   	-alpha,  
					-alpha,   		 beta,   		-alpha,    		-alpha,  	-alpha,  
					-alpha,  		-alpha,  		-alpha,  		-alpha, 	-alpha], 
		index	=	['N',    		'CA',    		'C',    		'O',    	'CB',     
					 'CG',     		'CD1',   		'CD2',   		'CE1',   	'CE2',   
					 'CZ',     		'H',     		'HA',      		'HB2',   	'HB3',    
					 'HD1',   		'HD2',   		'HE1',   		'HE2',  	'HZ']),

    'PRO': pd.Series([alpha,		-alpha,  		beta,  			beta,  		-alpha,			
					-alpha,  		-alpha,  		-alpha,  		-alpha,   	-alpha,  
					-alpha,   		-alpha,  		-alpha,    		-alpha], 		
		index	=	['N',     		'CA',    		'C',    		'O',    		'CB',     
					 'CG',     		'CD',    		'HA',    		'HB2',   		'HB3',   
					 'HG2',    		'HG3',    		'HD2',    		'HD3']),

    'ILE': 	pd.Series([beta, 		-alpha,  		beta,  			beta,  			-alpha,  	
					-alpha,  		-alpha,  		-alpha,   		beta,   		-alpha,  
					-alpha,   		-alpha,  		-alpha,    		-alpha,  		-alpha,  
					-alpha,  		-alpha,  		-alpha,  		-alpha], 
		index	=	['N',     		'CA',    		'C',    		'O',    		'CB',     
					 'CG1',    		'CG2',   		'CD1',   		'H',     		'HA',    
					 'HB',     		'HG12',   		'HG13',   		'HG21',  		'HG22',  
					 'HG23',  		'HD11',  		'HD12',  		'HD13']),
		
    'LEU': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					-alpha,  		-alpha,  		-alpha,   		beta,   		-alpha,  
					-alpha,   		-alpha,   		-alpha,   		-alpha,  		-alpha,  
					-alpha,  		-alpha,  		-alpha,  		-alpha], 
		index	=	['N',     		'CA',    		'C',    		'O',    		'CB',     
					'CG',     		'CD1',   		'CD2',   		'H',     		'HA',    
					'HB2',    		'HB3',    		'HG',     		'HD11',  		'HD12',  
					'HD13',  		'HD21',  		'HD22',   		'HD23']),
		
    'TRP': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					-alpha,  		-alpha,   		-alpha,  		beta,   		-alpha,  
					-alpha,   		-alpha,   		-alpha,   		-alpha,   		beta,   
					-alpha, 		-alpha,  		-alpha,   		-alpha,   		beta,  
					-alpha, 		-alpha, 		-alpha, 		-alpha], 
		index	=	['N',     		'CA',    		'C',    		'O',   			'CB',     
					'CG',     		'CD1',   		'CD2',   		'NE1',   		'CE2',   
					'CE3',    		'CZ2',    		'CZ3',    		'CH2',    		'H',     
					'HA',    		'HB2',  		'HB3',    		'HD1',    		'HE1',   
					'HE3',  		'HZ2',  		'HZ3',  		'HH2']),
			
    'VAL': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					-alpha,  		-alpha,   		beta,   		-alpha,  		-alpha,  
					-alpha,   		-alpha,   		-alpha,   		-alpha,   		-alpha,  
					-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',    		'CB',     
					'CG1',    		'CG2',    		'H',     		'HA',   		'HB',    
					'HG11',   		'HG12',   		'HG13',   		'HG21',   		'HG22',  
					'HG23']),
			
    'TYR': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	-
					alpha,  		-alpha,   		-alpha,  		-alpha,  		-alpha,   
					-alpha,   		beta,    		beta,   		-alpha,   		-alpha,  
					-alpha,  		-alpha,  		-alpha,  		-alpha,  		-alpha,  
					beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'CD1',   		'CD2',   		'CE1',   		'CE2',    
					'CZ',     		'OH',     		'H',      		'HA',    		'HB2',    
					'HB3',  		'HD1',   		'HD2',   		'HE1',    		'HE2',   
					'HH']),
			
    'MET': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					-alpha,   		beta,   		-alpha,   		beta,  			-alpha,   
					-alpha,  		-alpha,   		-alpha,  		-alpha,    		-alpha,  
					-alpha,  		-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',     
					'CG',    		'SD',     		'CE',    		'H',     		'HA',     
					'HB2',  		'HB3',     		'HG2',   		'HG3',    		'HE1',    
					'HE2',  		'HE3']),
			
    'ALA': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					beta,   		-alpha,   		-alpha,   		-alpha, 		-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',     
					'H',     		'HA',     		'HB1',   		'HB2',  		'HB3']),
			
    'THR': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  		
					beta,   		-alpha,    		beta,   		-alpha,  		-alpha,  
					beta,   		-alpha,   		-alpha,   		-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'OG1',   		'CG2',      	'H',    		'HA',    		'HB',    
					'HG1',   		'HG21',   		'HG22',   		'HG23']),
			
    'GLY': pd.Series([beta,  		-alpha,  		beta,  			beta,   		beta,   		
					-alpha,  		-alpha], 
	index	=		['N',     		'CA',   		'C',    		'O',     		'H',     
					'HA2',   		'HA3']),
			
    'CYS': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  		
					beta,    		beta,   		-alpha,  		-alpha,   		-alpha,  
					beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'SG',      		'H',     		'HA',    		'HB2',   		'HB3',   
					'HG']),
			
    'SER': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					beta,    		beta,   		-alpha,   		-alpha,  		-alpha,  
					beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'OG',      		'H',     		'HA',    		'HB2',   		'HB3',   
					'HG']),
			
    'GLN': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					-alpha,   		beta,    		beta,    		beta,   		beta,  
					-alpha,   		-alpha,  		-alpha,  		-alpha,  		-alpha,   
					beta,   		beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',     
					'CG',     		'CD',    		'OE1',   		'NE2',    		'H',    
					'HA',     		'HB2',   		'HB3',   		'HG2',   		'HG3',   
					'HE21',   		'HE22']),
			
    'HIS': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					-alpha,   		beta,    		-alpha,   		beta,   		beta,   
					beta,   		-alpha,  		-alpha,  		-alpha,   		beta,   
					-alpha,   		beta,			beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'ND1',    		'CD2',    		'CE1',   		'NE2',   
					'H',     		'HA',    		'HB2',   		'HB3',    		'HD1',   
					'HD2',    		'HE1', 			'HE2']), 
			
    'LYS': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					-alpha,  		-alpha,    		-alpha,   		gama,    		beta,   
					-alpha,  		-alpha,  		-alpha,  		-alpha,  		-alpha,  
					-alpha,  		-alpha,  		-alpha, 		-alpha,  		gama,  
					gama,   		gama], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'CD',     		'CE',     		'NZ',    		'H',     
					'HA',    		'HB2',   		'HB3',   		'HG2',    		'HG3',   
					'HD2',   		'HD3',  		'HE2',   		'HE3', 			'HZ1',  
					'HZ2',  		'HZ3']),
			
    'GLU': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					-alpha,   		gama,     		gama,     		gama,    		beta,   
					-alpha,  		-alpha,   		-alpha,  		-alpha,  		-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'CD',     		'OE1',    		'OE2',   		'H',     
					'HA',    		'HB2',    		'HB3',  		'HG2',    		'HG3']),
			
    'ASN': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					beta,    		beta,    		beta,    		beta,  			-alpha,   
					-alpha,  		-alpha,    		beta,  			beta], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'OD1',    		'ND2',     		'H',    		'HA',    
					'HB2',   		'HB3',    		'HD21',  		'HD22']),
			
    'ASP': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	
					gama,     		gama,     		gama,     		beta,  			-alpha,   
					-alpha,  		-alpha], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'OD1',    		'OD2',     		'H',    		'HA',    
					'HB2',   		'HB3']),
			
    'ARG': pd.Series([beta,  		-alpha,  		beta,  			beta,  			-alpha,  	 
					-alpha,  		-alpha,    		beta,    		gama,     		gama,    
					gama,    		beta,   		-alpha,  		-alpha, 		-alpha,  
					-alpha,  		-alpha,  		-alpha,  		-alpha,   		beta,    
					gama,   		gama,   		gama,   		gama], 
	index	=		['N',     		'CA',    		'C',    		'O',     		'CB',    
					'CG',     		'CD',     		'NE',      		'CZ',   		'NH1',   
					'NH2',   		'H',      		'HA',    		'HB2',   		'HB3',   
					'HG2',  		'HG3',   		'HD2',   		'HD3',    		'HE',    
					'HH11', 		'HH12', 		'HH21', 		'HH22']),
		

}

df = pd.DataFrame(hScale)


"""

		Statistics of the atomc level hydropathic scale (unweighted)

    
"""

# scores = [] 
# # .. Let's do some statistics .. for the sake of sanity check!
# for res_name in df.columns:
# 	print("\n Hydropathic profile of residue: ", res_name)
# 	hydroChar = df[res_name].iloc[:].dropna()
# 	hydroChar_ind = hydroChar.index[:]
# 	hydroChar_val = hydroChar[:]
# 	print(" Total hydropathic score: ", np.sum(hydroChar_val))
# 	print(" Average hydropathic score: ", np.mean(hydroChar_val))
# 	print(" Mode of hydropathic score: ", mode(hydroChar_val))
# 	print(" Median of hydropathic score: ", np.median(hydroChar_val))
# 	print(" Std of hydropathic score: ", np.std(hydroChar_val))
# 	scores = np.append(scores, hydroChar_val)


