"""


		This is your main function. It basically calls (locally) whatever you need. 
	

"""
import os
pdb_code = os.getcwd()[-4:]
print("\n \n .. Starting working with molecule:", pdb_code, "found in: \n", os.getcwd())
import sys
sys.path.insert(1, '/home/markos/hydroscale')


import time
import Methods

start_time = time.time()
 
Methods.HOLEOutputAnalysis()
Methods.PDBStructurePreperation()
Methods.CollectObservables()
# Methods.InsertVariants()
# Methods.PrepareFeatures()

Methods.InformationProfile()

print("--- %s seconds ---" % (time.time() - start_time))
exit()

# Methods_current.hydrSymmeties() 
feature_logDistMin 		= ["logDistMin"]
feature_logDistMax 		= ["logDistMax"]
feature_logDistInfl 	= ["logDistInfl"]

feature_dn 				= ["dn"]
feature_dn_pho 			= ["dn_pho"]
feature_dn_phi 			= ["dn_phi"]

feature_radial0 		= ["radial0"]
feature_radial0_pho 	= ["radial0_pho"]
feature_radial0_phi		= ["radial0_phi"]

feature_radial1_z 		= ["radial1_z_pho"]
feature_radial1_z_pho 	= ["radial1_z_pho"]
feature_radial1_z_phi 	= ["radial1_z_phi"]

feature_radial1_abs 	= ["radial1_abs"]
feature_radial1_abs_pho = ["radial1_abs_pho"]
feature_radial1_abs_phi = ["radial1_abs_phi"]

feature_radial2 		= ["radial2"]
feature_radial2_pho 	= ["radial2_pho"]

feature_radial3_z 	= ["radial3_z"]
feature_radial3_abs = ["radial3_abs"]

feature_fd 			= ["fd"]
feature_fd_phi 			= ["fd_phi"]
feature_fd_pho 			= ["fd_pho"]
feature_fd0 		= ["fd0"]
feature_fd0_pho 		= ["fd0_pho"]
feature_fd0_phi 		= ["fd0_phi"]
feature_fd1_abs 	= ["fd1_abs"]
feature_fd1_z 		= ["fd1_z"]
feature_fd1_z_pho 	= ["fd1_z_pho"]
feature_fd1_z_phi 	= ["fd1_z_phi"]
feature_fd2 		= ["fd2"]
feature_fd3_abs 	= ["fd3_abs"]
feature_fd3_z 		= ["fd3_z"]
feature_fd3_z_pho 	= ["fd3_z_pho"]
feature_fd3_z_phi 	= ["fd3_z_phi"]


features_geometries 		= ["euclDistInfl"]
features_radials			= ["radial", "radial0", "radial1_z", "radial2", "radial3_z", "radial4", "radial5_z"]
features_radials_abs		= ["radial", "radial0", "radial1_abs", "radial2", "radial3_abs", "radial3_abs"]
features_radials_pho 		= ["dn_pho", "radial0_pho", "radial1_z_pho", "radial2_pho", "radial3_z_pho"]
features_radials_phi 		= ["dn_phi", "radial0_phi", "radial1_z_phi", "radial2_phi", "radial3_z_phi"]
features_exponents_pho 		= ["fd_pho", "fd0_pho", "fd1_z_pho", "fd2_pho", "fd3_z_pho"]

features_exponents_pho_phi_ 		= ["fd_pho", "fd_phi"]
features_exponents_pho_phi_0 		= ["fd0_pho", "fd0_phi"]
features_exponents_pho_phi_1 		= ["fd1_z_pho", "fd1_z_phi"]
features_exponents_pho_phi_2 		= ["fd2_pho", "fd2_phi"]

features_exponents_z 			= ["fd", "fd0", "fd1_z", "fd2", "fd3_z", "fd4", "fd5_z"]
features_exponents_abs 			= ["fd", "fd0", "fd1_abs", "fd2", "fd3_abs"]
features_exponents_abs 		= ["fd", "fd0", "fd1_abs", "fd2", "fd3_abs"]
features_exponents_phi 		= ["fd_phi", "fd0_phi", "fd1_z_phi", "fd2_phi", "fd3_z_phi"]

chooseSeed_ = 80
balanceSeed_ = 0

"""

	Data management strategy:
	
	Train/Test your model. Then, validate it within the same group: 
	check whether all the data within the same category stem from the same distibution

	# Three main set of features always:
	a) geometries
	b) radials
	c) exponents

"""


"""		

		Group 1

"""

dataSet_disease = "GoF/LoF"
dataSet_diseaseExt = "Pathogenic(certain/likely/likely)"
dataSet_control = "Neutral"
dataSet_controlExt = "Benign(certain/likely/likely)"

dataSet_diseaseUnseen = "Pathogenic(certain/likely/likely)"
dataSet_controlUnseen = "Benign(certain/likely/likely)"

# ~ Methods_current.FeatureExploration(						["fd_phi"], 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt,
														
														# ~ r"Zero-th order hydration force ($z$-comp.)", XLABEL = True
												
									# ~ )

# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_radials_pho, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "radials_pho" )


# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_radials_phi, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "radials_phi" )
														
# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_pho, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "features_exponents_pho" )


# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_phi, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "features_exponents_phi" )

# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_radials, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "radials" )
													
													
# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_geometries, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "geometries"
													
													# ~ )
													
# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_pho_phi_, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "exponents_pho_phi_"
													
												# ~ )
												
Methods_current.FeatureEvaluationAlongPore_fourth(	
														features_exponents_z, 
												
														dataSet_disease, 
														dataSet_diseaseExt, 
												
														dataSet_control, 
														dataSet_controlExt, 
												
														"exponents_z"
													
												)

# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_abs, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "exponents_abs"
													
											# 	)

# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_pho_phi_0, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "exponents_pho_phi_0"
													
												# ~ )	

# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_pho_phi_1, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "exponents_pho_phi_1"
													
												# ~ )		
												
# ~ Methods_current.FeatureEvaluationAlongPore_fourth(	
														# ~ features_exponents_pho_phi_2, 
												
														# ~ dataSet_disease, 
														# ~ dataSet_diseaseExt, 
												
														# ~ dataSet_control, 
														# ~ dataSet_controlExt, 
												
														# ~ "exponents_pho_phi_2"
													
												# ~ )											
													
# ~ Methods_current.ScoresEvaluation(				
												# ~ ["geometries", "radials", "exponents_"], # "fd1_z_pho_group1", "fd1_z_phi_group1"], 

													# ~ dataSet_disease, 
													# ~ dataSet_diseaseExt, 
												
													# ~ dataSet_control, 
													# ~ dataSet_controlExt,
													
													# ~ dataSet_diseaseUnseen,
													# ~ dataSet_controlUnseen, chooseSeed_ = 0, score_type = "auc", score_entry = 0
												
													
													
												# ~ )
													

exit()

Methods_current.FeatureEvaluationAlongPore_fourth(	features_radials_pho, 
												
													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt, 
												
													"radials_pho_group1",
												
													balanceSeed_ = balanceSeed_ , chooseSeed_ = chooseSeed_
												
													)
													
Methods_current.FeatureEvaluationAlongPore_fourth(	features_radials_phi, 
												
													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt, 
												
													"radials_phi_group1",
												
													balanceSeed_ = balanceSeed_ , chooseSeed_ = chooseSeed_
												
													)

													
Methods_current.FeatureEvaluationAlongPore_fourth(	features_exponents_pho, 
												
													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt, 
												
													"exponents_pho_group1",
												
													balanceSeed_ = balanceSeed_ , chooseSeed_ = chooseSeed_
												
													)
													
Methods_current.FeatureEvaluationAlongPore_fourth(	features_exponents_phi, 
												
													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt, 
												
													"exponents_phi_group1",
												
													balanceSeed_ = balanceSeed_ , chooseSeed_ = chooseSeed_
												
													)
													
													
Methods_current.ScoresEvaluation(					["geometries_group1", "radials_pho_group1", "radials_phi_group1", "exponents_pho_group1", "exponents_phi_group1"], 

													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt,
													
													dataSet_diseaseUnseen,
													dataSet_controlUnseen,	
												
													balanceSeed_ = balanceSeed_ , chooseSeed_ = chooseSeed_
													
													)
													
exit()											
# Methods_current.FeatureEvaluationAlongPore_sec(features_geometries, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "geometries_group1")
# Methods_current.FeatureEvaluationAlongPore_sec(features_radials, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "radials_group1")
# Methods_current.FeatureEvaluationAlongPore_sec(features_exponents, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "exponents_group1")
# ~ Methods_current.ScoresEvaluation(["geometries_group1", "radials_group1", "exponents_group1"], balanceSeed_, 
													# ~ chooseSeed_,  

													# ~ dataSet_disease, 
													# ~ dataSet_diseaseExt, 
												
													# ~ dataSet_control, 
													# ~ dataSet_controlExt
												
													# ~ )

"""

	Group 2
	
"""

# ~ dataSet_disease = "GoF/LoF/Pathogenic(certain)"
# ~ dataSet_diseaseExt = "Pathogenic(likely)"
# ~ dataSet_control = "Neutral/Benign(certain)"
# ~ dataSet_controlExt = "Benign(likely/likely)"

# ~ Methods_current.FeatureEvaluationAlongPore_sec(features_geometries, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "geometries_group2")
# ~ Methods_current.FeatureEvaluationAlongPore_sec(features_radials, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "radials_group2")
# ~ Methods_current.FeatureEvaluationAlongPore_sec(features_exponents, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "exponents_group2")
# ~ Methods_current.ScoresEvaluation(["geometries_group2", "radials_group2", "exponents_group2"], balanceSeed_, 
													# ~ chooseSeed_,  

													# ~ dataSet_disease, 
													# ~ dataSet_diseaseExt, 
												
													# ~ dataSet_control, 
													# ~ dataSet_controlExt
												
													# ~ )


"""

	Group 3
	
"""

dataSet_disease = "GoF/LoF/Pathogenic(certain/likely)"
dataSet_diseaseExt = "ClinVar"
dataSet_control = "Neutral/Benign(certain/likely)"
dataSet_controlExt = "LikelyBenign"

dataSet_diseaseUnseen = "ClinVar" 
dataSet_controlUnseen = "LikelyBenign" 

# Methods_current.FeatureEvaluationAlongPore_fourth(features_geometries, dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, "geometries_group4")
# Methods_current.FeatureEvaluationAlongPore_third(features_radials, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, dataSet_diseaseUnseen, dataSet_controlUnseen, "radials_group3")
# Methods_current.FeatureEvaluationAlongPore_third(features_exponents, balanceSeed_, chooseSeed_,  dataSet_disease, dataSet_diseaseExt, dataSet_control, dataSet_controlExt, dataSet_diseaseUnseen, dataSet_controlUnseen, "exponents_group3")
Methods_current.ScoresEvaluation(["geometries_group3", "radials_group3", "exponents_group3"], balanceSeed_, 
													chooseSeed_,  

													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt,
													
													dataSet_diseaseUnseen, dataSet_controlUnseen
												
													)
													
Methods_current.ScoresEvaluation_sec(["geometries_group3", "radials_group3", "exponents_group3"], 
													

													dataSet_disease, 
													dataSet_diseaseExt, 
												
													dataSet_control, 
													dataSet_controlExt,
													
													dataSet_diseaseUnseen, 
													dataSet_controlUnseen,
													
													
													balanceSeed_ = balanceSeed_ 
													# chooseSeed_ = None
												
													)

# ~ feature_names = ["dn", "radial0", "radial1_z", "radial2", "radial3_z"]

# ~ # score_name = "geometry"
# ~ score_name = "radials"
# ~ # score_name = "fds"

# ~ Methods_current.FeatureEvaluationAlongPore_sec(	
													# ~ feature_names,
 
													# ~ balanceSeed_, 
													# ~ chooseSeed_,  

													# ~ dataSet_disease, 
													# ~ dataSet_diseaseExt, 
												
													# ~ dataSet_control, 
													# ~ dataSet_controlExt, 
												
												
													# ~ score_name 
												
												# ~ )

# data_set_control = "Benign(likely/likely)"
# data_set_disease = "Pathogenic(likely/likely)"

# Methods_current.FeatureEvaluationAlongPore(["dn", "radial0", "radial1_z", "radial2", "radial3_z"], 0,  data_set_disease, data_set_control, "dn")
# ~ Methods_current.ScoresEvaluation(["geometry", "radials", "fds"], balanceSeed_, 
													# ~ chooseSeed_,  

													# ~ dataSet_disease, 
													# ~ dataSet_diseaseExt, 
												
													# ~ dataSet_control, 
													# ~ dataSet_controlExt
												
												
													# ~ )						    									# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["radial0"], 0,  data_set_disease, data_set_control, "radial0") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["radial0"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["radial1_z"], 0,  data_set_disease, data_set_control, "radial1_z") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["radial1_z"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["radial2"], 0,  data_set_disease, data_set_control, "radial2") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["radial2"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["radial3_z"], 0,  data_set_disease, data_set_control, "radial3_z") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["radial3_z"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %


# Methods_current.FeatureEvaluationAlongPore(["fd"], 0, data_set_disease_learn, data_set_control_learn, "fd", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd_pho"], 0, data_set_disease_learn, data_set_control_learn, "fd_pho", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd_phi"], 0, data_set_disease_learn, data_set_control_learn, "fd_phi", 5)

# ~ Methods_current.FeatureEvaluationAlongPore(["fd0_pho"], 0, data_set_disease_learn, data_set_control_learn, "fd0_pho", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd0_phi"], 0, data_set_disease_learn, data_set_control_learn, "fd0_phi", 5)

# ~ Methods_current.FeatureEvaluationAlongPore(["fd1_z_phi"], 0, data_set_disease_learn, data_set_control_learn, "fd1_z_phi", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd1_z_pho"], 0, data_set_disease_learn, data_set_control_learn, "fd1_z_pho", 5)

# ~ Methods_current.FeatureEvaluationAlongPore(["fd2_phi"], 0, data_set_disease_learn, data_set_control_learn, "fd2_phi", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd2_pho"], 0, data_set_disease_learn, data_set_control_learn, "fd2_pho", 5)

# ~ Methods_current.FeatureEvaluationAlongPore(["fd3_z_pho"], 0, data_set_disease_learn, data_set_control_learn, "fd3_z_pho", 5)
# ~ Methods_current.FeatureEvaluationAlongPore(["fd3_z_phi"], 0, data_set_disease_learn, data_set_control_learn, "fd3_z_phi", 5)

# Methods_current.ScoresEvaluation(["fd_pho", "fd_phi", "fd0_pho", "fd0_phi", "fd1_z_phi", "fd1_z_pho", "fd2_pho", "fd2_phi", "fd3_z_pho", "fd3_z_phi"], 0, data_set_disease, data_set_control, 8) 
# Methods_current.FeatureEvaluationAlongPore_sec(["dn_pho", "radial0_pho", 	"radial1_z_pho", "radial2_pho", "radial3_z_pho"], 0,  data_set_disease, data_set_control, "dn_radial0_radial1_z_radial2_radial3_z_PHO", 5, 0)
# Methods_current.FeatureEvaluationAlongPore(["dn", "radial0", 	"radial1_abs", "radial2", "radial3_abs"], 0,  data_set_disease_learn, data_set_control_learn, "radial_ABS", 5)
# Methods_current.FeatureEvaluationAlongPore(["dn", "radial0", 	"radial1_z", "radial2", "radial3_z"], 0,  data_set_disease_learn, data_set_control_learn, "radial_z", 5)
# Methods_current.FeatureEvaluationAlongPore(["dn", "radial0", 	"radial1_z", "radial2", "radial3_z"], 0,  data_set_disease_learn, data_set_control_learn, "radial_z", 5)
# Methods_current.FeatureEvaluationAlongPore(["dn_phi", "radial0_phi", 	"radial1_z_phi", "radial2_phi", "radial3_z_phi"], 0,  data_set_disease_learn, data_set_control_learn, "dn_radial0_radial1_z_radial2_radial3_z_PHI", 5)
# Methods_current.FeatureEvaluationAlongPore(["fd_pho", "fd0_pho", 		"fd1_z_pho", 		"fd2_pho", 	"fd3_z_pho"], 0,  data_set_disease_learn, data_set_control_learn, "fd_fd0_fd1_z_fd2_fd3_z_PHO", 5)
# Methods_current.FeatureEvaluationAlongPore(["fd_phi", "fd0_phi", 		"fd1_z_phi", 		"fd2_phi", 	"fd3_z_phi"], 0,  data_set_disease_learn, data_set_control_learn, "fd_fd0_fd1_z_fd2_fd3_z_PHI", 5)
# Methods_current.FeatureEvaluationAlongPore(["fd_pho", "fd0_pho", 		"fd1_z_pho", 		"fd2_pho", 	"fd3_z_pho"], 0,  data_set_disease_learn, data_set_control_learn, "fd_pho_fd0_pho_fd1_z_pho_fd2_pho_fd3_z_pho", 5)
# Methods_current.FeatureEvaluationAlongPore(["fd_phi", "fd0_phi", 		"fd1_z_phi", 		"fd2_phi", 	"fd3_z_phi"], 0,  data_set_disease_learn, data_set_control_learn, "fd_phi_fd0_phi_fd1_z_phi_fd2_phi_fd3_z_phi", 5)
# Methods_current.FeatureEvaluationAlongPore(["logDistMax", "logDistInfl", "logDistMin"], 0, data_set_disease_learn, data_set_control_learn, "geometry", 3)  	 
# Methods_current.FeatureEvaluationAlongPore(["fd", "fd0", 		"fd1_z", 		"fd2", 	"fd3_z"], 0, data_set_disease_learn, data_set_control_learn, "fractal_exp_z", 3)  	 
# Methods_current.FeatureEvaluationAlongPore(["fd", "fd0", 		"fd1_abs", 		"fd2", 	"fd3_abs"], 0, data_set_disease_learn, data_set_control_learn, "fractal_exp_abs", 3)  	 
# Methods_current.ScoresEvaluation(["dn_radial0_radial1_z_radial2_radial3_z_PHO", 
#	"dn_radial0_radial1_z_radial2_radial3_z_PHI", "Max_Infl_Min", "fd_fd0_fd1_z_fd2_fd3_z_PHO", "fd_fd0_fd1_z_fd2_fd3_z_PHI"], 0, data_set_disease, data_set_control, 5) 
# Methods_current.ScoresEvaluation(["geometry", "fractal_exp_z", "radial_z"], 0, data_set_disease, data_set_control, 3) 
# Methods_current.ScoresEvaluation(["fd0_pho_fd1_z_pho_fd2_pho_fd3_z_pho"], 0, data_set_disease, data_set_control, 4) 



# Methods_current.ScoresEvaluation(["fd_fd0_fd1_z_fd2_fd3_z", "Max_Infl_Min", "dn_radial0_radial1_z_radial2_radial3_z"], 0, data_set_disease, data_set_control, 3) 

# ~ Methods_current.FeatureEvaluationAlongPore(["fd"], 0,  data_set_disease, data_set_control, "fd")
# ~ Methods_current.ScoresEvaluation(["fd"], 0, data_set_disease, data_set_control) 						    									# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["fd0"], 0,  data_set_disease, data_set_control, "fd0") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["fd0"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["fd1_z"], 0,  data_set_disease, data_set_control, "fd1_z") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["fd1_z"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["fd2"], 0,  data_set_disease, data_set_control, "fd2") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["fd2"], 0, data_set_disease, data_set_control) 						    								# accuracy: 69 %
# ~ Methods_current.FeatureEvaluationAlongPore(["fd3_z"], 0,  data_set_disease, data_set_control, "fd3_z") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["fd3_z"], 0, data_set_disease, data_set_control) 	


# Methods_current.FeatureEvaluationAlongPore(["dn", "radial0", "radial1_z", "radial2" , "radial3_z"], 0, data_set_disease, data_set_control, "radial_all") 						    								# accuracy: 69 %
# Methods_current.ScoresEvaluation(["radial_all"], 0, data_set_disease, data_set_control) 

# ~ Methods_current.FeatureEvaluationAlongPore(["fd", "fd0", "fd1_z", "fd2", "fd3_z"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)", "exponents") 						    								# accuracy: 69 %
# ~ Methods_current.ScoresEvaluation(["exponents"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)")




# Methods_current.ScoresEvaluation(["radial", "geometry", "exponents"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)")

# ~ Methods_current.FeatureEvaluationAlongPore(["radial0_pho_to_phi","radial1_z_pho_to_z_phi", "radial2_pho_to_phi", "radial3_z_pho_to_z_phi"], 0, "GoF/LoF/Pathogenic(certain)", "Neutral/Benign(certain/likely)")  
# ~ Methods_current.ScoresEvaluation(["radial0_pho_to_phi_radial1_z_pho_to_z_phi"], 0, "GoF/LoF/Pathogenic(certain)", "Neutral/Benign(certain/likely)")   

# Methods_current.FeatureEvaluationAlongPore(["radial0_pho", "radial1_z_pho", "radial2_pho" , "radial3_z_pho"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)")  						    								# accuracy: 69 %
# Methods_current.ScoresEvaluation(["radial0_pho_radial1_z_pho"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)")  

# Methods_current.FeatureEvaluationAlongPore(["radial0_phi", "radial1_z_phi", "radial2_phi" , "radial3_z_phi"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)") 						    								# accuracy: 69 %
# Methods_current.ScoresEvaluation(["radial0_phi_radial1_z_phi"], 0, "GoF/LoF/Pathogenic(certain/likely)", "Neutral/Benign(certain/likely)", "intrac")  

# Methods_current.FeatureEvaluationAlongPore(["fd0_pho_to_phi", "fd1_z_pho_to_z_phi", "fd2_pho_to_phi" , "fd3_z_pho_to_z_phi"], 0) 						    								# accuracy: 69 %
# Methods_current.ScoresEvaluation(["fd0_pho_to_phi_fd1_z_pho_to_z_phi"], 0)

# Methods_current.FeatureEvaluationAlongPore(["radial0_phi", "radial1_z_phi", "radial2_phi" , "radial3_z_phi"]) 						    								# accuracy: 69 %

						    								# accuracy: 69 %
# Methods_current.ScoresEvaluation(["logDistInfl_logDistMax", "fd_fd0", "radial0_radial1_z"]) 						    								# accuracy: 69 %

						    	# accuracy: 69 %
# Methods_current.FeatureEvaluationAlongPore(["fd", "fd0"]) 						    	# accuracy: 69 %
# Methods_current.FeatureEvaluationAlongPore(["fd", "fd0", "fd1_z", "fd2", "fd3_z"]) 								    	# accuracy: 69 %
# Methods_current.FeatureEvaluationAlongPore(["radial0", "radial1_z", "radial2", "radial3_z"]) 								    	# accuracy: 69 %
# Methods_current.FeatureEvaluationAlongPore(["fd_pho", "fd0_pho", "fd1_z_pho", "fd2_pho", "fd3_z_pho"]) 	    		# accuracy: 70 %
#Methods_current.FeatureEvaluationAlongPore(["fd0_phi", "fd0_pho", "fd2_phi", "fd2_pho"]) 	    	# accuracy: 66 %

# Methods_current.FeatureEvaluationAlongPore(["logDistInfl", "radial0", "radial1_abs", "radial2", "radial3_abs"])	    # accuracy: 74 %
# Methods_current.FeatureEvaluationAlongPore(["radial0", "radial1_z", "radial2", "radial3_z"])						# accuracy: 75 %
# Methods_current.FeatureEvaluationAlongPore(["radial0_pho", "radial1_z_pho", "radial2_pho", "radial3_z_pho"])	# [1538.  185.  720.  172.   1262.  121.  773. 1169. 1242. 1500.]
# Methods_current.FeatureEvaluationAlongPore(["radial0_phi", "radial1_z_phi", "radial2_phi", "radial3_z_phi"])	# [1538.  185.  172.  1262.  121.   773.  1500.]

# Methods_current.FeatureEvaluationAlongPore(["radialMol0", "radialMol1_z", "radialMol2", "radialMol3_z"])		# fails
# Methods_current.FeatureEvaluationAlongPore(["radialMol0_pho", "radialMol1_abs_pho", "radialMol2_pho", "radialMol3_abs_pho"])	# fails


# Methods_current.FeatureEvaluationAlongPore(["logDistMinSEC", "logDistMinTHIRD", "logDistMinHFTB", "logDistMinHPB", "logDistInfl", "fd"])
# Methods_current.FeatureEvaluationAlongPore(["fd_pho", "exp_pho", "exp_dehydr", "exp_pho_sec", "exp_dehydr_third"])
# Methods_current.FeatureEvaluationAlongPore(["fd", "exp_pho"])
# Methods_current.testing()
# Methods_current.CombineFeaturesSec(["exp_phi_f", "exp_pho_f"])
# Methods_current.FeatureIntraDistances("radial1_abs_to_0_f")
# Methods_current.Clustering(["exp_phi_maxMedian",s "exp_dehydr_maxMedian", "fd_maxMedian"])
# Methods_current.Clustering(["exp_phi_sec_maxMedian", "exp_dehydr_third_maxMedian", "exp_pho_sec_maxMedian"])
# Methods_current.Clustering(["exp_phi_maxMedian", "exp_dehydr_maxMedian", "fd_phi_maxMedian"])

# Methods_current.Clustering(["fd_phi_f_maxMedian", "exp_hydr_third_f_maxMedian", "exp_phi_sec_f_maxMedian", "logDistInfl_f_maxMedian"])
# Methods_current.Clustering(["fd_phi_maxMedian"])
# Methods_current.Clustering(["fd_phi_f_maxMedian", "exp_phi_f_maxMedian"])
# Methods_current.Clustering(["fd_phi_f_maxMedian", "exp_phi_f_maxMedian", "exp_dehydr_f_maxMedian"])
# Methods_current.Clustering(["logDistInfl_f_maxMedian", "exp_phi_f_maxMedian", "exp_hydr_f_maxMedian"])
# Methods_current.Clustering(["fd_phi_f_maxMedian", "exp_phi_f_maxMedian", "exp_hydr_f_maxMedian", "exp_phi_sec_f_maxMedian"]
# Methods_current.Clustering(["fd_phi_maxMedian", "exp_phi_maxMedian", "exp_hydr_maxMedian", "exp_phi_sec_maxMedian", "exp_hydr_third_maxMedian"])
# Methods_current.Clustering(["fd_phi_maxMedian", "exp_phi_maxMedian", "exp_hydr_maxMedian", "exp_phi_sec_maxMedian", "exp_hydr_third_maxMedian", "logDistHPB_maxMedian"])
# Methods_current.Clustering(["logDistInfl_maxMedian", "exp_hydr_third_maxMedian", "exp_phi_sec_maxMedian", "fd_phi_maxMedian"])
# Methods_current.Regressing(["logDistHPB_resMedian"])
# Methods_current.Clustering(["logDistTHIRD_maxMedian", "exp_hydr_third_maxMedian", "exp_phi_sec_maxMedian"])
# Methods_current.Clustering(["logDistInfl_maxMedian", "exp_dehydr_maxMedian", "exp_pho_maxMedian"])
# Methods_current.Clustering(["fd_maxMedian", "exp_hydr_third_maxMedian", "exp_phi_sec_maxMedian"])

# Methods_current.PDBStructurePreperation()
# exit()
# Methods_current.CalculateEntropies()
# Methods2_current.GeometricAnalysis()

# Methods2.EntropicAnalysis()

exit()

# After succesful "preparation", let's proceed with the real stuff: start the analysis! 

import analysis_toolbox
analysis_toolbox.analyze_pdb(molChar)

