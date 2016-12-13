_______________________________________________________________________________

Supervised learning of bag-of-features descriptors using sparse coding
_______________________________________________________________________________

	- This code implements supervised bag-of-features approach shown in the SHREC14 report.
	- If you use this code to publish, please cite using the "CITEME.bib" file.
	- Most of our new code is in the "supervised" folder.
	- I include a few trained dictionaries used for the "Shape-Google" comparison.
	
	- Keep in mind that this is an experimental code supplied "as-is".
	- Bug reports are appreciated! (but I cannot guarantee how long will it take me to fix them)

		
_______________________________________________________________________________

Reproducing results
_______________________________________________________________________________

	- SHREC14 human 
		- down-sample using the MeshLab script "QECD_4500.mlx" in the 'data' folder
		- rescale the "real" dataset by ~53.7
		- rescale the "synthetic" dataset by ~138.3

	- SHREC15 scalability 
		- down-sample using the MeshLab script "QECD_4500.mlx" in the 'data' folder 
		- rescale each shape so that the sum of all triangle area is 40,000 (similar to ShapeGoogle)
		
_______________________________________________________________________________

3rd party dependencies
_______________________________________________________________________________


1) Shape Google
	- This code is based on that of a previous work:
		"Shape Google: Geometric Words and Expressions for Invariant Shape Retrieval"
	- the 1st commit of this project is a slim version of the latter.
	- The original (full) Shape-Google version can be downloaded from:
		http://www.lix.polytechnique.fr/~maks/shapegoogle_code.zip
	- Additionally ,we used a slightly different (equi-scaled) version of the shape-data.


2) SPAMS - SPArse Modeling Software
	- we used version: 2.3, but later version will probably work as well.
	- compiled mex files for win32 and win64 are included.
	- the source for these can be downloaded from 
		http://spams-devel.gforge.inria.fr/

		
		
		
_______________________________________________________________________________

Running the code from a "clean" checkout 
_______________________________________________________________________________

1) first, run the script "./scripts/experiments/run_precomputation.m" to create:
	- LBO eigen-decomposition
	- HKS descriptors
	- dictionary with k-means clustering
	- dictionary with Unsupervised DL
	- ground-truth for shape - classes
	
2) then, run  the script "./scripts/supervised/test_SupDictlearn_split.m" and get a supervised trained dictionary


_______________________________________________________________________________

Limitation \ Issues
_______________________________________________________________________________

	- The training of M and P does not work even though they pass finite difference tests
		(see "./scripts/supervised/test_grads_FULL.m")
	- The only time I could get them to work is when the gradient descent is not stochastic (i.e. "regular" GD).


