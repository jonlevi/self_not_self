# Fitting a maximum entropy model of a peptidome

The code in this directory shows how you go from a reference proteome to a full 2-point max ent model of the peptidome.

The steps are as follows:
1) Generate 9-mer matrix from reference proteome fasta, split data into train and test set
2) Fit model to data
3) Calculate entropies using thermodynamic integration

For an example, we will calculate a max ent model for 9-mer peptides in the Listeria proteome

## Generate Peptides
By running `python generate_reference_matrices.py`, we take in the Listeria proteome fasta file, create all possible 9-mer peptides, and split the data into a training set and test set. These will be output in compressed .csv.gz files.

## Fit the model through MCMC and gradient descent
We run `python fit_model.py` which takes the training peptide set saved from the previous step, and fits the Langrange multipliers using gradient descent and MCMC sampling. The energies are computed for a specific parameterization using the functions in `compute_energies.py`. The final parameter set is output as a .npz file.

## Calculate entropies using thermodynamic integration
Once the parameters are fit, we can use thermodynamic integration and MCMC sampling to find the partition function and then calculate the entropies. Running `python evaluate_entropies.py` will do so for the Listeria example, outputting a CSV with the entropy and the log of the partition function. Note that this code can also output samples from the model if you keep the matrix of peptides from MCMC chain, which can be useful for calculating things like coincidence probabilities, triplet correlations, etc. or for usage in hamming distance distribution analyses or further filtering by netMHC predictions. 
