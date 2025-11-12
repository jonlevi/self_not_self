# Fitting a maximum entropy model of a peptidome

The code in this directory shows how you go from a reference proteome to a full 2-point max ent model of the peptidome.

The steps are as follows:
1) Generate 9-mer matrix from reference proteome fasta, split data into train and test set
2) Fit model to data
3) Calculate entropies using thermodynamic integration

For an example, we will calculate a max ent model for 9-mer peptides in the Listeria proteome

## Generate Peptides
By running `python generate_reference_matrices.py`, we take in the Listeria proteome fasta file, create all possible 9-mer peptides, and split the data into a training set and test set. These will be output in compressed .csv.gz files. To do this, we use the helpful utility function [load_proteome_as_df](https://github.com/jonlevi/self_not_self/blob/main/lib/io.py#L83) to load the FASTA file, and [filter_unique](https://github.com/jonlevi/self_not_self/blob/main/lib/main.py#L130) to get unique 9-mers. We then encode the amino acid one letter characters as integers for faster processing using [kmers_to_matrix](https://github.com/jonlevi/self_not_self/blob/main/lib/main.py#L177). Finally we split the data and output the training set and test set for the next steps.

## Fit the model through MCMC and gradient descent
With the training set of peptides, we run `python fit_model.py` which fits the model's Langrange multipliers using gradient descent and MCMC sampling. The current energies are computed for a specific parameterization using the functions in `compute_energies.py`. The final equilibrium parameter set is output as a .npz file. The energies are broken down into the 2-point potts function term and the moment terms. Various helper functions are used to find (pairwise and independent) frequencies and run the sampling.

## Calculate entropies using thermodynamic integration
Once the parameters are fit, we can use thermodynamic integration to find the partition function, which is needed to calculate the entropies and probabilities. Running `python evaluate_entropies.py` will do so for the Listeria example, outputting a CSV with the entropy and the (log of) the partition function. Note that this code can also output samples from the model if you keep the matrix of peptides from MCMC chain, which can be useful for calculating things like coincidence probabilities, triplet correlations, etc. or for usage in hamming distance distribution analyses or further filtering by netMHC predictions. Note, due to a change in scipy dependencies, the line `Fint = scipy.integrate.simpson(Fprimes, x=xs)` might need to be changed to `Fint = scipy.integrate.simps(Fprimes, xs)` if you are using an older version of scipy.
        
## Final Product
We now have the following files: `Listeria_entropy`.csv, a `Listeria_kmer_matrix_test.csv.gz`, `Listeria_kmer_matrix_train.csv.gz`, `Listeria_model_params.npz`. We can further sample from the model by running something like `matrix = mcmcsampler(x0, energy, jump, **mcmc_kwargs)` (with the appropriately parametrized energy function) if we want model samples for downstream analyses.

## Divergences
We did not cover calculating divergences between two different peptidomes, but the code to do so is a trivial extension of the code above. Once two different peptidome models are fit and the partition functions are calculated, you can sample from both models and calculate the KL divergence using the two energy functions and the two partition functions. Code for this can be found at `https://github.com/andim/peptidome/blob/master/code/maxent/evaluate_dkls.py`
