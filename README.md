# How different are self and nonself?
This repo is the minimal code for reproducing the figure set and analyses for our paper   [How different are self and nonself](https://arxiv.org/abs/2212.12049)

There is a lot more code and work that went into this project, including the code for actually inferring the models and a lot of analyses that did not make it into the paper. For those, we encourage you to visit our main repo at: https://github.com/andim/peptidome 

# Running the code in this repo
Each figure in our paper can be reproduced by running the appropriate jupyter notebook in the `notebooks` folder. All dependencies used are publically available and can be pip installed per usual, although if you prefer we have an environment yml file provided that can be used to set up a conda environment.

# Downloading the data
In order to run the code in these notebooks you will need access to the formatted data that is used. Due to the size of these files, they are not included in the github repository. Most of the data can be downloaded at [this link](https://www.dropbox.com/scl/fi/nsflv77c87jy53e8kh2wu/data.zip?rlkey=lfwi5d2l8b0fv14gs9j524xvh&e=1&st=3wshuawc&dl=0) (~1.3GB). Some of the larger files (>10GB) are not included in that dropbox folder (including the MHC haplotypes data, per allele netMHC filtered peptide sets, and some of the larger markov chain monte carlo data) but can be provided in an appropriate manner upon request. Please email levinej4@mskcc.org with any requests for additional data.



