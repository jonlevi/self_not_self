import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from lib import *
from lib.maxent import *

k = 9
proteome = 'Listeria'

filterlength = 12
seed = 12345

prng = np.random.RandomState(seed=seed)

df = load_proteome_as_df(proteome)
df.drop_duplicates(subset=['Sequence'], inplace=True)
seqs = df['Sequence']

train, test = train_test_split(seqs, test_size=0.5, random_state=prng)

for i, (label, data) in enumerate([('train', train), ('test', test)]):
    matrix = kmers_to_matrix(filter_unique(data, k=k, filterlength=filterlength))
    np.savetxt(f'Listeria_kmer_matrix_{label}.csv.gz', matrix, fmt='%i')
