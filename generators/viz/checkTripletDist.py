from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.style.use('seaborn-white')
sns.set_palette('dark')


all_triplets = {
    'E8_F4': [12, 13, 14],
    'S16_E8_1': [8, 9, 12],
    'S16_E8_2': [10, 11, 13],
    'R1_R2_1': [0, 1, 8],
    'R1_R2_2': [2, 3, 9],
    'R1_R2_3': [4, 5, 10],
    'R1_R2_4': [6, 7, 11],
    'P_S1': [0, 8, 12],
    'P_S2': [7, 11, 13]
}


def load_brackets(fmt='TTT'):
    with open('allBrackets{}.json'.format(fmt)) as f:
        brackets = json.load(f)['brackets']
        return brackets


def pool_dist(filepath, key):
    with open(filepath) as f:
        data = json.load(f)
        d = data['triplets'][key]
        for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
            if t not in d:
                d[t] = 0
        return d


def observed_dist(year, key, brackets):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    triplets, counts = np.unique(vectors[:, all_triplets[key]], axis=0, return_counts=True)
    triplets = np.apply_along_axis(''.join, 1, triplets.astype(str))
    d = {t: c for t, c in zip(triplets, counts)}
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in d:
            d[t] = 0
    return d


def region_bit_dist(filepath, year, brackets):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    observed = np.mean(vectors, axis=0)

    with open(filepath) as f:
        data = json.load(f)
    counts = np.array(data['bit_count'])[:60].reshape(-1, 15).sum(axis=0)
    pool = counts / (data['count'] * 4)
    return pool, observed


if __name__ == '__main__':
    import sys
    models_filepath = sys.argv[1]
    output = sys.argv[2]

    brackets = load_brackets()

    with open(models_filepath) as f:
        models = json.load(f)['models']

    for model in models:
        for year in range(2013, 2019):
            print('{} - {}'.format(model['modelName'], year))
            print('=' * 80)
            filepath = 'Experiments/10kTrials/Batch00/vectorStats_{}_{}.json'.format(model['modelName'], year)
            for name, triplet in all_triplets.items():
                pool = pool_dist(filepath, name)
                observed = observed_dist(year, name, brackets)

                df = pd.DataFrame.from_dict({'From Simulation': pool, 'Observed': observed})
                df = df / df.sum(axis=0)
                print('Dist triplet {}'.format(name))
                print(df)
                print()
                df.plot.bar(rot=0)
                plt.title('Triplet {} dist. comparison - {} - {}'.format(name, model['modelName'], year))
                plt.savefig(output + '/dist-{}-{}-{}.png'.format(model['modelName'], year, name))
                plt.close()
                # plt.show()

            pool, observed = region_bit_dist(filepath, year, brackets)
            df = pd.DataFrame.from_dict({'From Simulation': pool, 'Observed': observed})
            df.plot.bar(rot=0)
            plt.title('Dist of region bits - {} - {}'.format(model['modelName'], year))
            print('Dist of region bits')
            print(df)
            print()
            plt.savefig(output + '/dist-region-bits-{}-{}.png'.format(model['modelName'], year))
            plt.close()
            # plt.show()
