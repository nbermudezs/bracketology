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
from itertools import product


plt.style.use('seaborn-white')
sns.set_palette('colorblind')


all_triplets = {
    'E8_F4': {
        'bits': [12, 13, 14],
        'section': 'triplets'
    },
    'S16_E8_1': {
        'bits': [8, 9, 12],
        'section': 'triplets'
    },
    'S16_E8_2': {
        'bits': [10, 11, 13],
        'section': 'triplets'
    },
    'R1_R2_1': {
        'bits': [0, 1, 8],
        'section': 'triplets'
    },
    'R1_R2_2': {
        'bits': [2, 3, 9],
        'section': 'triplets'
    },
    'R1_R2_3': {
        'bits': [4, 5, 10],
        'section': 'triplets'
    },
    'R1_R2_4': {
        'bits': [6, 7, 11],
        'section': 'triplets'
    },
    'NCG': {
        'bits': [60, 61, 62],
        'section': 'non-regional-triplets'
    },
    'R4_R5_1': {
        'bits': [14, 29, 60],
        'section': 'non-regional-triplets'
    },
    'R4_R5_2': {
        'bits': [44, 59, 61],
        'section': 'non-regional-triplets'
    },
    'P_S1': {
        'bits': [0, 8, 12],
        'section': 'paths'
    },
    'P_S2': {
        'bits': [7, 11, 13],
        'section': 'paths'
    },
    'P_S3': {
        'bits': [5, 10, 13],
        'section': 'paths'
    },
    'P_S4': {
        'bits': [3, 9, 12],
        'section': 'paths'
    },
    'P_S5': {
        'bits': [2, 9, 12],
        'section': 'paths'
    },
    'P_S6': {
        'bits': [4, 10, 13],
        'section': 'paths'
    },
    'P_S7': {
        'bits': [6, 11, 13],
        'section': 'paths'
    },
    'P_S8': {
        'bits': [1, 8, 12],
        'section': 'paths'
    },
    'P_R2_1': {
        'bits': [8, 12, 14],
        'section': 'paths'
    },
    'P_R2_2': {
        'bits': [9, 12, 14],
        'section': 'paths'
    },
    'P_R2_3': {
        'bits': [10, 13, 14],
        'section': 'paths'
    },
    'P_R2_4': {
        'bits': [11, 13, 14],
        'section': 'paths'
    },
    'P_R4_R6_1': {
        'bits': [14, 60, 62],
        'section': 'non-regional-paths'
    },
    'P_R4_R6_2': {
        'bits': [29, 60, 62],
        'section': 'non-regional-paths'
    },
    'P_R4_R6_3': {
        'bits': [44, 61, 62],
        'section': 'non-regional-paths'
    },
    'P_R4_R6_4': {
        'bits': [59, 61, 62],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_1': {
        'bits': [12, 14, 60],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_2': {
        'bits': [13, 14, 60],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_3': {
        'bits': [27, 29, 60],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_4': {
        'bits': [28, 29, 60],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_5': {
        'bits': [42, 44, 61],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_6': {
        'bits': [43, 44, 61],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_7': {
        'bits': [57, 59, 61],
        'section': 'non-regional-paths'
    },
    'P_R3_R5_8': {
        'bits': [58, 59, 61],
        'section': 'non-regional-paths'
    }
}


def load_brackets(fmt='TTT'):
    with open("allBrackets{}.json".format(fmt)) as f:
        data = json.load(f)['brackets']
    # with open("allBracketsTTT_permuted_0.json") as f:
    #     data2 = json.load(f)['brackets']
    # return data + data2
    return data


def permutations(brackets):
    result = []
    for comb in product([0, 1], repeat=3):
        clone = []
        for bracket in brackets:
            full_vector = bracket['bracket']['fullvector']
            new_vector = np.array(list(full_vector), dtype=int)
            if comb[0] == 1:
                new_vector = np.concatenate((new_vector[15:30], new_vector[:15], new_vector[30:]))
                new_vector[60] = 1 - new_vector[60]
            if comb[1] == 1:
                new_vector = np.concatenate((new_vector[:30], new_vector[45:60], new_vector[30:45], new_vector[60:]))
                new_vector[61] = 1 - new_vector[61]
            if comb[2] == 1:
                new_vector = np.concatenate((new_vector[30:60], new_vector[:30], new_vector[60:]))
                new_vector[62] = 1 - new_vector[62]
            clone.append({
                'bracket': {
                    'year': bracket['bracket']['year'],
                    'fullvector': ''.join(new_vector.astype(str))
                }
            })
        print(comb)
        result.append({'brackets': clone, 'permutation': comb})
    return result


def observed_dist(year, key, brackets):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    triplets, counts = np.unique(vectors[:, all_triplets[key]['bits']], axis=0, return_counts=True)
    triplets = np.apply_along_axis(''.join, 1, triplets.astype(str))
    d = {t: c for t, c in zip(triplets, counts)}
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in d:
            d[t] = 0
    return d


def comb_to_title(comb):
    if np.all(np.equal(comb, 1)):
        return 'All swapped'
    if np.all(np.equal(comb, 0)):
        return 'Original'
    names = ['R1 and R2', 'R3 and R4', 'NCG']
    selected = [name for i, name in enumerate(names) if comb[i] == 1]
    return ', '.join(selected)


if __name__ == '__main__':
    import os
    brackets = load_brackets('TTT')
    permuted = permutations(brackets)
    combs = list(product([0, 1], repeat=3))
    for i, permutation in enumerate(permuted[1:]):
        with open('allBracketsTTT_permuted_{}.json'.format(i), 'w') as f:
            json.dump(permutation, f, indent=4)

    for triplet in ['R4_R5_1', 'R4_R5_2', 'NCG']:
        fig, ax = plt.subplots(2, 4, figsize=(13, 7), sharey=True)
        for idx, (comb, permutation) in enumerate(zip(combs, permuted)):
            row = idx // 4
            col = idx % 4
            permuted = observed_dist(2019, triplet, permutation['brackets'])
            order = sorted(permuted, key=lambda x: x[1])
            title = comb_to_title(comb)
            pd.DataFrame.from_dict({title: permuted}).sort_values(by=title).plot.bar(rot=0, ax=ax[row, col], color='black')
            # df.sort_values(by='Permutation').plot.bar(rot=0, sharex=False, ylim=(0, 10))
        fig.tight_layout()

        if not os.path.exists('isomorphism'):
            os.makedirs('isomorphism')
        
        plt.savefig('isomorphism/{}_comparison.png'.format(triplet), figsize=(13, 7))
        plt.close()
