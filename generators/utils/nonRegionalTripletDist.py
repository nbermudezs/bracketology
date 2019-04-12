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
import rpy2
import rpy2.robjects as robjects
import seaborn as sns
from scipy.stats import chisquare


plt.style.use('seaborn-white')
sns.set_palette('colorblind')


triplets = {
    'NCG': [60, 61, 62],
    'R4_R5_1': [14, 29, 60],
    'R4_R5_2': [44, 59, 61]
}

paths = {
    'R4_R6_1': [14, 60, 62],
    'R4_R6_2': [29, 60, 62],
    'R4_R6_3': [44, 61, 62],
    'R4_R6_4': [59, 61, 62]
}


def load_brackets(fmt='TTT'):
    with open('allBrackets{}.json'.format(fmt)) as f:
        brackets = json.load(f)['brackets']
        return brackets


def triplet_values(brackets, year):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    return {name: vectors[:, bits] for name, bits in triplets.items()}


def path_values(brackets, year):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    return {name: vectors[:, bits] for name, bits in paths.items()}


def observed_dist(triplets):
    triplets, counts = np.unique(triplets, axis=0, return_counts=True)
    triplet_labels = np.apply_along_axis(''.join, 1, triplets.astype(str))
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in triplet_labels:
            triplet_labels = np.append(triplet_labels, t)
            counts = np.append(counts, 0)
    return {l: c for l, c in zip(triplet_labels, counts)}


def expected_dist(brackets, year, bits):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    triplets = vectors[:, bits]
    p_1 = np.mean(triplets, axis=0)
    p_0 = 1 - p_1
    p = [p_0, p_1]
    result = {}
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        values = [int(x) for x in list(t)]
        triplet_p = np.prod([p[values[i]][i] for i in range(3)])
        result[t] = (year - 1985) * triplet_p
    return result


def uniformity_check(observed, expected):
    chi, p = chisquare(list(observed.values()), expected)
    print('Uniformity chi-square test p-value', p)

    values = list(observed.values())
    keys = list(observed.keys())
    arr = 'array(c{}, dim=c(2, 2, 2))'.format(
        tuple(np.array(values)[np.argsort(keys)].astype(int).tolist()))
    res = robjects.r(
        'library(hypergea); hypergeom.test(' + arr + ")['p.value']")
    p_value = np.array(res[0])[0]
    print('Uniformity Fisher exact test p-value', p_value)


def plot_dist(brackets, year, values, bits, name, prefix):
    observed = observed_dist(values)
    expected = expected_dist(brackets, year, bits)
    data = {'Observed': observed, 'Expected (ind)': expected}
    df = pd.DataFrame.from_dict(data)
    df.plot.bar(rot=0)
    # plt.show()
    plt.title('3-bit path value distribution - {}'.format(name))
    plt.savefig('DistPlots/TTT/{}-{}.png'.format(prefix, name))
    plt.cla()
    plt.clf()
    values = list(observed.values())
    keys = list(observed.keys())
    arr = 'array(c{}, dim=c(2, 2, 2))'.format(
        tuple(np.array(values)[np.argsort(keys)].astype(int).tolist()))
    res = robjects.r('library(hypergea); hypergeom.test(' + arr + ")['p.value']")
    p_value = np.array(res[0])[0]
    print('Independence Fisher exact test p-value', p_value)

    uniformity_check(observed, np.repeat((year - 1985) / 8, 8))
    print()
    # print('m = array(c{}, dim=c(2, 2, 2))'.format(tuple(np.array(list(observed.values()))[np.argsort(observed.keys())].astype(int).tolist())))


if __name__ == '__main__':
    brackets = load_brackets()
    year = 2019

    paths_data = path_values(brackets, year)
    for name, values in paths_data.items():
        print('path {}'.format(name))
        plot_dist(brackets, year, values, paths[name], name, prefix='3bit_path')

    triplets_data = triplet_values(brackets, year)
    for name, values in triplets_data.items():
        print('triplet {}'.format(name))
        plot_dist(brackets, year, values, triplets[name], name, prefix='triplet')
