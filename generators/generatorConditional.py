#!/usr/bin/env python
import time
import json
import numpy as np
import os.path
import sys

from scoringUtils import getActualBracketVector
from scoringUtils import scoreFFFBracket, scoreBracket
from utils.runtimeSummary import RuntimeSummary
from samplingUtils import getChampion, getRunnerUp
from samplingUtils import getE8SeedBottom, getE8SeedTop
from samplingUtils import getF4SeedSplit, getF4SeedTogether
from numbers import Number


def load_ref_brackets(fmt='TTT'):
    with open("allBrackets{}.json".format(fmt)) as f:
        data = json.load(f)
        vectors = {
            int(bracket['bracket']['year']):
                np.array(list(bracket['bracket']['fullvector']), dtype=int)
            for bracket in data['brackets']}
    return vectors

probs = {}
all_triplets = {
    'E8_F4': {
        'bits': [12, 13, 14]
    },
    'S16_E8_1': {
        'bits': [8, 9, 12]
    },
    'S16_E8_2': {
        'bits': [10, 11, 13]
    },
    'R1_R2_1': {
        'bits': [0, 1, 8]
    },
    'R1_R2_2': {
        'bits': [2, 3, 9]
    },
    'R1_R2_3': {
        'bits': [4, 5, 10]
    },
    'R1_R2_4': {
        'bits': [6, 7, 11]
    },
    'NCG': {
        'bits': [60, 61, 62]
    },
    'R4_R5_1': {
        'bits': [14, 29, 60]
    },
    'R4_R5_2': {
        'bits': [44, 59, 61]
    }
}

all_paths = {
    'P_S1': {
        'bits': [0, 8, 12]
    },
    'P_S2': {
        'bits': [7, 11, 13]
    },
    'P_S3': {
        'bits': [5, 10, 13]
    },
    'P_S4': {
        'bits': [3, 9, 12]
    },
    'P_S5': {
        'bits': [2, 9, 12]
    },
    'P_S6': {
        'bits': [4, 10, 13]
    },
    'P_S7': {
        'bits': [6, 11, 13]
    },
    'P_S8': {
        'bits': [1, 8, 12]
    },
    'P_R2_1': {
        'bits': [8, 12, 14]
    },
    'P_R2_2': {
        'bits': [9, 12, 14]
    },
    'P_R2_3': {
        'bits': [10, 13, 14]
    },
    'P_R2_4': {
        'bits': [11, 13, 14]
    },
    'P_R4_R6_1': {
        'bits': [14, 60, 62]
    },
    'P_R4_R6_2': {
        'bits': [29, 60, 62]
    },
    'P_R4_R6_3': {
        'bits': [44, 61, 62]
    },
    'P_R4_R6_4': {
        'bits': [59, 61, 62]
    },
    'P_R3_R5_1': {
        'bits': [12, 14, 60]
    },
    'P_R3_R5_2': {
        'bits': [13, 14, 60]
    },
    'P_R3_R5_3': {
        'bits': [27, 29, 60]
    },
    'P_R3_R5_4': {
        'bits': [28, 29, 60]
    },
    'P_R3_R5_5': {
        'bits': [42, 44, 61]
    },
    'P_R3_R5_6': {
        'bits': [43, 44, 61]
    },
    'P_R3_R5_7': {
        'bits': [57, 59, 61]
    },
    'P_R3_R5_8': {
        'bits': [58, 59, 61]
    }
}

all_patterns = {
    'E8_F4': {
        'bits': np.array([12, 13, 14]),
        'section': 'triplets'
    },
    'S16_E8_1': {
        'bits': np.array([8, 9, 12]),
        'section': 'triplets'
    },
    'S16_E8_2': {
        'bits': np.array([10, 11, 13]),
        'section': 'triplets'
    },
    'R1_R2_1': {
        'bits': np.array([0, 1, 8]),
        'section': 'triplets'
    },
    'R1_R2_2': {
        'bits': np.array([2, 3, 9]),
        'section': 'triplets'
    },
    'R1_R2_3': {
        'bits': np.array([4, 5, 10]),
        'section': 'triplets'
    },
    'R1_R2_4': {
        'bits': np.array([6, 7, 11]),
        'section': 'triplets'
    },
    'NCG': {
        'bits': np.array([60, 61, 62]),
        'section': 'non-regional-triplets'
    },
    'R4_R5_1': {
        'bits': np.array([14, 29, 60]),
        'section': 'non-regional-triplets'
    },
    'R4_R5_2': {
        'bits': np.array([44, 59, 61]),
        'section': 'non-regional-triplets'
    },
    'P_S1': {
        'bits': np.array([0, 8, 12]),
        'section': 'paths'
    },
    'P_S2': {
        'bits': np.array([7, 11, 13]),
        'section': 'paths'
    },
    'P_S3': {
        'bits': np.array([5, 10, 13]),
        'section': 'paths'
    },
    'P_S4': {
        'bits': np.array([3, 9, 12]),
        'section': 'paths'
    },
    'P_S5': {
        'bits': np.array([2, 9, 12]),
        'section': 'paths'
    },
    'P_S6': {
        'bits': np.array([4, 10, 13]),
        'section': 'paths'
    },
    'P_S7': {
        'bits': np.array([6, 11, 13]),
        'section': 'paths'
    },
    'P_S8': {
        'bits': np.array([1, 8, 12]),
        'section': 'paths'
    },
    'P_R2_1': {
        'bits': np.array([8, 12, 14]),
        'section': 'paths'
    },
    'P_R2_2': {
        'bits': np.array([9, 12, 14]),
        'section': 'paths'
    },
    'P_R2_3': {
        'bits': np.array([10, 13, 14]),
        'section': 'paths'
    },
    'P_R2_4': {
        'bits': np.array([11, 13, 14]),
        'section': 'paths'
    },
    'P_R4_R6_1': {
        'bits': np.array([14, 60, 62]),
        'section': 'non-regional-paths'
    },
    'P_R4_R6_2': {
        'bits': np.array([29, 60, 62]),
        'section': 'non-regional-paths'
    },
    'P_R4_R6_3': {
        'bits': np.array([44, 61, 62]),
        'section': 'non-regional-paths'
    },
    'P_R4_R6_4': {
        'bits': np.array([59, 61, 62]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_1': {
        'bits': np.array([12, 14, 60]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_2': {
        'bits': np.array([13, 14, 60]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_3': {
        'bits': np.array([27, 29, 60]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_4': {
        'bits': np.array([28, 29, 60]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_5': {
        'bits': np.array([42, 44, 61]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_6': {
        'bits': np.array([43, 44, 61]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_7': {
        'bits': np.array([57, 59, 61]),
        'section': 'non-regional-paths'
    },
    'P_R3_R5_8': {
        'bits': np.array([58, 59, 61]),
        'section': 'non-regional-paths'
    }
}

perturbed_ps = {
    '25_1985': [
        1.0,
        0.4229103348680731,
        0.8686253918935398,
        1.0,
        0.8123704992250859,
        1.0,
        0.7077991428824237,
        1.0,
    ],
    '26_1985': [
        1.0,
        0.426407803982116,
        0.8185896601561352,
        1.0,
        0.7023989582786588,
        1.0,
        0.6732989797077819,
        1.0,
    ],
    '27_1985': [
        1.0,
        0.40952672906312887,
        0.7854648378949232,
        1.0,
        0.7606474009097475,
        1.0,
        0.6539118110780404,
        1.0,
    ],
    '28_1985': [
        1.0,
        0.47544408691902595,
        0.7559601878920554,
        0.9487586660088168,
        0.7109945653089622,
        1.0,
        0.6877373278227099,
        1.0,
    ],
    '30_1985': [
        1.0,
        0.6897381544845143,
        0.6574444680210119,
        0.9879908588573346,
        0.6011024159891302,
        0.9211618267386112,
        0.8727621098790732,
        1.0,
    ],
    '31_1985': [
        0.9926470588235294,
        0.5,
        0.6544117647058824,
        0.7941176470588235,
        0.625,
        0.8455882352941176,
        0.5069725126093771,
        0.9411764705882353,
    ],
    '29_2002': [
        1.0,
        0.7294731173878907,
        0.6258223531539129,
        0.9164091224773702,
        0.5715239568648995,
        1.0,
        0.7069538395226441,
        1.0,
    ],
    '28_2002': [
        1.0,
        0.6810812994722260,
        0.6782287991306060,
        1.0,
        0.6065824706992460,
        0.9954042914277010,
        0.6102388677298300,
        1.0
    ],
    '30_2002': [
        1.0,
        0.7074624431810751,
        0.7470166821001302,
        1.0,
        0.5968548284479815,
        0.9691087987086129,
        0.8588856269144272,
        1.0
    ]
}

REGIONAL_FIXED_BITS = {
    1: np.array([1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1]),
    2: np.array([-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 0, -1, 0, 0]),
    3: np.array([-1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
    4: np.array([-1, -1, -1, 1, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, 1]),
    5: np.array([-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 0, -1, 1]),
    6: np.array([-1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 0]),
    7: np.array([-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 0, 0]),
    8: np.array([-1, 1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 1, -1, 1]),
    9: np.array([-1, 0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 1, -1, 1]),
    10: np.array([-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 1, -1, 0, 0]),
    11: np.array([-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 1, -1, -1, 1, 0]),
    12: np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, 1, -1, -1, 0, -1, 1]),
    13: np.array([-1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0, -1, 1]),
    14: np.array([-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, 1, 0]),
    15: np.array([-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, -1, 0, 0]),
    16: np.array([0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1])
}

CONDITIONALS = {
    # ((bit1, val1), (bit2, val2)): P(bit3=1|bit1=val1, bit2=val2)
}

TOP_SEEDS = [1, 16, 8, 9, 5, 12, 4, 13]

def fill_all_pattern_probs():
    global all_patterns, CONDITIONALS
    names = list(all_patterns.keys())
    for year in range(2013, 2020):
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        vectors = vectors[:, :60].reshape(-1, 15)
        for name in names:
            triplet = all_patterns[name]['bits']
            if np.greater(triplet, 14).any():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_patterns[name][year] = {
                'p': cdf,
                'triplets': triplets
            }
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        for name in names:
            triplet = all_patterns[name]['bits']
            if np.greater(14, triplet).all():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_patterns[name][year] = {
                'p': cdf,
                'triplets': triplets
            }

    names = list(all_patterns.keys())
    combs = [[0, 1], [0, 2], [1, 2]]
    values = [[0, 0], [0, 1], [1, 0], [1, 1]]

    from itertools import product

    for comb in combs:
        for val in values:
            key = tuple(zip(comb, val))
            CONDITIONALS[key] = {}
            for year in range(2013, 2020):
                CONDITIONALS[key][year] = {}
                vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
                region_vectors = vectors[:, :60].reshape(-1, 15)
                for name in names:
                    triplet = all_patterns[name]['bits']
                    # its a triplet
                    if all_patterns[name]['section'].startswith('non-regional'):
                        data = vectors[:, triplet]
                    else:
                        data = region_vectors[:, triplet]
                    data = data[(data[:, comb] == val).all(axis=1), :]
                    data = np.delete(data, comb, axis=1)
                    if len(data) == 0:
                        data = [list(x) for x in product([0, 1], repeat=data.shape[1])]
                    triplets, counts = np.unique(data, axis=0, return_counts=True)
                    cdf = [1. * counts[:i].sum() / counts.sum()
                           for i in range(len(counts) + 1)]
                    CONDITIONALS[key][year][name] = {
                        'p': cdf,
                        'triplets': triplets
                    }

    combs = [[0, 1], [1, 2], [0], [1], [2]]
    all_values = [
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [[0], [1]],
        [[0], [1]],
        [[0], [1]]
    ]
    for comb, values in zip(combs, all_values):
        for val in values:
            key = tuple(zip(comb, val))
            if key not in CONDITIONALS:
                CONDITIONALS[key] = {}
            for year in range(2013, 2020):
                if year not in CONDITIONALS[key]:
                    CONDITIONALS[key][year] = {}
                vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
                region_vectors = vectors[:, :60].reshape(-1, 15)
                for name in names:
                    triplet = all_patterns[name]['bits']
                    if not name.startswith('P_'):
                        continue
                    # its a path
                    if all_patterns[name]['section'].startswith('non-regional'):
                        data = vectors[:, triplet]
                    else:
                        data = region_vectors[:, triplet]
                    data = data[(data[:, comb] == val).all(axis=1), :]
                    data = np.delete(data, comb, axis=1)
                    if len(data) == 0:
                        data = [list(x) for x in product([0, 1], repeat=data.shape[1])]
                    triplets, counts = np.unique(data, axis=0, return_counts=True)
                    cdf = [1. * counts[:i].sum() / counts.sum()
                           for i in range(len(counts) + 1)]
                    CONDITIONALS[key][year][name] = {
                        'p': cdf,
                        'triplets': triplets
                    }


def fill_triplet_probs():
    global all_triplets
    names = list(all_triplets.keys())
    for year in range(2013, 2020):
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        vectors = vectors[:, :60].reshape(-1, 15)
        for name in names:
            triplet = all_triplets[name]['bits']
            if np.greater(triplet, 14).any():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_triplets[name][year] = {
                'p': cdf,
                'triplets': triplets
            }
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        for name in names:
            triplet = all_triplets[name]['bits']
            if np.greater(14, triplet).all():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_triplets[name][year] = {
                'p': cdf,
                'triplets': triplets
            }


def fill_path_probs():
    global all_paths
    names = list(all_paths.keys())
    for year in range(2013, 2020):
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        vectors = vectors[:, :60].reshape(-1, 15)
        for name in names:
            triplet = all_paths[name]['bits']
            if np.greater(triplet, 14).any():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_paths[name][year] = {
                'p': cdf,
                'triplets': triplets
            }
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        for name in names:
            triplet = all_paths[name]['bits']
            if np.greater(14, triplet).all():
                continue
            triplets, counts = np.unique(vectors[:, triplet], axis=0, return_counts=True)
            cdf = [1. * counts[:i].sum() / counts.sum()
                   for i in range(len(counts) + 1)]
            all_paths[name][year] = {
                'p': cdf,
                'triplets': triplets
            }


def getP(model, year, bit_id):
    if model.get('annealing_model') is not None and bit_id < 60 and (bit_id % 15) < 8:
        return perturbed_ps[model.get('annealing_model')][bit_id % 15]
    base_p = probs[year][bit_id]
    if model.get('perturbation') and isinstance(model['perturbation'], Number):
        if model.get('perturbationType') == 'fixed':
            p = base_p + np.random.uniform(-model['perturbation'], model['perturbation'])
        else:
            p = np.random.uniform((1 - model['perturbation']) * base_p, (1 + model['perturbation']) * base_p)
    else:
        p = base_p
    return np.clip(p, 0., 1.)


def getValues(bracket, year, pattern_key):
    n = np.random.rand()
    for i in range(8):
        if n > all_patterns[pattern_key][year]['p'][i] and n < all_patterns[pattern_key][year]['p'][i + 1]:
            return all_patterns[pattern_key][year]['triplets'][i]


def fixRegionalBits(winner):
    return REGIONAL_FIXED_BITS[winner].copy()


def testRegionalBits():
    for expected, bits in REGIONAL_FIXED_BITS.items():
        seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        while len(seeds) > 1:
            n_games = len(seeds) // 2
            new_seeds = []
            for g in range(n_games):
                if bits[g] == -1:
                    new_seeds.append(-1)
                elif bits[g] == 1:
                    new_seeds.append(seeds[g * 2])
                else:
                    new_seeds.append(seeds[g * 2 + 1])
            seeds = new_seeds
            bits = bits[n_games:]
        winner = seeds[0]
        assert winner == expected


def fixBitsFromNCG(bracket, champion, runnerUp):
    if bracket[-1] == 1:
        if bracket[-3] == 1:
            championRegion = 0
        else:
            championRegion = 1
        if bracket[-2] == 1:
            ruRegion = 2
        else:
            ruRegion = 3
    else:
        if bracket[-2] == 1:
            championRegion = 2
        else:
            championRegion = 3
        if bracket[-3] == 1:
            ruRegion = 0
        else:
            ruRegion = 1
    championOffset = championRegion * 15
    ruOffset = ruRegion * 15
    assert ruRegion != championRegion
    bracket[championOffset:championOffset + 15] = fixRegionalBits(champion)
    bracket[ruOffset:ruOffset + 15] = fixRegionalBits(runnerUp)
    return bracket, championRegion, ruRegion

def genBracketWithoutEndModel(model, year):
    n = np.random.rand(63)
    p = [getP(model, year, i) for i in range(63)]
    bracket = (n < p).astype(int)
    for region in range(4):
        for t in model.get('triplets', []):
            n = np.random.rand()
            for i in range(8):
                if n > all_triplets[t][year]['p'][i] and n < \
                        all_triplets[t][year]['p'][i + 1]:
                    bracket[15 * region + np.array(all_triplets[t]['bits'])] = \
                    all_triplets[t][year]['triplets'][i]
                    break

        for t in model.get('paths', []):
            n = np.random.rand()
            for i in range(8):
                if n > all_paths[t][year]['p'][i] and n < \
                        all_paths[t][year]['p'][i + 1]:
                    bracket[15 * region + np.array(all_paths[t]['bits'])] = \
                    all_paths[t][year]['triplets'][i]
                    break

    # non-regional patterns
    for t in model.get('non-regional-triplets', []):
        n = np.random.rand()
        for i in range(8):
            if n > all_triplets[t][year]['p'][i] and n < \
                    all_triplets[t][year]['p'][i + 1]:
                bracket[np.array(all_triplets[t]['bits'])] = \
                all_triplets[t][year]['triplets'][i]
                break

    for t in model.get('non-regional-paths', []):
        n = np.random.rand()
        for i in range(8):
            if n > all_paths[t][year]['p'][i] and n < all_paths[t][year]['p'][i + 1]:
                bracket[np.array(all_paths[t]['bits'])] = \
                all_paths[t][year]['triplets'][i]
                break
    return bracket


def getE8Bracket(model, year):
    bracket = np.repeat(-1, 63)
    for region in range(4):
        s1 = getE8SeedBottom(year, model)
        s2 = getE8SeedTop(year, model)
        region_bracket = fixRegionalBits(s1)
        region_bracket_2 = fixRegionalBits(s2)
        region_bracket[region_bracket_2 != -1] = region_bracket_2[region_bracket_2 != -1]
        region_bracket[-1] = -1
        bracket[region * 15:region * 15 + 15] = region_bracket
    return fillEmptySpaces(bracket, model, year)

def getF4ABracket(model, year):
    bracket = np.repeat(-1, 63)
    for region in range(4):
        winner = getF4SeedTogether(year, model)
        bracket[region * 15:region * 15 + 15] = fixRegionalBits(winner)
    return fillEmptySpaces(bracket, model, year)

def getF4BBracket(model, year):
    bracket = np.repeat(-1, 63)
    for region in range(4):
        winner = getF4SeedSplit(year, model)
        bracket[region * 15:region * 15 + 15] = fixRegionalBits(winner)
    return fillEmptySpaces(bracket, model, year)

def genNCGBracket(model, year):
    bracket = np.repeat(-1, 63)
    champion = getChampion(year, model)
    runnerUp = getRunnerUp(year, model)
    print(champion, runnerUp)
    ncg_triplet = getValues(bracket, year, 'NCG')
    bracket[[60, 61, 62]] = ncg_triplet
    bracket, _, _ = fixBitsFromNCG(bracket, champion, runnerUp)
    return fillEmptySpaces(bracket, model, year)


def getCombinedEndModelBracket(model, year):
    bracket = np.repeat(-1, 63)
    champion = getChampion(year, model)
    runnerUp = getRunnerUp(year, model)
    ncg_triplet = getValues(bracket, year, 'NCG')
    bracket[[60, 61, 62]] = ncg_triplet
    bracket, _, _ = fixBitsFromNCG(bracket, champion, runnerUp)

    f4_seeds = [getF4SeedSplit(year, model) for _ in range(4)]
    for region in range(4):
        if bracket[region * 15 + 14] == -1:
            bracket[region * 15:region * 15 + 15] = fixRegionalBits(f4_seeds[region])
    assert np.all(bracket[[14, 29, 44, 59]] != -1)
    return fillEmptySpaces(bracket, model, year)


def getNCG_E8ModelBracket(model, year):
    bracket = np.repeat(-1, 63)
    champion = getChampion(year, model)
    runnerUp = getRunnerUp(year, model)
    ncg_triplet = getValues(bracket, year, 'NCG')
    bracket[[60, 61, 62]] = ncg_triplet
    bracket, championRegion, ruRegion = fixBitsFromNCG(bracket, champion, runnerUp)

    f4_seeds = [getF4SeedSplit(year, model) for _ in range(4)]
    for region in range(4):
        if bracket[region * 15 + 14] == -1:
            bits = fixRegionalBits(f4_seeds[region])
            if f4_seeds[region] in TOP_SEEDS:
                other_seed = getE8SeedBottom(year, model)
            else:
                other_seed = getE8SeedTop(year, model)
            bits[bits == -1] = fixRegionalBits(other_seed)[bits == -1]
            bracket[region * 15:region * 15 + 15] = bits
        else:
            bits = bracket[region * 15:region * 15 + 15]
            if region == championRegion:
                if champion in TOP_SEEDS:
                    other_seed = getE8SeedBottom(year, model)
                else:
                    other_seed = getE8SeedTop(year, model)
            elif region == ruRegion:
                if runnerUp in TOP_SEEDS:
                    other_seed = getE8SeedBottom(year, model)
                else:
                    other_seed = getE8SeedTop(year, model)
            bits[bits == -1] = fixRegionalBits(other_seed)[bits == -1]
            bracket[region * 15:region * 15 + 15] = bits

    assert np.all(bracket[[60, 61, 62]] != -1)
    assert np.all(bracket[[14, 29, 44, 59]] != -1)
    assert np.all(bracket[[12, 13, 27, 28, 42, 43, 57, 58]] != -1)
    assert np.count_nonzero(bracket != -1) == 31
    return fillEmptySpaces(bracket, model, year)


def fillWithPowerModel(bracket, model, year):
    return bracket


def fillEmptySpaces(bracket, model, year):
    for key in ['non-regional-paths', 'non-regional-triplets']:
        for t in model.get(key, []):
            n = np.random.rand()
            selector = all_patterns[t]['bits']
            if np.all(bracket[selector] == -1):
                for i in range(8):
                    if n > all_patterns[t][year]['p'][i] and n < all_patterns[t][year]['p'][i + 1]:
                        bracket[selector] = all_patterns[t][year]['triplets'][i]
                        break
            elif np.all(bracket[selector] != -1):
                continue
            else:
                bits = np.array([0, 1, 2])[bracket[selector] != -1]
                pending = np.array([0, 1, 2])[bracket[selector] == -1]
                values = bracket[selector][bits]
                cond_table_key = tuple(zip(bits, values))
                try:
                    cdf = CONDITIONALS[cond_table_key][year][t]
                except:
                    import pdb; pdb.set_trace()
                for i in range(len(cdf)):
                    if n > cdf['p'][i] and n < cdf['p'][i + 1]:
                        bracket[pending] = cdf['triplets'][i]

    for key in ['paths', 'triplets']:
        for t in model.get(key, []):
            for region in range(4):
                n = np.random.rand()
                selector = all_patterns[t]['bits'] + region * 15
                if np.all(bracket[selector] == -1):
                    for i in range(8):
                        if n > all_patterns[t][year]['p'][i] and n < all_patterns[t][year]['p'][i + 1]:
                            bracket[selector] = all_patterns[t][year]['triplets'][i]
                            break
                elif np.all(bracket[selector] != -1):
                    continue
                else:
                    bits = np.array([0, 1, 2])[bracket[selector] != -1]
                    pending = np.array([0, 1, 2])[bracket[selector] == -1]
                    values = bracket[selector][bits]
                    cond_table_key = tuple(zip(bits, values))
                    cdf = CONDITIONALS[cond_table_key][year][t]
                    for i in range(len(cdf)):
                        if n > cdf['p'][i] and n < cdf['p'][i + 1]:
                            bracket[pending] = cdf['triplets'][i]

    if model.get('filler') == 'power':
        return fillWithPowerModel(bracket, model, year)
    for bit in range(63):
        if bracket[bit] == -1:
            n = np.random.rand()
            bracket[bit] = 1 if n < getP(model, year, bit) else 0
    return bracket


def generateBracket(model, year):
    if model.get('endModel') is None:
        return genBracketWithoutEndModel(model, year)
    elif model['endModel'] == 'NCG':
        return genNCGBracket(model, year)
    elif model['endModel'] == 'F4_A':
        return getF4ABracket(model, year)
    elif model['endModel'] == 'F4_B':
        return getF4BBracket(model, year)
    elif model['endModel'] == 'E8':
        return getE8Bracket(model, year)
    elif model['endModel'] == 'combined':
        return getCombinedEndModelBracket(model, year)
    elif model['endModel'] == 'NCG_E8':
        return getNCG_E8ModelBracket(model, year)
    else:
        raise Exception('Not implemented yet')


def performExperiments(numTrials, year, batchNumber, model):
    summarizer = RuntimeSummary(model)
    correctVector = getActualBracketVector(year)

    scores = [None] * numTrials
    scoreMethod = scoreFFFBracket if model.get('format') == 'FFF' else scoreBracket

    for n in range(numTrials):
        newBracketVector = generateBracket(model, year)
        summarizer.analyze_bracket(newBracketVector)
        newBracketScore = scoreMethod(newBracketVector, correctVector)
        scores[n] = newBracketScore[0]

    bracketListDict = {'year': year, 'actualBracket': ''.join(str(bit) for bit in correctVector), 'scores': scores}

    if numTrials < 1000:
        folderName = 'Experiments/{0}Trials'.format(numTrials)
    else:
        folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))
    batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)

    outputFilename = '{2}/generatedScores_{0}_{1}.json'.format(model['modelName'], year, batchFolderName)
    summaryFilename = '{2}/vectorStats_{0}_{1}.json'.format(model['modelName'], year, batchFolderName)
    with open(outputFilename, 'w') as outputFile:
        outputFile.write(json.dumps(bracketListDict))
    summarizer.to_json(summaryFilename)


######################################################################
# This script runs experiments with the given models,
# number of trials, and number of batches for 2013 through 2018.
######################################################################

# Load models
if len(sys.argv) > 3:
    modelFilename = sys.argv[3]
else:
    modelFilename = 'models.json'
with open(modelFilename, 'r') as modelFile:
    modelsDataJson = modelFile.read().replace('\n', '')

modelsDict = json.loads(modelsDataJson)
modelsList = modelsDict['models']

numTrials = int(sys.argv[1])
numBatches = int(sys.argv[2])

if len(sys.argv) == 5:
    modelIndex = int(sys.argv[4])
else:
    modelIndex = -1

testRegionalBits()

for modelId, modelDict in enumerate(modelsList):
    if modelIndex != -1 and modelIndex != modelId:
        continue
    modelName = modelDict['modelName']

    if modelDict.get('generator') != 'conditional_generator':
        continue

    all_brackets = load_ref_brackets(modelDict.get('format', 'TTT'))
    # calculate bitwise MLE probs
    for year in range(2013, 2020):
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        probs[year] = np.mean(vectors, axis=0)

    fill_all_pattern_probs()
    fill_triplet_probs()
    fill_path_probs()

    print '{0:<8s}: {1}'.format(modelName, time.strftime("%Y-%m-%d %H:%M"))

    for year in range(2013, 2020):
        print '\t {0}: {1}'.format(year, time.strftime("%Y-%m-%d %H:%M"))
        for batchNumber in range(numBatches):
            print '\t\t {0}: {1}'.format(batchNumber, time.strftime("%Y-%m-%d %H:%M"))
            if numTrials < 1000:
                folderName = 'Experiments/{0}Trials'.format(numTrials)
            else:
                folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))

            if not os.path.exists(folderName):
                os.makedirs(folderName)

            batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)
            if not os.path.exists(batchFolderName):
                os.makedirs(batchFolderName)

            performExperiments(numTrials, year, batchNumber, modelDict)

