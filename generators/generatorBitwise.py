#!/usr/bin/env python
import time
import json
import numpy as np
import os.path
import sys

from scoringUtils import getActualBracketVector
from scoringUtils import scoreFFFBracket, scoreBracket
from utils.runtimeSummary import RuntimeSummary


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

def fill_triplet_probs():
    global all_triplets
    names = list(all_triplets.keys())
    for year in range(2013, 2019):
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
    for year in range(2013, 2019):
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
    if model.get('perturbation'):
        if model.get('perturbationType') == 'fixed':
            p = base_p + np.random.uniform(-model['perturbation'], model['perturbation'])
        else:
            p = np.random.uniform((1 - model['perturbation']) * base_p, (1 + model['perturbation']) * base_p)
    else:
        p = base_p
    return np.clip(p, 0., 1.)


def generateBracket(model, year):
    if model.get('endModel') is None:
        n = np.random.rand(63)
        p = [getP(model, year, i) for i in range(63)]
        bracket = (n < p).astype(int)
        for region in range(4):
            for t in model.get('triplets', []):
                n = np.random.rand()
                for i in range(8):
                    if n > all_triplets[t][year]['p'][i] and n < all_triplets[t][year]['p'][i+1]:
                        bracket[15 * region + np.array(all_triplets[t]['bits'])]= all_triplets[t][year]['triplets'][i]
                        break

            for t in model.get('paths', []):
                n = np.random.rand()
                for i in range(8):
                    if n > all_paths[t][year]['p'][i] and n < all_paths[t][year]['p'][i + 1]:
                        bracket[15 * region + np.array(all_paths[t]['bits'])] = all_paths[t][year]['triplets'][i]
                        break

        # non-regional patterns
        for t in model.get('non-regional-triplets', []):
            n = np.random.rand()
            for i in range(8):
                if n > all_triplets[t][year]['p'][i] and n < all_triplets[t][year]['p'][i + 1]:
                    bracket[np.array(all_triplets[t]['bits'])] = all_triplets[t][year]['triplets'][i]
                    break

        for t in model.get('non-regional-paths', []):
            n = np.random.rand()
            for i in range(8):
                if n > all_paths[t][year]['p'][i] and n < all_paths[t][year]['p'][i + 1]:
                    bracket[np.array(all_paths[t]['bits'])] = all_paths[t][year]['triplets'][i]
                    break
        return bracket
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

for modelId, modelDict in enumerate(modelsList):
    if modelIndex != -1 and modelIndex != modelId:
        continue
    modelName = modelDict['modelName']

    all_brackets = load_ref_brackets(modelDict.get('format', 'TTT'))
    # calculate bitwise MLE probs
    for year in range(2013, 2019):
        vectors = np.vstack([v for y, v in all_brackets.items() if y < year])
        probs[year] = np.mean(vectors, axis=0)

    fill_triplet_probs()
    fill_path_probs()

    print '{0:<8s}: {1}'.format(modelName, time.strftime("%Y-%m-%d %H:%M"))

    for year in range(2013, 2019):
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

