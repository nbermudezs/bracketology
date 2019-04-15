#!/usr/bin/env python
import time
import json
import numpy as np
import pandas as pd
import os.path
import random
import sys
from math import log, ceil, floor

from collections import defaultdict
from samplingUtils import getE8SeedBottom, getE8SeedTop
from samplingUtils import getF4SeedSplit, getF4SeedTogether
from samplingUtils import getChampion, getRunnerUp

from scoringUtils import getActualBracketVector
from scoringUtils import scoreBracket


######################################################################
# Author:
#     Ian Ludden
#
# Date created:
#     15 Apr 2018
#
# Last modified:
#     18 Dec 2018
#
# This general version handles all parameters previously implemented
# separately in runExperimentsFixedAlpha.py,
# runExperimentsSampleF4.py, and runExperimentsSampleE8.py.
#
# Specifically, this version supports:
# - "Forward" Power Model
# - "Reverse" Power Model (generate champ and runner-up, then forward)
# - "Reverse" Power Model with F4 (also generate other two F4 seeds)
# - F4 Model 1, where F4 seeds are generated using "Model 1,"
#   then power model is applied to games before and after
# - F4 Model 2, where F4 seeds are generated using "Model 2,"
#   then power model is applied to games before and after
# - E8 Model, where E8 seeds are generated,
#   then power model is applied to games before and after
#
# Also, the Round 1 alpha values are optionally grouped as:
# (1, 16) alone
# (2, 15) alone
# (3, 14), (4, 13)
# (5, 12), (6, 11), (7, 10)
# (8, 9) alone and fixed at 0.5 probability (alpha = 0)
#
# By default, all weighted alpha values are computed using the
# standard weighting (multiply each alpha by [# matchups]).
#
# If "isSeedWeighted" is set to "True", then the seed-weighted
# average alpha values are used.
#
# This version no longer requires models to specify the alpha value
# parameters for each round. Round 1 is always matchup-specific
# (with optional grouping), and Rounds 2-6 always use a
# weighted average.
######################################################################

BT_probs = {}
def load_BT_probs():
    global BT_probs
    for year in range(2013, 2020):
        year_dist = defaultdict(float)
        data = pd.read_csv('bradleyTerry/probs-{}.csv'.format(year), usecols=['player1', 'player2', 'prob1wins']).values
        for i in range(data.shape[0]):
            s1, s2, p = data[i, :]
            s1 = int(s1[1:])
            s2 = int(s2[1:])
            year_dist[(s1, s2)] = p
            year_dist[(s2, s1)] = 1. - p
        BT_probs[year] = year_dist


# Returns the estimated probability that s1 beats s2
def getP(s1, s2, model, year, roundNum):
    if s1 == s2:
        return 0.5
    return BT_probs[year][(s1, s2)]


# This function generates a 63-element list of 0s and 1s
# to represent game outcomes in a bracket. The model specifies
# which alpha value(s) to use for each round.
def generateBracket(model, year):
    bracket = []

    # random.seed()

    endModel = 'None'
    if 'endModel' in model:
        endModel = model['endModel']

    e8Seeds = []
    if endModel == 'E8':
        for i in range(4):
            e8Seeds.append(getE8SeedTop(year))
            e8Seeds.append(getE8SeedBottom(year))
    else:
        e8Seeds = [-1, -1, -1, -1, -1, -1, -1, -1]

    f4Seeds = []
    if endModel == 'F4_1':
        for i in range(4):
            f4Seeds.append(getF4SeedTogether(year))
    elif endModel == 'F4_2':
        for i in range(4):
            f4Seeds.append(getF4SeedSplit(year))
    else:
        f4Seeds = [-1, -1, -1, -1]

    ncgSeeds = [-1, -1]
    if 'Rev' in endModel:
        champion = getChampion(year)
        runnerUp = getRunnerUp(year)
        champRegion = int(floor(random.random() * 4))
        champHalf = champRegion / 2
        ruRegion = int(floor(random.random() * 2))

        if champHalf == 0:
            ncgSeeds = [champion, runnerUp]
        else:
            ncgSeeds = [runnerUp, champion]

        ffrRegion = 1 - ruRegion

        if champRegion < 2:
            ruRegion += 2
            ffrRegion += 2
            ffcRegion = 1 - champRegion
        else:
            ffcRegion = 5 - champRegion

        f4Seeds[champRegion] = champion
        f4Seeds[ruRegion] = runnerUp

    if endModel == 'Rev_4':
        f4Seeds[ffcRegion] = getF4SeedTogether(year)
        f4Seeds[ffrRegion] = getF4SeedTogether(year)

    # Loop through regional rounds R64, R32, and S16
    for region in range(4):
        seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        for roundNum in range(1, 5):
            numGames = int(len(seeds) / 2)
            newSeeds = []
            for gameNum in range(numGames):
                s1 = seeds[2 * gameNum]
                s2 = seeds[2 * gameNum + 1]

                # Force any fixed F4/E8 seeds to make it through
                s1Wins = (s1 == f4Seeds[region]) or ((roundNum < 4) and (
                        (s1 == e8Seeds[2 * region]) or (
                        s1 == e8Seeds[2 * region + 1])))
                s2Wins = (s2 == f4Seeds[region]) or ((roundNum < 4) and (
                        (s2 == e8Seeds[2 * region]) or (
                        s2 == e8Seeds[2 * region + 1])))

                if s1Wins:
                    p = 1
                elif s2Wins:
                    p = 0
                else:
                    p = getP(s1, s2, model, year, roundNum)

                if random.random() <= p:
                    bracket.append(1)
                    newSeeds.append(s1)
                else:
                    bracket.append(0)
                    newSeeds.append(s2)
            seeds = newSeeds
        f4Seeds[region] = seeds[0]

    # Round 5:
    for gameNum in range(2):
        s1 = f4Seeds[2 * gameNum]
        s2 = f4Seeds[2 * gameNum + 1]

        if 'Rev' in endModel:
            if (2 * gameNum == champRegion) or (2 * gameNum == ruRegion):
                p = 1
            elif (2 * gameNum + 1 == champRegion) or (
                    2 * gameNum + 1 == ruRegion):
                p = 0
            else:
                p = getP(s1, s2, model, year, 5)
        else:
            p = getP(s1, s2, model, year, 5)

        if random.random() <= p:
            bracket.append(1)
            ncgSeeds[gameNum] = s1
        else:
            bracket.append(0)
            ncgSeeds[gameNum] = s2

    # Round 6:
    s1 = ncgSeeds[0]
    s2 = ncgSeeds[1]

    if 'Rev' in endModel:
        if champHalf == 0:
            p = 1
        else:
            p = 0
    else:
        p = getP(s1, s2, model, year, 6)

    if random.random() <= p:
        bracket.append(1)
    else:
        bracket.append(0)

    return bracket


# Unused: if we want to measure this later, we can.
#
# # This function computes how many picks a bracket
# # got correct given the bracket's score vector.
# def calcCorrectPicks(scoreVector):
# 	numCorrectPicks = 0
# 	for roundNum in range(1, 7):
# 		numCorrectPicks += scoreVector[roundNum] / (10 * (2 ** (roundNum - 1)))
# 	return numCorrectPicks


# This function generates and scores brackets
# for the given year using the given model.
# It prints the results in JSON format.
def performExperiments(numTrials, year, batchNumber, model):
    correctVector = getActualBracketVector(year)

    scores = [None] * numTrials
    for n in range(numTrials):
        newBracketVector = generateBracket(model, year)
        newBracketScore = scoreBracket(newBracketVector, correctVector)
        # numCorrectPicks = calcCorrectPicks(newBracketScore)
        scores[n] = newBracketScore[0]

    bracketListDict = {'year': year, 'actualBracket': ''.join(
        str(bit) for bit in correctVector), 'scores': scores}

    if numTrials < 1000:
        folderName = 'Experiments/{0}Trials'.format(numTrials)
    else:
        folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))
    batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)

    outputFilename = '{2}/generatedScores_{0}_{1}.json'.format(
        model['modelName'], year, batchFolderName)
    with open(outputFilename, 'w') as outputFile:
        outputFile.write(json.dumps(bracketListDict))


######################################################################
# This script runs experiments with the given models,
# number of trials, and number of batches for 2013 through 2018.
######################################################################
load_BT_probs()

# Load models
modelFilename = sys.argv[3]
with open(modelFilename, 'r') as modelFile:
    modelsDataJson = modelFile.read().replace('\n', '')

modelsDict = json.loads(modelsDataJson)
modelsList = modelsDict['models']

numTrials = int(sys.argv[1])
numBatches = int(sys.argv[2])
if len(sys.argv) == 5:
    years = [int(sys.argv[4])]
else:
    years = range(2013, 2020)

# import cProfile, pstats
# from io import StringIO
# pr = cProfile.Profile()
# pr.enable()

for modelDict in modelsList:
    modelName = modelDict['modelName']

    print '{0:<8s}: {1}'.format(modelName, time.strftime("%Y-%m-%d %H:%M"))

    for year in years:
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

#
# pr.disable()
# s = StringIO()
# ps = pstats.Stats(pr).sort_stats('cumulative')
# ps.print_stats()
