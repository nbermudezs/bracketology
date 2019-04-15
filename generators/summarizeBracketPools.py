#!/usr/bin/env python
import json
import os.path
import random
import sys
from math import log
from bracketClassDefinitions import Bracket
from bracketClassDefinitions import Region
from bracketClassDefinitions import SimpleBracket
from bracketClassDefinitions import buildBracketFromJson
from scoringUtils import applyRoundResults
from scoringUtils import scoreBracket
import numpy as np


# Author: 	Ian Ludden
# Date: 	07 Mar 2018
# Modified: 18 Dec 2018

def summarizeBatch(numTrials, batchNumber, modelsList, outputFile):
    for year in range(2013, 2020):
        outputFile.write('{0} Tournament:,'.format(year))

        numModels = len(modelsList)

        maxScores = []
        minScores = []
        countAboveEspnMin = []
        # maxCorrectPicks = []
        percentile95 = []
        percentile99 = []
        proportionsAbovePF = []

        if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
            meanScores = []
            varianceScores = []
            medianScores = []

        for index in range(numModels):
            modelName = modelsList[index]['modelName']

            if numTrials < 1000:
                folderName = 'Experiments/{0}Trials'.format(numTrials)
            else:
                folderName = 'Experiments/{0}kTrials'.format(
                    int(numTrials / 1000))

            batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)
            inputFilename = '{2}/generatedScores_{0}_{1}.json'.format(
                modelName, year, batchFolderName)

            if not os.path.exists(inputFilename):
                continue

            outputFile.write('{0},'.format(modelName))
            with open(inputFilename, 'r') as inputFile:
                dataJson = inputFile.read().replace('\n', '')

            dataPyDict = json.loads(dataJson)
            scores = dataPyDict['scores']

            brackets = []
            for score in scores:
                newBracket = SimpleBracket(None, [score])
                brackets.append(newBracket)

            nBrackets = len(brackets)

            # Determine max scoring bracket, as well as 95/99th percentiles
            brackets.sort(key=lambda x: x.scores[0], reverse=True)
            maxScoringBracket = brackets[0]
            minScoringBracket = brackets[-1]
            bracket95 = brackets[int(0.05 * nBrackets) - 1]
            bracket99 = brackets[int(0.01 * nBrackets) - 1]

            # Determine max correct picks
            # brackets.sort(key=lambda x: x.correctPicks, reverse=True)
            # maxCorrectPicksBracket = brackets[0]

            # Determine score of Pick Favorite model
            actualBracket = dataPyDict['actualBracket']
            actualBracketVector = [int(actualBracket[i]) for i in
                                   range(len(actualBracket))]

            pickFavoriteString = '111111111000101111111111000101111111111000101111111111000101111'
            pickFavoriteVector = [int(pickFavoriteString[i]) for i in range(63)]
            pfScores = scoreBracket(pickFavoriteVector, actualBracketVector,
                                    True)
            pfTotalScore = pfScores[0]

            bracketsAbovePickFavorite = [b for b in brackets if
                                         b.scores[0] >= pfTotalScore]
            nBracketsAbovePF = len(bracketsAbovePickFavorite)
            proportionAbovePF = nBracketsAbovePF * 1.0 / nBrackets

            bracketsInEspnLeaderboard = [b for b in brackets if
                                         b.scores[0] >= espnMin[year - 2013]]
            countAboveEspnMin.append(len(bracketsInEspnLeaderboard))

            if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
                scores = [x.scores[0] for x in brackets]
                meanScore = np.mean(scores)
                varianceScore = np.var(scores)
                medianScore = np.median(scores)

            maxScores.append(maxScoringBracket.scores[0])
            minScores.append(minScoringBracket.scores[0])
            # maxCorrectPicks.append(maxCorrectPicksBracket.correctPicks)
            percentile95.append(bracket95.scores[0])
            percentile99.append(bracket99.scores[0])
            proportionsAbovePF.append(proportionAbovePF)

            if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
                meanScores.append(meanScore)
                varianceScores.append(varianceScore)
                medianScores.append(medianScore)

        outputFile.write('\n')

        outputFile.write('Max score,')
        for val in maxScores:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')
        outputFile.write('Min score,')
        for val in minScores:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')

        if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
            outputFile.write('Median score,')
            for val in medianScores:
                outputFile.write('{0},'.format(val))
            outputFile.write('\n')

            outputFile.write('Mean score,')
            for val in meanScores:
                outputFile.write('{0},'.format(val))
            outputFile.write('')
            outputFile.write('\n')

            outputFile.write('Var(scores),')
            for val in varianceScores:
                outputFile.write('{0},'.format(val))
            outputFile.write('\n')

        outputFile.write('No. in ESPN top 100,')
        for val in countAboveEspnMin:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')

        # outputFile.write('Max correct picks,')
        # for i in range(numModels):
        # 	outputFile.write('{0},'.format(maxCorrectPicks[i]))
        # outputFile.write('\n')

        outputFile.write('95th percentile,')
        for val in percentile95:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')

        outputFile.write('99th percentile,')
        for val in percentile99:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')

        outputFile.write('Proportion >= PF ({0}),'.format(pfTotalScore))
        for val in proportionsAbovePF:
            outputFile.write('{0},'.format(val))
        outputFile.write('\n')
        outputFile.write('\n')


# This script summarizes the experiments generated by
# runExperiments.py.

espnMin = [1590, 1520, 1760, 1630, 1650, 1550, 1730]
MEAN_MEDIAN_VAR_CUTOFF = 10001

numTrials = int(sys.argv[1])
numBatches = int(sys.argv[2])

# Load models
if len(sys.argv) >= 4:
    modelFilename = sys.argv[3]
else:
    modelFilename = 'models.json'
with open(modelFilename, 'r') as modelFile:
    modelsDataJson = modelFile.read().replace('\n', '')

if len(sys.argv) == 5:
    outputDir = sys.argv[4]
else:
    outputDir = 'Summaries'

modelsDict = json.loads(modelsDataJson)
modelsList = modelsDict['models']

if numTrials < 1000:
    trialsString = '{0}'.format(numTrials)
else:
    trialsString = '{0}k'.format(int(numTrials / 1000))

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

for batchNumber in range(numBatches):
    outputFilename = '{0}/exp_{1}_batch_{2:02d}.csv'.format(outputDir,
                                                            trialsString,
                                                            batchNumber)
    with open(outputFilename, 'w') as outputFile:
        summarizeBatch(numTrials, batchNumber, modelsList, outputFile)
