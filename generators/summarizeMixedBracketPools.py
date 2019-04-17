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
# Modified: 16 Apr 2018

def summarizeBatch(numTrials, batchNumber, modelsFilename, outputFile):
    for year in range(2013, 2020):
        outputFile.write('{0} Tournament:,'.format(year))

        numModels = 1
        maxScores = []
        countAboveEspnMin = []
        # maxCorrectPicks = []
        percentile95 = []
        percentile99 = []
        proportionsAbovePF = []

        if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
            meanScores = []
            varianceScores = []
            medianScores = []

        outputFile.write('{0},'.format(modelsFilename))

        if numTrials < 1000:
            folderName = 'Experiments/{0}Trials'.format(numTrials)
        else:
            folderName = 'Experiments/{0}kTrials'.format(
                int(numTrials / 1000))

        batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)
        inputFilename = '{2}/generatedScores_{0}Ensemble_{1}.json'.format(
            modelsFilename, year, batchFolderName)

        with open(inputFilename, 'r') as inputFile:
            dataPyDict = json.load(inputFile)
        scores = dataPyDict['scores']

        brackets = []
        for score in scores:
            newBracket = SimpleBracket(None, [score])
            brackets.append(newBracket)

        nBrackets = len(brackets)

        # Determine max scoring bracket, as well as 95/99th percentiles
        brackets.sort(key=lambda x: x.scores[0], reverse=True)
        maxScoringBracket = brackets[0]
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
        for i in range(numModels):
            outputFile.write('{0},'.format(maxScores[i]))
        outputFile.write('\n')

        if numTrials < MEAN_MEDIAN_VAR_CUTOFF:
            outputFile.write('Median score,')
            for i in range(numModels):
                outputFile.write('{0},'.format(medianScores[i]))
            outputFile.write('\n')

            outputFile.write('Mean score,')
            for i in range(numModels):
                outputFile.write('{0},'.format(meanScores[i]))
            outputFile.write('')
            outputFile.write('\n')

            outputFile.write('Var(scores),')
            for i in range(numModels):
                outputFile.write('{0},'.format(varianceScores[i]))
            outputFile.write('\n')

        outputFile.write('No. in ESPN top 100,')
        for i in range(numModels):
            outputFile.write('{0},'.format(countAboveEspnMin[i]))
        outputFile.write('\n')

        # outputFile.write('Max correct picks,')
        # for i in range(numModels):
        # 	outputFile.write('{0},'.format(maxCorrectPicks[i]))
        # outputFile.write('\n')

        outputFile.write('95th percentile,')
        for i in range(numModels):
            outputFile.write('{0},'.format(percentile95[i]))
        outputFile.write('\n')

        outputFile.write('99th percentile,')
        for i in range(numModels):
            outputFile.write('{0},'.format(percentile99[i]))
        outputFile.write('\n')

        outputFile.write('Proportion >= PF ({0}),'.format(pfTotalScore))
        for i in range(numModels):
            outputFile.write('{0:10.4f},'.format(proportionsAbovePF[i]))
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
ensembleName = modelFilename.split('/')[-1].replace('.json', '')

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
        summarizeBatch(numTrials, batchNumber, ensembleName, outputFile)
