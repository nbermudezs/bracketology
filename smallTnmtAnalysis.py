import csv
import math
import numpy as np
import os
from pprint import pprint
import sys

######################################################################
# 
# Author: 	Ian Ludden
# Date:		23 July 2019
# 
# smallTnmtAnalysis.py
# 
# Computes the likelihood of each outcome for a seeded four-team 
# tournament. The hope is to observe patterns that may hold in the 
# 64-team March Madness tournament. 
# 
# The structure of the four-team tournament is:
# 
# Round 1		Round 2
# 
# 1 --- 
#      |________
#      |        |
# 4 ---         |
#                ---- Champion
# 3 ---         |
#      |________|
#      |
# 2 --- 
# 
######################################################################

def computeLogLikelihood(bracketVector, winProbs):
	"""Computes the log-likelihood of the given bracket vector."""
	totalLogProb = 0
	# Round 1:
	# 1 vs 4
	if bracketVector[0] == 1:
		totalLogProb += math.log(winProbs[1,4])
		winner1 = 1
	else:
		totalLogProb += math.log(winProbs[4,1])
		winner1 = 4

	# 3 vs 2
	if bracketVector[1] == 1:
		totalLogProb += math.log(winProbs[3,2])
		winner2 = 3
	else:
		totalLogProb += math.log(winProbs[2,3])
		winner2 = 2

	# Round 2:
	if bracketVector[2] == 1:
		totalLogProb += math.log(winProbs[winner1,winner2])
	else:
		totalLogProb += math.log(winProbs[winner2,winner1])

	return totalLogProb


if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit('Must provide csv file with win probabilities.')

	inputFilename = str(sys.argv[1])
	with open(inputFilename, 'r') as f:
		reader = csv.reader(f)
		# Ignore top row of seed numbers
		next(reader)
		data = list(reader)

	arrayData = np.array(data).astype(float)
	winProbs = np.zeros((5,5))
	winProbs[1:,1:] = arrayData[:,1:]

	brackets = ['000','001','010','011',
				'100','101','110','111']
	for bracket in brackets:
		bracketVector = [int(bracket[i]) for i in range(len(bracket))]
		print('{0} {1:.4f}'.format(bracket, computeLogLikelihood(bracketVector, winProbs)))
