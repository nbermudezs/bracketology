#!/usr/bin/env python
import csv
import numpy as np
import os.path
import sys

######################################################################
# Author: 	Ian Ludden
# Date: 	05 Mar 2018
# Modified: 10 Apr 2018
#
# This script prepares multiple batches of treatments for
# multiple comparisons with the best (MCB) analysis.
######################################################################

numTrials = int(sys.argv[1])
metricName = sys.argv[2]
minBatchNum = int(sys.argv[3])
maxBatchNum = int(sys.argv[4])

if len(sys.argv) > 5:
    summaries_root = sys.argv[5]
else:
    summaries_root = 'Summaries'

if numTrials < 1000:
    trialsString = '{0}'.format(numTrials)
else:
    trialsString = '{0}k'.format(int(numTrials / 1000))

for year in range(2013, 2020):
    sys.stdout.write('{0} Tournament:\n'.format(year))
    sys.stdout.write('Batch,\n')

    values = []
    for batchNum in range(minBatchNum, maxBatchNum + 1):
        batchFilename = '{2}/exp_{0}_batch_{1:02d}.csv'.format(
            trialsString, batchNum, summaries_root)

        with open(batchFilename, 'rb') as csvfile:
            reader = csv.reader(csvfile)

            isCorrectYear = False
            for row in reader:
                if len(row) == 0:
                    continue
                if str(year) in row[0]:
                    isCorrectYear = True
                if isCorrectYear and (metricName in row[0]):
                    sys.stdout.write('{0}, '.format(batchNum))
                    sys.stdout.write(', '.join(row[1:]))
                    scores = [int(x) for x in row[1:-1]]
                    values.append(scores)
                    sys.stdout.write('\n')
                    break
    sys.stdout.write(
        'mean, ' + ', '.join(np.mean(values, axis=0).astype(str)) + '\n')
    sys.stdout.write(
        'var, ' + ', '.join(np.var(values, axis=0, ddof=0).astype(str)) + '\n')
    sys.stdout.write(
        'std, ' + ', '.join(np.std(values, axis=0, ddof=0).astype(str)) + '\n')
    sys.stdout.write('\n\n\n')
