#!/usr/bin/env python
import numpy as np
import random
from math import log, ceil, floor

# Author:
#     Ian Ludden
#
# Date created:
#     09 Apr 2018
#
# Last modified:
#     09 Apr 2019
#
# These utilities support sampling from the Final Four, 
# Elite Eight, Champion, and Runner-up seed distributions. 

# In the following parameter lists, 
# index 0 corresponds to parameters for predicting 2013 
# (i.e. using data from 1985 through 2012)

############################################################
# Parameters for National Champion:
############################################################
pChamp = [0.5090909091, 0.5178571429, 0.4761904762, 0.484375, 0.4848484848,
          0.4925373134, 0.5]
champSum = [0.9966270601, 0.997079845, 0.9943325575, 0.9950034329, 0.9950400208,
            0.9956022259, 0.99609375]
pRU = [0.3835616438, 0.3766233766, 0.3529411765, 0.3604651163, 0.367816092,
       0.375, 0.3736263736]
ruSum = [0.9791494637, 0.9771963641, 0.9692708809, 0.9720157349, 0.9744878244,
         0.9767169356, 0.9763044002]

############################################################
# Parameters for Final Four all together:
############################################################
pF4 = [0.4160, 0.4036, 0.3926, 0.3929, 0.3772, 0.3757, 0.3789]
pF4Sum = [0.9984, 0.9980, 0.9975, 0.9975, 0.9966, 0.9965, 0.9967]

# F4 probability of choosing 11 (instead of using trunc. geom.)
pF4Choose11 = [0.025, 0.024, 0.022, 0.021, 0.020, 0.019, 0.026]

############################################################
# Parameters for Final Four split:
############################################################
nTop = [105, 108, 110, 113, 116, 119, 122]
nBottom = [7, 8, 10, 11, 12, 13, 14]
pF4Top = [0.4565, 0.4519, 0.4545, 0.4612, 0.464, 0.4667, 0.4692]
pF4TopSum = [0.9742, 0.9729, 0.9737, 0.9755, 0.9763, 0.9770, 0.9776]

############################################################
# Parameters for Elite Eight: ("Top" is 1,16,8,9,5,12,4,13)
############################################################
topSeeds = [0, 1, 4, 5, 8, 9, 12, 13, 16]
bottomSeeds = [0, 2, 3, 6, 7, 10, 11, 14, 15]

pE8Top = [0.56, 0.54, 0.54, 0.54, 0.54, 0.53, 0.51]
pE8TopSum = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
pE8Choose1 = [0.33, 0.31, 0.31, 0.32, 0.34, 0.34, 0.36]

pE8Bottom = [0.48, 0.49, 0.49, 0.49, 0.49, 0.48, 0.49]
pE8BottomSum = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]
pE8Choose11 = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.04]


# Returns a sample of a truncated geometric r.v.
# with parameter p and probabilities that add to pSum.
def getTruncGeom(p, pSum):
    u = random.random() * pSum
    return int(ceil(log(1 - u) / log(1 - p)))


# Returns a perturbed probability value.
# The perturbation depends on the specified parameters and it can be either
# a fixed amount (e.g. 0.1) or a percentage of the argument probability
# (e.g. prob + prob * 0.1).
# Regardless of the perturbation, the returned value is ensured to be a
# valid probability value (i.e., between 0 and 1)
def perturbProb(prob, model):
    if model and model.get('perturbation'):
        if model['perturbation'].get('type', 'rv') == 'rv':
            percent = model['perturbation'].get('percent', 0.1)
            return np.clip(
                prob + np.random.uniform(low=0, high=prob * percent), 0., 1.)
        elif model['perturbation'].get('type', 'rv') == 'fixed':
            percent = model['perturbation'].get('percent', 0.1)
            return np.clip(prob + percent, 0., 1.)
    return prob


# Returns a seed for the National Champion.
def getChampion(year, model=None):
    pC = perturbProb(pChamp[year - 2013], model)
    cSum = champSum[year - 2013]
    return getTruncGeom(pC, cSum)


# Returns a seed for the National Runner-Up.
def getRunnerUp(year, model=None):
    pR = perturbProb(pRU[year - 2013], model)
    rSum = ruSum[year - 2013]
    return getTruncGeom(pR, rSum)


# Returns a seed for the Final Four, sampled 
# according to a truncated geometric distribution 
# with additional probability of choosing an 11 seed.
# This is also referred to as Model1.
def getF4SeedTogether(year, model=None):
    p = perturbProb(pF4[year - 2013], model)
    pSum = pF4Sum[year - 2013]

    seed = -1

    if random.random() <= pF4Choose11[year - 2013]:
        seed = 11
    else:
        seed = getTruncGeom(p, pSum)

    return seed


# Returns a seed for the Final Four, sampled
# from a truncated geometric distribution for Seeds 1-6 
# or a fixed-weight distribution for Seeds 7-12. 
# The 7-12 distribution weights 9, 10, and 12 evenly 
# and gives 7, 8, and 11 double that weight. 
# This is also referred to as Model2.
def getF4SeedSplit(year, model=None):
    nTopHalf = nTop[year - 2013]
    nBottomHalf = nBottom[year - 2013]
    nTotal = nTopHalf + nBottomHalf

    pUseTop = nTopHalf * 1.0 / nTotal

    if random.random() <= pUseTop:
        p = perturbProb(pF4Top[year - 2013], model)
        pSum = pF4TopSum[year - 2013]
        seed = getTruncGeom(p, pSum)
    else:
        seedList = [7, 7, 8, 8, 9, 10, 11, 11, 12]
        index = random.randint(0, len(seedList) - 1)
        seed = seedList[index]

    return seed


# Returns a seed for the "top half" of the Elite Eight, sampled
# according to a modified truncated geometric distribution. 
# The top half is the seeds 1, 4, 5, 8, 9, 12, 13, 16.
def getE8SeedTop(year, model=None):
    pChoose1 = pE8Choose1[year - 2013]

    if random.random() <= pChoose1:
        seed = 1
    else:
        p = perturbProb(pE8Top[year - 2013], model)
        pSum = pE8TopSum[year - 2013]
        index = getTruncGeom(p, pSum)
        seed = topSeeds[index]

    return seed


# Returns a seed for the "bottom half" of the Elite Eight, sampled
# according to a modified truncated geometric distribution. 
# The bottom half is the seeds 2, 3, 6, 7, 10, 11, 14, 15.
def getE8SeedBottom(year, model=None):
    pChoose11 = pE8Choose11[year - 2013]

    if random.random() <= pChoose11:
        seed = 11
    else:
        p = perturbProb(pE8Bottom[year - 2013], model)
        pSum = pE8BottomSum[year - 2013]
        index = getTruncGeom(p, pSum)
        seed = bottomSeeds[index]

    return seed


def toDistribution(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


def cdfToPdf(x):
    return [x[0]] + [(x[i+1] - x[i]) for i in range(len(x) - 1)]


def minMaxNormalization(x, new_min=0., new_max=1.):
    _min = np.min(x)
    _max = np.max(x)
    if new_min is None:
        new_min = _min
    return (x - _min) / (_max - _min) * (new_max - new_min) + new_min


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    champs = [getChampion(2019) for _ in range(10000)]
    sns.distplot(champs, bins=np.max(champs) - np.min(champs), kde=False)
    plt.show()

