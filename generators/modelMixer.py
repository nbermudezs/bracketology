__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import numpy as np
import sys

from collections import defaultdict
from utils.extractScores import extract_scores


def sample(n, scores):
    n_models = len(scores)
    counts = defaultdict(int)
    for i in range(n):
        model_i = np.random.randint(n_models)
        counts[model_i] += 1
    return np.concatenate([np.random.choice(scores[model_i], size, replace=False)
                           for model_i, size in counts.items()])


def read_scores(path):
    with open(path) as f:
        data = json.load(f)
        return data['scores'], data['actualBracket']


if __name__ == '__main__':
    import sys

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

    out = modelFilename.split('/')[-1].replace('.json', '')

    for year in range(2013, 2020):
        for batchNumber in range(numBatches):
            if numTrials < 1000:
                folderName = 'Experiments/{0}Trials'.format(numTrials)
            else:
                folderName = 'Experiments/{0}kTrials'.format(
                    int(numTrials / 1000))
            batchFolderName = '{0}/Batch{1:02d}'.format(folderName,
                                                        batchNumber)

            models_scores = []
            actual_bracket = None
            for modelDict in modelsList:
                modelName = modelDict['modelName']
                scoresFilepath = '{2}/generatedScores_{0}_{1}.json'.format(
                    modelName, year, batchFolderName)
                scores, actual_bracket = read_scores(scoresFilepath)
                models_scores.append(scores)

            result = sample(numTrials, models_scores)
            outputFilepath = '{2}/generatedScores_{0}Ensemble_{1}.json'.format(out, year, batchFolderName)
            with open(outputFilepath, 'w') as f:
                json.dump({'scores': result.tolist(), 'actualBracket': actual_bracket}, f)
