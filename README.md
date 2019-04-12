## User Guide

This repository summarizes the work done on bracketology by Ian G. Ludden, Meghan Shanks, and Nestor A. Bermudez between 2017 and 2019.


### Installation

The majority of scripts in this repository are written in Python. 
While most of the scripts can be executed in both Python 2.x and 3.x, their compatibility with these versions is offered as a best effort approach, i.e., we didn't spend time making sure they work on both versions.
If a script does not work on one of the Python versions feel free to update it accordingly. 
Similarly, we have tried to make the documentation of the scripts and processes comprehensive but if details were missing, please submit an update.

We recommend setting up virtual environment to install the dependencies of this repository --- you can follow the instructions
in [virtualenv](https://virtualenv.pypa.io/en/latest/) to do so, or use your favorite environment manager.

Once you have set up your environment, install the necessary dependencies by executing
```bash
pip install -r requirements.txt
```

### Project structure
This repository is divided in two main directories: *generators* and *search*.
The *generators* directory contains scripts to fit the Power Model (**TODO: add citation**), to fit the Bradley-Terry model (**TODO: add citation**), to generate bracket pools using the multiple variations of these models, and to score and summarize the performance of a bracket pool.

#### Input Data
All the models presented in this repository use only the team seeds as input. As a result, a bracket can be represented in a 63-bit vector where each bit encodes 
the outcome of a match-up. The majority of work done as of April 2019 uses a positional encoding of the game. 
That is, in the traditional representation of a bracket (see Fig 1), the two teams in the match-up are placed one on top of the other; then the corresponding bit has a value one if the seed in the top position won the match-up and zero otherwise.
That being said, the game outcome can also be encoded comparing the seed of the teams competing, i.e., the bit has a value one if the team with the numerically lower seed won the game.
Future work can involve defining and utilizing other encodings.

The *generators* contains two JSON files that summarize the vector encoding of the *modern-era* tournaments (i.e., tournaments after 1985): 
- *allBracketsTTT.json*: positional encoding of the brackets
- *allBracketsFFF.json*: *pick-favorite* encoding of the brackets (i.e., bit value one if lower seed wins).

#### Preprocessing Scripts
The Power Model defines parameters $\alpha$ for each seed match-up. These parameters are calculated  as a function of
the Maximum Likelihood Estimate of the particular seed match-up. The *generators/fitPowerModel.py* script 
takes the bracket input files and estimates the Power Model parameters.

Similarly, *generators/fitBradleyTerry.py* can be used to estimate the $\beta$ parameters needed for the Bradley-Terry model, which are saved in a JSON file under the *generators/bradleyTerry* directory.

#### Bracket Pool Generators
The *generators* directory contains scripts to generate bracket pools using various approaches:
- *generatorPower.py* supports the generation of bracket pools using one of the five original Power Model variations proposed by (**TODO: add citation**).
- *generatorSA.py* supports the same five variations of the Power Model but also supports the use of different first round probabilities (obtained using Simulated Annealing). 
The specific probability values used are described in **TODO: add citation**.
- *generatorBitwise.py* can be used to generate brackets were each bit is calculated solely based on the MLE of each bit, without using seed information as in the Power Model.
- *generatorBradleyTerry.py* can generate bracket pools using the original Bradley-Terry model and its backwards variations analogous to the Power Model variants proposed in **TODO: add citation**.
In particular, the Bradley-Terry generator requires winning probabilities that are computed by first executing the *generators/utils/preprocessForBradleyTerry.py* script, followed by the *generators/fitBradleyTerry.py* script.

All these models share the same structure and accept the same 
bash options:
```
python <generator.py> <numTrials> <numBatches> <modelsFilepath>
```
where *numTrials* is the size of the bracket pool, *numBatches* is the number of replications, each of which generates an entire bracket pool of size *numTrials*, and *modelsFilepath* is the relative path to a JSON file
that specifies the model parameters for the generator.

The models file is a JSON document containing a list of models, each of which adhere to the following definition:
```json
{
  "modelName": "str",
  "modelDesc": "[str]",
  "endModel": "[str]",
  "annealing_model": "[str]",
  "bradleyTerry": "[bool]",
  "format": "[str]"
}
```
where the value in the snippet above indicates the data type of the attribute and the square brackets around it indicate that the field is optional.

- The `modelName` attribute is the main identifier for the model and it is used as part of the filename for the file where the brackets, scores, and statistics collected during generation are stored.
- `modelDesc` is specified for documentation only and it is not used in any of the scripts available in this repo.
- The default behaviour of the scripts is to generate brackets in a forward fashion (i.e., first determine the outcomes of the first round, then the second, etc...), the `endModel` attribute allows
the generator to create brackets backwards; in particular, it determines the seeds that will reach a particular round, locking some of the earlier round outcomes. As April 2019, five `endModel` values are supported:
    - **F4_1**: this is supported by the *generatorPower.py* and *generatorBradleyTerry.py* scripts. Use **F4_A** instead for the other scripts.
    - **F4_2**: this is supported by the *generatorPower.py* and *generatorBradleyTerry.py* scripts. Use **F4_B** instead for the other scripts.
    - **E8**: supported by all scripts. It locks down the seeds for the teams that reach the Elite Eight round.
    - **Rev_2**: this is supported by the *generatorPower.py* and *generatorBradleyTerry.py* scripts. Use **NCG** for other scripts. It locks down the seeds and regions for the champion
    and runner up.
    - **Rev_4**: supported by *generatorPower.py* and *generatorBradleyTerry.py* scripts. Use **combined** for other scripts. This model locks down the champion, runner up, and the other two regional champions.
    These models determine which seed will reach the F4, E8, or NCG rounds by using a truncated geometric function.
- By default, the generators use Maximum Likelihood Estimate winning probabilities. **TODO: add citation** used a simulated annealing algorithm to perturb the MLE probabilities for the first
round of the tournament. The `annealing_model` can be used to tell the generator to use a particular set of these probabilities instead of the MLE probabilities. 
This attribute allows one of ten values: 25_1985, 26_1985, 27_1985, 28_1985, 29_1985, 30_1985, 31_1985, 28_2002, 29_2002, 30_2002.
- `format` determines the outcome encoding used for the vector representation of the bracket. Currently, only **TTT** is supported for both generation and scoring of brackets.
The generators can create brackets using the **FFF** encoding but the scoring function needs to be updated to support the scoring of this definition.

#### Model Evaluation and Comparison
It is of interest to assess the quality of a bracket pool and use these metrics to 
compare multiple models. 

(**TODO: add citation**) defines two performance metrics for a bracket pool: the Max Score and the EPSN Count, i.e., the number of brackets in the pool that have an ESPN score at least as good as the worst bracket in the ESPN Top 100 Leaderboard.
The *generators/scoringUtils.py* script implements the ESPN scoring function, which is used by the *summarizeBracketPools.py* script
to calculate summary statistics, the Max Score, and ESPN Count metrics for each bracket pool.
The *summarizeBracketPools.py* script can be executed as
```
python summarizeBracketPools.py <numTrials> <numBatches> <modelsFilepath> <outputDir>
```
where *numTrials*, *numBatches* and *modelsFilepath* should be the same used when generating the bracket and *outputDir* must be a relative path to a directory (may not exist) where
the files summarizing the bracket pools would be stored.

Once each bracket pool is analyzed by the *summarizeBracketPools.py* script, we use a Multiple Comparison with the Best (MCB) method to compare the models between each other.
The output of *summarizeBracketPools.py* is fed into the *prepareForMCB.py* script as a preprocessing step by executing
```
python prepareForMCB.py <numTrials> <metric> <batchStart> <batchEnd> <summaryDir>
```
where *numTrials* should match the one used for the other scripts, *metric* can take the **ESPN** or **Max score** values, to tabulate the ESPN count or Max score metrics, respectively, *batchStart* and *batchEnd* determine which batches should be summarize -- in practice, *batchStart*=0 and *batchEnd*=*numBatches*-1 are the most common values. Finally, the *summaryDir* is the relative path to the directory 
where the output of *summarizeBracketPools.py* was stored.

#### Utility files
- *generators/utils/extractScores.py* can be used to take one of the bracket pool files that contain the actual brackets and convert it
into the new format that contains only the scores of the brackets.
- *generators/utils/tripletsUniformityTest.py* performs a Chi-square test over the distribution of values of triplets of bits to check whether their distribution seems uniform.
- *generators/utils/preprocessForBradleyTerry.py* takes the historical brackets and creates the summary necessary to fit the Bradley-Terry model (winning records for each seed match-up).


#### Visualization
The `generators/viz` directory contains scripts to plot score distributions, boxplots for model comparison, and triplet distributions.
Most of these scripts adhere to the same parameters used by the generator scripts:
```
python <script.py> <numTrials> <numBatches> <modelsFilepath> <summaryDir>
```


### CHANGELOG
- The generators no longer store the vector representation of the generated brackets. Instead, the scores and statistics about the seeds that reach each of the rounds
are tracked and save. This change was motivated by i) reducing the runtime, ii) allowing really large bracket pools (1B+) to be generated (memory constrains), and iii) storing space.
- The *summarizeBracketPool.py* is a modified version of the original *summarizeExperimentsFixedAlpha.py* script from *ian-ludden* repo. The new script handles the new generators where only the scores are saved to disk.
- The *summarizeBracketPool.py* accepts a parameter to specify the output directory of the summary files. This allows for summaries of different generators/models to be kept separate.
- *generatorPower.py* now supports an optional parameter that allows the generation of brackets for a single year instead of the 2013-2018 range.
- All other scripts except *generatorPower.py* support an optional parameter that allows the generation of brackets for **one** of the models in the models file given instead of generating brackets for all of them.


### Contribution Guidelines
**...TODO...**