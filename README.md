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
This generator also supports the use of 3-bit paths and triplets. That is, it uses the observed distribution of a group (or groups) of 3bits to decide the value of these bits in a generated bracket.
A detailed explanation of these paths and triplets can be found in [this pdf](doc triplets and paths.pdf).
- *generatorBradleyTerry.py* can generate bracket pools using the original Bradley-Terry model and its backwards variations analogous to the Power Model variants proposed in **TODO: add citation**.
In particular, the Bradley-Terry generator requires winning probabilities that are computed by first executing the *generators/utils/preprocessForBradleyTerry.py* script, followed by the *generators/fitBradleyTerry.py* script.
- *generatorBinomial.py* supports the same generation options as *generatorBitwise.py* but determines the 
number of upsets between seed $i$ and $17-i$ in the first round of the tournament by sampling from a binomial distribution between 0 and 4. Once the number of upsets has been sampled, the regions that will observe the upset are selected randomly with a uniform distribution.
- *generatorConditional.py* can generate brackets with and without triplets and modified first round probabilities but it also supports backwards generation that can result in some bits of the triplets and paths to be locked in. 
Then, the remaining bits of said triplets/paths are filled from a conditional probability table based on the locked in bits.

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
  "format": "[str]",
  "triplets": [
    "str"
  ],
  "non-regional-triplets": [
    "str"
  ],
  "paths": [
    "str"
  ],
  "non-regional-paths": [
    "str"
  ],
  "perturbation": {
    "alphaMLEprobs": "str",
    "rounds": [
      "int"
    ],
    "trunc": "bool",
    "percent": "float",
    "type": "str"
  },
  "generator": "[str]"
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
    - **NCG_E8**: generates a bracket by first determining the NCG seeds, the remaining F4 seeds and the remaining E8 seeds. Note that each E8 seed locks down three bits in the previous rounds. This endModel then locks down 3x8+4+2+1=31 bits.
    This endModel is currently supported by *generatorPower.py* and *generatorConditional.py* only.
- By default, the generators use Maximum Likelihood Estimate winning probabilities. **TODO: add citation** used a simulated annealing algorithm to perturb the MLE probabilities for the first
round of the tournament. The `annealing_model` can be used to tell the generator to use a particular set of these probabilities instead of the MLE probabilities. 
This attribute allows one of ten values: 25_1985, 26_1985, 27_1985, 28_1985, 29_1985, 30_1985, 31_1985, 28_2002, 29_2002, 30_2002.
- `format` determines the outcome encoding used for the vector representation of the bracket. Currently, only **TTT** is supported for both generation and scoring of brackets.
The generators can create brackets using the **FFF** encoding but some of the generators don't support the new scoring function, namely, the *generatorPower.py* and *generatorBradleyTerry.py*. The other scripts fully support FFF but no experiments have been performed with it.
- `triplets` specifies a list of triplet names whose bits will be determined from the distribution of the triplet values and not at a bitwise level. There are seven regional triplets: E8_F4, S16_E8_1, S16_E8_2, R1_R2_1, R1_R2_2, R1_R2_3, and R1_R2_4. 
For more details, see `doc triplets and paths.pdf`.
- `non-regional-triplets` same as `triplets` but for the triplets that involve the last rounds: NCG, R4_R5_1, R4_R5_2.
- `paths` each element is one of 12 possible 3bit paths: P_S1, P_S2, P_S3, P_S4, P_S5, P_S6, P_S7, P_S8, P_R2_1, P_R2_2, P_R2_3, P_R2_4.
- `non-regional-paths` list of 3bit paths from P_R4_R6_1, P_R4_R6_2, P_R4_R6_3, P_R4_R6_4, P_R3_R5_1, P_R3_R5_2, P_R3_R5_3, P_R3_R5_4, P_R3_R5_5, P_R3_R5_6, P_R3_R5_7, P_R3_R5_8.
- `perturbation` is an object that specifies the type of perturbation that can be applied to the Power Model. In particular, two types of perturbation can be used: (i) a perturbation in the 
observed MLE winning probabilities that are used to calculate the Power Model parameters ($alpha$ values), and (ii) the parameter of the truncated geometric distributions used for the 
backward models (i.e., E8, NCG, F4_A, F4_B). In the first case, the attributes **alphaMLEprobs** and **rounds** need to be specified; **alphaMLEprobs** accepts "fixed10" and "fixed20" values and **rounds** can be one or more of the rounds 2 through 6.
In the case of the backwards models, three attributes need to be specified **percent**, **type** and **trunc** (must be set to **true**). The type can be either "fixed" or "rv" (which adds a randomly sampled noise) and **percent** 
specifies the amount of perturbation. For example, **percent**=0.1 and **type**="rv" determines a perturbation around the original probability value that follows a uniform distribution between 0 and 10% of the original probability value.
- `generator` is present in some of the models as a way of documenting which of the generator scripts needs to be used to generate brackets with said model.
It has no other use as of the time of writing.

In addition to the different generators (e.g. *generatorPower.py*), the *generators/modelMixer.py* script can be used to create a new pool of brackets (in fact, only the scores are saved) that contains a mixture of
brackets generated by different models, in equal proportions. For example, the script can be used to create a pool of 1000 brackets where 500 brackets were created using the NCG model and the other 500 were created using the
E8 model. Note that this script does not generate any bracket by itself but rather assumes the brackets have been generated and it samples from the total number of brackets available for each model to create the new pool.
The script follows the same convention of execution:
```
python modelMixer.py <numTrials> <numBatches> <modelsFilename>
```
The files containing the brackets/scores will use the modelsFilename as their own name. For example, if a `ludden.json` file contains the five models proposed by (**TODO: add citation**), and the *modelMixer.py* script is
used to generate a bracket pool mixing brackets from all five models, the pool files will be saved as `generatedScores_ludden_{year}.json`. These files cannot be summarized using the standard *summarizeBracketPools.py* script; instead, 
the *summarizeMixedBracketPools.py* script must be used. Similarly, once the summary is ready, the *prepareMixedPoolsForMCB.py* script must be used to format the data for the MCB analysis. 
Note that *prepareMixedPoolsForMCB.py* is significantly slower than *prepareForMCB.py*; this is due to the calculation of tail probabilities for observing the
Max Score in the mixed pool based on the score distributions of the individual bracket pools.

#### Models
The `generators/models` directory contains the models files for the experiments that were performed between 2017 and 2019. These include the original Power Model experiments, the experiments that uses the simulated annealing first round probabilities, 
the experiments perturbing the later rounds of the Power Model, the experiments using the binomial distribution for the first round upsets, and all the experiments involving triplets and 3bit paths.

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

The *generators/MCB.py* script performs multiple comparison tests for model selection: (i) Multiple Comparison with the Best (MCB), (ii) one-way ANOVA, (iii) Wilcoxon sum of ranks, and (iv) KW H-test.
Additionally, it calculates the summary statistics for the replications of the bracket pool generations and provides utility functions to print the data in LaTex table format. Finally, it creates boxplots comparing the different models.
The current implementation was used to compare the ten settings for the simulated annealing work from (**TODO: add citation**), but it can be modified to compare any given set of models.

#### Utility files
- *generators/utils/extractScores.py* can be used to take one of the bracket pool files that contain the actual brackets and convert it
into the new format that contains only the scores of the brackets.
- *generators/utils/tripletsUniformityTest.py* performs a Chi-square test over the distribution of values of triplets of bits to check whether their distribution seems uniform.
- *generators/utils/preprocessForBradleyTerry.py* takes the historical brackets and creates the summary necessary to fit the Bradley-Terry model (winning records for each seed match-up).
- *generators/utils/isomorphismTest.py* can be used to check whether the vector encoding of a bracket is isomorphic (i.e., the bit distribution is the same across permutations of the regions).
- *generators/utils/tripletDist.py* and *generators/utils/3bitPathDist.py* perform a Chi-square and a 3-way Fisher's exact test over all triplets and 3bit paths, respectively, to
check if the observed distribution of a triplet (or path) is statistically equivalent to the distribution obtained by independently looking at the three bits that constitute the triplet (or path).
The groups of bits analyzed by these scripts are within a single region. It is also of interest to look at triplets or paths that involve the bits after the Final Four round. The analysis for these groups is
done in *generators/utils/nonRegionalTripletDist.py*.
- *generators/utils/upsetDist.py* computes and plots the number of upsets on each of the first round of the tournament.

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