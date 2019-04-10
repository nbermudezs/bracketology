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

#### Bracket Pool Generators
The *generators* directory contains scripts to generate bracket pools using various approaches:
- *generatorPower.py* supports the generation of bracket pools using one of the five original Power Model variations proposed by (**TODO: add citation**).


### Contribution Guidelines
**...TODO...**