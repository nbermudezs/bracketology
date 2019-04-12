#!/usr/bin/env python3

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import numpy as np
import os
import pandas as pd
import sys
import time
from collections import defaultdict
from itertools import combinations
from triplets.priors.PriorDistributions import read_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FMT = 'TTT'
BASE = 'Likelihood/OptimizeSince2002'
PLOT_TITLE_TEMPLATE = '{}: Dist. of # of matches using P_MLE until {}'
PLOT_FILEPATH_TEMPLATE = BASE + '/Plots/count_dist-p_mle-leq-{}-{}.png'
CSV_FILEPATH_TEMPLATE = BASE + '/Distributions/{}-p_mle-leq-{}-{}.csv'
LOG_TEMPLATE = BASE + '/Logs/energy-and-p_over_time-{}-{}.1.txt'
LOG_BESTS_TEMPLATE = BASE + '/Logs/best_over_time-{}-{}.1.txt'
MIN_CLOSED_FORM_ALPHA = 29
ALL_R1_GAMES = list(range(0, 32))
ROUND_1_INDEXER = np.arange(0, 60) % 15 < 8
N_tau = 20000

uniform_eta_5 = {
    'model10': [ # 25_1985
        1.0,
        0.577924,
        0.738162,
        1.,
        0.750773,
        1.0,
        0.789084,
        1.0
    ],
    'model11': [ # 26_1985
        1.,
        0.492703,
        0.865210,
        1.,
        0.792790,
        1.,
        0.735384,
        1.
    ],
    'model12': [ # 27_1985
        1.,
        0.515851,
        0.767134,
        1.,
        0.659071,
        1.,
        0.624253,
        1.
    ],
    'model13': [ # 28_1985
        1.,
        0.503742,
        0.744503,
        0.985903,
        0.589451,
        0.983175,
        0.629174,
        1.
    ],
    'model1': [ # 29_1985
        1.,
        0.549093,
        0.750055,
        0.934562,
        0.667669,
        0.979045,
        0.654386,
        1.
    ],
    'model14': [ # 30_1985
        1.,
        0.702520,
        0.607727,
        0.955776,
        0.527676,
        0.914587,
        0.731135,
        1.
    ],
    'model15': [ # 31_1985
        0.979950,
        0.522624,
        0.608672,
        0.954450,
        0.815560,
        0.859511,
        0.544364,
        0.986404
    ],
    'model21': [ # 28_2002
        1,
        0.571343,
        0.717079,
        0.951833,
        0.608841,
        1.,
        0.719049,
        1.
    ],
    'model20': [ # 29_2002
        1.0,
        0.642990,
        0.715042,
        0.785525,
        0.673557,
        1.,
        0.829395,
        1.
    ],
    'model23': [ # 30_2002
        1.0,
        0.5160255517843886,
        0.7835308548717009,
        0.8281494585145786,
        0.5682714030506578,
        0.8851682475462623,
        0.5408879753527419,
        0.969572984028147
    ]
}

closed_form_ps = {
    'model0': [
        0.9237013504639907,
        0.8495560921908135,
        0.5311264293573632,
        0.7965369548729775,
        0.7288374967540105,
        0.8054611355579138,
        0.772612240923251,
        0.858300402787333
    ],
    'model1': [
        0.9197261243735679,
        0.8508602315048152,
        0.5332784523933183,
        0.7963364084655133,
        0.7279130141173678,
        0.8164436317928461,
        0.7721898512335448,
        0.8635796079547514
    ],
    'model14': [
        0.8908014837566403,
        0.8253364760259677,
        0.5322097910588279,
        0.7792879559544097,
        0.7144131914540977,
        0.7904877527832495,
        0.7525610649861302,
        0.8369906841452863
    ],
    'model15': [
        0.8656224326500999,
        0.8034569069117807,
        0.5305985899567973,
        0.7640543019770105,
        0.7023032709215519,
        0.7684936066247227,
        0.7355598873197952,
        0.8139368364799848
    ],
    'model16': [
        0.8432209155953801,
        0.7844620548039365,
        0.5294762630138414,
        0.7500897507367827,
        0.6913133469148858,
        0.750016259137182,
        0.720470211533205,
        0.7940374143790992
    ],
    'model20': [
        0.9109937322413584,
        0.8503347302175364,
        0.565933883741003,
        0.7437863758074852,
        0.706003652802423,
        0.8049227953769291,
        0.7862862784719382,
        0.9230889044416283
    ]
}

if not os.path.exists(BASE):
    os.makedirs(BASE)
    os.makedirs(BASE + '/Plots')
    os.makedirs(BASE + '/Logs')
    os.makedirs(BASE + '/Distributions')


std_functions = {
    'std_fn_1': lambda t, T: 0.3 * np.exp((t - T) / T)
}


with open('allBrackets{}.json'.format(FMT)) as f:
    brackets = {
        int(x['bracket']['year']): np.array(list(x['bracket']['fullvector']), dtype=int)
        for x in json.load(f)['brackets']}


def h(round_num):
    if round_num == 1:
        return np.array(list(range(0, 8)))
    elif round_num == 2:
        return np.array(list(range(8, 12)))
    elif round_num == 3:
        return np.array([12, 13])
    elif round_num == 4:
        return np.array([14])
    else:
        return np.array([])


def g(P, trials):
    batch = []
    for trial in range(trials):
        indices = [np.random.choice([0, 1, 2, 3], c, replace=False)
                   for c in np.random.binomial(4, P)]
        for region in range(4):
            bracket = np.array([1 if region in x else 0 for x in indices])
            batch.append(bracket)
    return batch


def estimate_score_distribution(ref, B, indices, trials):
    distribution = defaultdict(lambda: 1)
    _ = [distribution[score] for score in range(0, len(indices) * 4 + 1)]

    for trial in range(trials):
        trial_score = 0
        for region in range(4):
            bracket = B[4 * trial + region]
            trial_score += np.count_nonzero(bracket[indices] == ref[indices + region * 15])
        distribution[trial_score] += 1
    return distribution


def maximum_likelihood(evidence):
    return np.prod(evidence, axis=0)


indices = h(round_num=1)

# np.random.seed(1991)
SAVE_PLOTS = False


def store_plot_and_csv(df, year, name, optimized_years):
    df['sum'] = df.sum(axis=1)
    total = df.sum(axis=0)
    total.name = 'sum'
    df = df.append(total)
    df.to_csv(CSV_FILEPATH_TEMPLATE.format('count_dist', year, name))

    p_df = df / df.loc['sum']
    p_df['sum'] = np.prod(p_df[optimized_years], axis=1)
    p_df.to_csv(CSV_FILEPATH_TEMPLATE.format('all_p_count_dist', year, name))

    df[list(range(year, 2019))].drop('sum').plot.bar()
    plt.title(PLOT_TITLE_TEMPLATE.format(name, year))
    plt.savefig(PLOT_FILEPATH_TEMPLATE.format(year, name))
    plt.close()


def closed_form_eval(omega, psi, alpha):
    omega_R = np.repeat([omega], 4, axis=0).flatten()
    all_p = np.vstack((1 - omega_R, omega_R))
    energy = 0.
    for ref_year in range(psi, 2019):
        year_p = 0
        bracket = brackets[ref_year]
        X_y = bracket[:60][ROUND_1_INDEXER]
        for k in range(alpha, 33):
            year_p += 1./100001
            for c_k in combinations(ALL_R1_GAMES, k):
                p_match = [all_p[X_y[m], m] for m in c_k]
                p_mismatch = [all_p[1 - X_y[n], n]
                              for n in list(set(ALL_R1_GAMES) - set(c_k))]
                year_p += np.prod(p_match) * np.prod(p_mismatch)
        energy += np.log(year_p)
    return -energy


def experiment(P, add_noise, trials, model, current_temp=0, initial_temp=0):
    horizon = 33 - model.get('alpha')
    # print('Using data until year {}'.format(year))
    if add_noise:
        if model.get('annealed_bit') is not None:
            idx = model['annealed_bit']
        else:
            idx = np.random.choice(indices)
        zeros = np.zeros_like(P)
        if model.get('noise') == 'normal':
            if type(model['std']) == str:
                scale = std_functions[model['std']](current_temp, initial_temp)
            else:
                scale = model['std']
            zeros[idx] = np.random.normal(scale=scale)
        elif model.get('noise') == 'uniform':
            margin = model.get('noise_margin')
            low = -margin * P[idx]
            high = margin * P[idx]
            zeros[idx] = np.random.uniform(low, high)
        perturbed_P = np.clip(P + zeros, a_min=0, a_max=1)
    else:
        perturbed_P = P

    if model.get('closed_form', False) and model.get('alpha') >= MIN_CLOSED_FORM_ALPHA:
        energy = closed_form_eval(perturbed_P, model.get('likelihood_start_year'), model.get('alpha'))
        return perturbed_P, energy, None

    B = g(perturbed_P, trials)

    distributions = []
    series = []
    for ref_year in range(model.get('likelihood_start_year'), 2019):
        dist = estimate_score_distribution(brackets[ref_year], B, indices, trials)

        df = pd.DataFrame.from_dict(dist, orient='index').rename({0: 'count'}, axis='columns')
        # df.plot.bar()
        distributions.append((df['count'] / df['count'].sum()).values)
        series.append(pd.Series(df['count'], name=ref_year))

    objective = model.get('objective', 'mle')
    if objective == 'mle':
        mle = maximum_likelihood(distributions)[-horizon:]
        energy = -np.log(sum(mle))
    elif objective == 'sum':
        energy = -np.log(np.sum(distributions, axis=0)[-horizon:].sum())
    else:
        energy = 0
    return perturbed_P, energy, pd.concat(series, axis=1)


def print_setup(bit_P, score_P, old_score_P=None):
    print('{Pr(bit i = 1): 0<= i <= 7} = ', bit_P[indices].tolist())
    print('-log(Pr(Matches >= 29)) = {}'.format(score_P))
    if old_score_P is not None:
        print('Energy difference: {}'.format(np.exp(-old_score_P) - np.exp(-score_P)))
    print('=' * 150)


single_bit_P = [
    1.0,
    0.5928450489751618,
    0.6950938557087003,
    0.9189491423484074,
    0.7747201971300952,
    1,
    0.688728061242975,
    1
]

perturbed_ps = {
    'model10': [
        1.0,
        0.4229103348680731,
        0.8686253918935398,
        1.0,
        0.8123704992250859,
        1.0,
        0.7077991428824237,
        1.0,
    ],
    'model11': [
        1.0,
        0.426407803982116,
        0.8185896601561352,
        1.0,
        0.7023989582786588,
        1.0,
        0.6732989797077819,
        1.0,
    ],
    'model12': [
        1.0,
        0.40952672906312887,
        0.7854648378949232,
        1.0,
        0.7606474009097475,
        1.0,
        0.6539118110780404,
        1.0,
    ],
    'model13': [
        1.0,
        0.47544408691902595,
        0.7559601878920554,
        0.9487586660088168,
        0.7109945653089622,
        1.0,
        0.6877373278227099,
        1.0,
    ],
    'model14': [
        1.0,
        0.6897381544845143,
        0.6574444680210119,
        0.9879908588573346,
        0.6011024159891302,
        0.9211618267386112,
        0.8727621098790732,
        1.0,
    ],
    'model15': [
        0.9926470588235294,
        0.5,
        0.6544117647058824,
        0.7941176470588235,
        0.625,
        0.8455882352941176,
        0.5069725126093771,
        0.9411764705882353,
    ],
    'model20': [
        1.0,
        0.7294731173878907,
        0.6258223531539129,
        0.9164091224773702,
        0.5715239568648995,
        1.0,
        0.7069538395226441,
        1.0,
    ],
    'model21': [
        1.0,
        0.6810812994722260,
        0.6782287991306060,
        1.0,
        0.6065824706992460,
        0.9954042914277010,
        0.6102388677298300,
        1.0
    ],
    'model23': [
        1.0,
        0.7074624431810750,
        0.7470166821001300,
        1.0,
        0.5968548284479815,
        0.9691087987086129,
        0.8588856269144272,
        1.0
    ],
    'model30': [
        1.0,
        0.0,
        0.15639768832340517,
        1.0,
        1.0,
        0.823616960019198,
        0.5363817471415633,
        1.0,
    ]
}


def run_annealing(model, start_year, stop_year):
    optimized_years = list(range(model.get('likelihood_start_year'), 2019))
    # pr = Pr(M_i >= 29)
    for year in np.arange(start=start_year, stop=stop_year + 1, step=1):
        unpooled, pooled = read_data(model['format'], year)
        pooled = pooled.astype(int).values
        P = np.mean(pooled, axis=0)
        # import pdb; pdb.set_trace()
        # P = np.array(perturbed_ps['model23'])

        # P = np.array([1.0, 0.5928450489751618, 0.6950938557087003, 0.9189491423484074, 0.721206790987267, 1.0, 0.7345229047483148, 1.0, 0.0, 1.0, 1.0, 1.0, 0.8031569810310453, 0.9244015862122775, 1.0])

        print('=' * 50 + 'Baseline for data before {}'.format(year) + '=' * 50)
        prev_P, prev_pr, count_df = experiment(P, add_noise=False, trials=100000, model=model)
        store_plot_and_csv(count_df, year - 1, 'model23_uni5', optimized_years=optimized_years)
        # exit(1)

        if not model.get('closed_form', False):
            store_plot_and_csv(count_df, year - 1, 'model23', optimized_years=optimized_years)
        # this is used to set a single bit to the optimal value we found so far
        # P[0] = 1.0
        # P[1] = 0.4231184017779921
        # P[2] = 0.752286265633227
        # P[3] = 0.9491817654669973
        # P[4] = 0.721206790987267
        # P[5] = 1.0
        # P[6] = 0.7345229047483148
        # P[7] = 1.0

        # prev_P = np.array([ 1.        ,  0.49166667,  0.63333333,  0.79166667,  0.65833333,
        #     0.85      ,  0.60833333,  0.94166667,  0.86666667,  0.49166667,
        #     0.475     ,  0.35      ,  0.76666667,  0.40833333,  0.60833333])
        # prev_pr = 258.53736133404186
        print_setup(prev_P, prev_pr)

        pr_0 = prev_pr
        best_pr = prev_pr

        T_0 = model.get('initial_temp', 6.)
        t = T_0

        print('Logs: ' + LOG_TEMPLATE.format(model['name'], year))
        print('Best P Logs: ' + LOG_BESTS_TEMPLATE.format(model['name'], year))
        f = open(LOG_TEMPLATE.format(model['name'], year), 'w')
        best_log = open(LOG_BESTS_TEMPLATE.format(model['name'], year), 'w')
        counter = 0
        while t >= 0:
            print('=' * 50 + 'Temperature: {}'.format(t) + '=' * 50)

            favorable_count = 0
            for trial_at_fixed_temperature in range(10000):
                print('t={:05d}'.format(trial_at_fixed_temperature), end='\r')
                new_P, new_pr, count_df = experiment(prev_P, add_noise=True, trials=10000, model=model, current_temp=t, initial_temp=T_0)

                accept_p = 1 if new_pr < prev_pr else min(1., np.exp(-(new_pr - prev_pr) / t))
                if np.random.rand() < accept_p:
                    f.write('{},{}: {}\n'.format(counter, new_pr,
                                                 ','.join(new_P[:8].astype(str))))
                    prev_P = new_P
                    prev_pr = new_pr
                    favorable_count += 1

                    print('t={0:05d} - ACCEPTED. Energy: {1}'.format(trial_at_fixed_temperature, new_pr))
                    if new_pr < best_pr:
                        print_setup(new_P, new_pr, pr_0)
                        best_pr = new_pr
                        if not model.get('closed_form', False):
                            store_plot_and_csv(count_df, year - 1, model['name'], optimized_years=optimized_years)
                        best_log.write('{},{}: {}\n'.format(counter, new_pr, ','.join(new_P[:8].astype(str))))
                    if favorable_count == 50:
                        break
                counter += 1
                if model.get('cooling_decay', 1) != 1 and counter >= N_tau:
                    break
            f.flush()
            if model['cooling'] == 'linear':
                t = (t - model.get('cooling_delta', 0.)) * model.get('cooling_decay', 1.)
            if model.get('cooling_decay', 1) != 1 and counter >= N_tau:
                break


if __name__ == '__main__':
    # import cProfile, pstats
    # from io import StringIO
    # pr = cProfile.Profile()
    # pr.enable()

    model_filepath = sys.argv[1]
    start_year = int(sys.argv[2])
    stop_year = int(sys.argv[3])
    if len(sys.argv) == 5:
        selected_model = sys.argv[4]
    else:
        selected_model = None

    with open(model_filepath, 'r') as f:
        models = json.load(f)['models']

    for model in models:
        if selected_model is not None and selected_model != model['name']:
            continue
        run_annealing(model, start_year, stop_year)

    # pr.disable()
    # s = StringIO()
    # ps = pstats.Stats(pr).sort_stats('cumulative')
    # ps.print_stats()
