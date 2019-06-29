from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from scipy.stats import f_oneway, kruskal, mannwhitneyu
import seaborn as sns

sns.set_palette('dark')
plt.style.use("seaborn-white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150


N = 13 # number of sampled replications
R = 25 # number of replications
root = 'forPaper'
treatments = [
    '25_1985',
    '26_1985',
    '27_1985',
    '28_1985',
    '29_1985',
    '30_1985',
    '31_1985',
    '28_2002',
    '29_2002',
    '30_2002'
]


data = {
    'score': dict(),
    'count': dict()
}

metric_factor = {
    'score': {
        2013: 1,
        2014: 1,
        2015: 1,
        2016: 1,
        2017: 1,
        2018: 1,
        2019: 1
    },
    'count': {
        2013: 10 * 8.15,
        2014: 10 * 11.,
        2015: 10 * 11.6,
        2016: 10 * 13.,
        2017: 10 * 18.8,
        2018: 10 * 17.3,
        2019: 10 * 17.2
    }
}


def process_file(f, has_summary_stats):
    result = {}
    for year in range(2013, 2020):
        year_result = []
        next(f)  # skip year in header
        next(f)  # skip the "Batch" delimiter
        for _ in range(R):
            replication = next(f).replace('\n', '').split(',')
            observations = np.array(replication[1:-1])
            if observations.shape[0] > 5:
                observations = observations[[0, 2, 4, 6, 8]].astype(float)
            else:
                observations = observations.astype(float)
            year_result.append(observations)
        if has_summary_stats:
            next(f)  # remove line with means
            next(f)  # remove line with std
        next(f)  # empty line at the end of each year
        result[year] = np.vstack(year_result).T
    return result


def prepare_data():
    for metric in data.keys():
        with open('{}/{}-baseline.csv'.format(root, metric)) as f:
            baseline = process_file(f, has_summary_stats=False)
        for treatment in treatments:
            print(treatment)
            print('=' * 80)
            with open('{}/{}-{}.csv'.format(root, metric, treatment)) as f:
                treatment_results = process_file(f, has_summary_stats=True)
                improvement = {}
                for year in range(2013, 2020):
                    # print('diff', treatment_results[year] - baseline[year])
                    if metric == 'score':

                        print(year, (treatment_results[year] - baseline[year]).sum(axis=1))
                        # import pdb; pdb.set_trace()
                    improvement[year] = (treatment_results[year] - baseline[year]).sum(axis=0) * metric_factor[metric][year]
                data[metric][treatment] = improvement


def unbiased_std(matrix):
    t, n = matrix.shape
    nu = t * (n - 1)
    y_bar = np.mean(matrix, axis=1)[:, np.newaxis]
    return np.sqrt(((matrix - y_bar) ** 2.).sum() / nu)


def build_treatment_matrix(metric, year):
    result = []
    for treatment in treatments:
        result.append(data[metric][treatment][year])
    return np.vstack(result)


def mcb(matrix):
    h = 2.56 # Table from textbook; 9 degrees of freedom, n = 25, alpha = 0.05
    # samples = matrix[:, np.random.choice(matrix.shape[1], N)]
    samples = np.vstack([x[np.random.choice(matrix.shape[1], N)] for x in matrix])
    # import pdb; pdb.set_trace()
    s_v = unbiased_std(samples)
    print('s_v', s_v)
    y_bar = np.mean(samples, axis=1)
    intervals = []
    for i in range(y_bar.shape[0]):
        mu_i = y_bar[i]
        not_indices = np.setxor1d(np.indices(y_bar.shape), [i])
        others = y_bar[not_indices]
        # import pdb; pdb.set_trace()
        intervals.append((
            int(mu_i - np.max(others) - h * s_v * np.sqrt(2. / N)),
            int(mu_i - np.max(others) + h * s_v * np.sqrt(2. / N))
        ))
    return intervals


def mwu(matrix):
    """
        Performs pairwise Mann-Whitney U tests on the matrix of 
        sample measurements for the 10 treatments. 
    """
    t, n = matrix.shape
    for i in range(t):
        pValues = []
        for j in range(t):
            _, p = mannwhitneyu(matrix[i, :], matrix[j, :], alternative='two-sided')
            pValues.append(np.round(p, 6))
        print(treatments[i] + ' & ' + ' & '.join(np.array(pValues, dtype=str)))


def anova(matrix):
    samples = []
    for i in range(matrix.shape[0]):
        # if i != 0 and i != 3:
        #     continue
        samples.append(matrix[i, :])
    f_value, p_value = f_oneway(*samples)
    print('p: {}'.format(p_value))


def kruskal_wallis(matrix):
    samples = []
    for i in range(matrix.shape[0]):
        samples.append(matrix[i, :])
    f_value, p_value = kruskal(*samples)
    print('p: {}'.format(p_value))


def boxplots(matrix):
    fig, ax = plt.subplots()
    ax.boxplot(matrix.T)
    ax.set_xticklabels(treatments, fontsize=8)
    plt.title('Comparison of max score improvement for different treatments')
    plt.xlabel('Treatment')
    plt.ylabel('Improvement in max score')
    plt.tight_layout()
    # plt.show()
    plt.savefig('tmp.png', dpi=200)



def as_latex(matrix):

    means = matrix.mean(axis=1)[:, np.newaxis]
    std = matrix.std(ddof=0, axis=1)[:, np.newaxis]
    phi = np.count_nonzero(matrix > 0, axis=1)[:, np.newaxis]
    labels = np.array([x.replace('_', '\_') for x in treatments])[:, np.newaxis]
    all_data = np.concatenate((
        labels,
        # matrix.astype(int).astype(str),
        means.astype(int).astype(str),
        std.astype(int).astype(str),
        phi.astype(int).astype(str)), axis=1).astype(str)
    # data = np.concatenate((labels, all_data[:, 15:]), axis=1)
    # import pdb; pdb.set_trace()
    data = all_data
    lines = [' & '.join(x) for x in data]
    for line in lines:
        print(line + '\\\ \hline')


def main(_):
    prepare_data()
    matrix = np.zeros((len(treatments), R))

    for year in range(2013, 2020):
        metric = 'count' # 'score' or 'count'
        matrix += build_treatment_matrix(metric, year)
    as_latex(matrix)
    # exit(0)
    np.set_printoptions(linewidth=2000)
    print(matrix)
    print('MEANS', matrix.mean(axis=1).round())
    print('S', matrix.std(axis=1, ddof=1).round(2))

    # print('One-way ANOVA')
    # anova(matrix)
    # print('-' * 120)

    # Kruskal-Wallis test to see whether there is 
    # at least one pair with significant difference
    print('Kruskal-Wallis H-test')
    kruskal_wallis(matrix)
    print('-' * 120)

    # Pairwise Mann-Whitney U tests to see which treatments are better
    print('Pairwise Mann-Whitney U tests')
    mwu(matrix)
    print('-' * 120)

    # boxplots(matrix)

    # MCB analysis for given metric
    print('MCB intervals:')
    intervals = mcb(matrix)
    pprint(intervals)
    print('-' * 120)


if __name__ == '__main__':
    main(0)
