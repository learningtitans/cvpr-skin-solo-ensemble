"""
Create ensembles by averaging predictions.

Create three types of ensembles:

1) Average models with same architecture and same split.
2) Average models with different architectures and same split, at random.
3) Average models with different architectures and same split, sorted by
   validation AUC.

Outputs to results/ensemble_{1,2,3}.csv
"""

from random import shuffle

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             average_precision_score,
                             confusion_matrix)


def calculate_metrics(preds):
    labels_array = preds['label'].values.astype(int)
    scores_array = preds['score'].values.astype(float)
    auc = roc_auc_score(labels_array, scores_array)
    acc = accuracy_score(
        labels_array, np.where(scores_array >= 0.5, 1, 0))
    avp = average_precision_score(labels_array, scores_array)
    conf_matrix = confusion_matrix(labels_array, scores_array >= 0.5)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)

    return {'auc': auc, 'acc': acc, 'avp': avp,
            'spec': specificity, 'sens': sensitivity}


def ensemble(ids, epoch='best'):
    if epoch == 'best+last':
        dfs = [pd.read_csv(
            'results/{}/test-aug-best.csv'.format(i)) for i in ids]
        dfs += [pd.read_csv(
            'results/{}/test-aug-last.csv'.format(i)) for i in ids]
    else:
        dfs = [pd.read_csv(
            'results/{}/test-aug-{}.csv'.format(i, epoch)) for i in ids]
    df_concat = pd.concat(dfs)
    df_means = df_concat.groupby('image').mean()
    return df_means


def first_ensemble(data):
    ensemble_ids = {}
    ensemble_1 = pd.DataFrame(columns=('split', 'net', 'epoch', 'auc', 'acc',
                                       'avp', 'spec', 'sens'))

    for epoch in ('best', 'last', 'best+last'):
        for split in range(1, 6):
            ensemble_ids[split] = {}
            for net in data.name.unique():
                models = data.loc[
                    ((data['name'] == net) & (data['split'] == split))]
                ensemble_ids[split][net] = list(models['id'])
                ensemble_1 = ensemble_1.append(
                    {'net': net, 'split': split, 'epoch': epoch,
                     **calculate_metrics(
                         ensemble(list(models['id']), epoch=epoch))
                     },
                    ignore_index=True)

    return ensemble_1


def second_ensemble(data):
    ensemble_2 = pd.DataFrame(columns=('split', 'n_models', 'epoch', 'auc',
                                       'acc', 'avp', 'spec', 'sens'))

    for _ in range(10):
        for epoch in ('best', 'last'):
            for split in range(1, 6):
                filtered = data.loc[(data['split'] == split)]
                ids = list(filtered['id'])
                shuffle(ids)
                for i in range(1, len(ids)+1):
                    ensemble_2 = ensemble_2.append(
                        {'n_models': i, 'split': split, 'epoch': epoch,
                         **calculate_metrics(ensemble(ids[0:i], epoch=epoch))},
                        ignore_index=True)

    return ensemble_2


def third_ensemble(data):
    ensemble_3 = pd.DataFrame(columns=('split', 'n_models', 'epoch', 'auc',
                                       'acc', 'avp', 'spec', 'sens'))

    for epoch in ('best', 'last'):
        for split in range(1, 6):
            filtered = data.loc[(data['split'] == split)]
            filtered = filtered.sort_values('best_val_auc', ascending=False)
            ids = list(filtered['id'])
            for i in range(1, len(ids)+1):
                ensemble_3 = ensemble_3.append(
                    {'n_models': i, 'split': split, 'epoch': epoch,
                     **calculate_metrics(ensemble(ids[0:i], epoch=epoch))},
                    ignore_index=True)

    return ensemble_3


def main():
    results = pd.read_csv('results_github.csv', header=None)
    results.columns = pd.read_csv('columns.csv').columns
    ensemble_1 = first_ensemble(results)
    ensemble_2 = second_ensemble(results)
    ensemble_3 = third_ensemble(results)

    ensemble_1.to_csv('results/ensemble_1.csv', index=False)
    ensemble_2.to_csv('results/ensemble_2.csv', index=False)
    ensemble_3.to_csv('results/ensemble_3.csv', index=False)


if __name__ == '__main__':
    main()
