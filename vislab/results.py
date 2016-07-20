"""
Code for parsing results of prediction tasks.
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import sklearn.metrics
import vislab
import vislab.results_viz
import vislab.dataset_viz

# Canonical prediction prefix.
pred_prefix = 'pred_'


def get_balanced_dataset_ind(df_, gt_col, random_seed=0):
    """
    Return integer index of the subset of df_ that has a balanced set of
    True/False values for gt_col.
    """
    pos_ind = np.where(df_[gt_col])[0]
    neg_ind = np.where(np.equal(df_[gt_col], False))[0]
    np.random.seed(random_seed)
    if pos_ind.shape[0] > neg_ind.shape[0]:
        ind = np.hstack((
            np.random.choice(pos_ind, size=neg_ind.shape[0], replace=False), neg_ind))
    else:
        ind = np.hstack((
            pos_ind, np.random.choice(neg_ind, size=pos_ind.shape[0], replace=False)))
    return ind


def pred_accuracy_at_threshold(
        df_, gt_col, threshold, verbose=False, balanced=True):
    """
    Calculate accuracy at threshold.
    [default] For random balanced subset.

    TODO: average over several balanced splits.

    Parameters
    ----------
    df_: pandas.DataFrame
    gt_col: string
    threshold: float
        Threshold for positive predictions.
    verbose: boolean
        Print classification report if true.
    balanced : boolean
        Should we take a random balanced by binary label subset for accuracy calculation?

    Returns
    -------
    acc: float
        Accuracy on df_[gt_col] using
        df_[pred_prefix + gt_col] > threshold.
    """
    if balanced:
        ind = get_balanced_dataset_ind(df_, gt_col)
    else:
        ind = np.arange(len(df_), dtype=int)
    gt = df_[gt_col].values[ind].astype(bool)
    preds = (df_[pred_prefix + gt_col] > threshold).values[ind]
    acc = sklearn.metrics.accuracy_score(gt, preds)
    if verbose:
        print sklearn.metrics.classification_report(gt, preds)
    return acc


def learn_accuracy_threshold(
        df_, gt_col,
        thresholds=np.hstack((np.linspace(0, 1, num=20), -np.logspace(-2, 0, 20), np.logspace(-2, 0, 20), 1 - np.logspace(-2, 0, 20), np.logspace(-2, 0, 20) - 1)),
        # thresholds=np.logspace(-2, 0, 20) - 1,
        balanced=True):
    """
    Do cross-validation of thresholds for prediction accuracy.

    Parameters
    ----------
    df_: pandas.DataFrame
        Must have gt_col and pred_prefix + gt_col columns.
    gt_col: string
        Name of the boolean column we care about.
    thresholds: iterable of float
    balanced : bool
        Should we take a random balanced by binary label subset for accuracy calculation?

    Returns
    -------
    best_threshold: float
    accs: list of float
        Accuracies for the thresholds considered.
    """
    accs = [
        pred_accuracy_at_threshold(
            df_, gt_col, threshold,
            verbose=False, balanced=balanced)
        for threshold in thresholds
    ]
    best_threshold = thresholds[np.argmax(accs)]
    return best_threshold, accs


def learn_accuracy_thresholds_for_preds_panel(
        preds_panel, cache_filename=None, force=False, balanced=True):
    """
    Find the positive prediction thresholds that maximize accuracy
    on the validation set for all settings and labels in preds_panel,
    returning a DataFrame of thresholds and a DataFrame of test-set
    accuracies.

    Basically calculates per-class accuracies for every setting.

    Parameters
    ----------
    preds_panel: pandas.Panel
        Such as is loaded by

    balanced : boolean
        For each class: Should we take a random,
        balanced by binary label, subset for accuracy calculation?

    Returns
    -------
    threshold_df: pandas.DataFrame
    acc_df: pandas.DataFrame
    """
    # names of the collections containing df in HDF5 cache file
    threshold_df_name = 'threshold_df' + ('' if balanced else '_nonbalanced')
    acc_df_name = 'acc_df' + ('' if balanced else '_nonbalanced')

    if cache_filename is not None and os.path.exists(cache_filename) and not force:
        threshold_df = pd.read_hdf(cache_filename, threshold_df_name)
        acc_df = pd.read_hdf(cache_filename, acc_df_name)
        return threshold_df, acc_df

    thresholds = defaultdict(dict)
    test_accs = defaultdict(dict)

    for setting_name in preds_panel.minor_axis:
        pred_df = preds_panel.minor_xs(setting_name)
        pred_df_val = pred_df[pred_df['split'] == 'val']
        pred_df_test = pred_df[pred_df['split'] == 'test']

        pred_cols = [_ for _ in pred_df.columns if _.startswith(pred_prefix)]
        gt_cols = [_.replace(pred_prefix, '') for _ in pred_cols]

        for gt_col in gt_cols:
            best_threshold, val_accs = learn_accuracy_threshold(
                pred_df_val, gt_col, balanced=balanced)

            thresholds[setting_name][gt_col] = best_threshold

            test_accs[setting_name][gt_col] = pred_accuracy_at_threshold(
                pred_df_test, gt_col, best_threshold, balanced=balanced)
            sys.stdout.write('.')
        sys.stdout.write('\n')

    threshold_df = pd.DataFrame(thresholds)
    acc_df = pd.DataFrame(test_accs)

    if cache_filename is not None:
        threshold_df.to_hdf(cache_filename, threshold_df_name, mode='w')
        acc_df.to_hdf(cache_filename, acc_df_name, mode='a')

    return threshold_df, acc_df


def regression_metrics(
        pred_df, name='', balanced=True, with_plot=False, with_print=False):
    """
    TODO: write docstring
    """
    y_true = pred_df['label']
    y_pred = pred_df['pred']

    metrics = {}
    metrics['r2_score'] = sklearn.metrics.r2_score(y_true, y_pred)
    metrics['mse'] = sklearn.metrics.mean_squared_error(y_true, y_pred)

    return metrics


def binary_metrics(
        pred_df, name='', balanced=False, with_plot=False, with_print=False):
    """
    Binary classification metrics.

    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must contain 'label' (int or bool) and 'pred' (float) columns.
    name: string
        Name of the classification task: for example, the class name.
    balanced: bool
        If True, the evaluation considers a class-balanced subset of
        the dataset.
    with_plot: bool
        If True, plot curves and return handles to figures (otherwise
        return handles to None).
    with_print: bool
    -----------
    Return: dict metrics
    metrics['auc'] = area under ROC curve
    metrics['ap'] = average precision on binary task
    ...
    all other metrics will be computed using cut-off threshold between classes = 0
    (not very adequate).
    """
    name = '{} balanced'.format(name) if balanced else name

    # Drop rows without a label and make sure it's a bool.
    pred_df = pred_df.dropna(subset=['label'])
    pred_df['label'] = pred_df['label'] > 0

    # The prediction we loaded is a raw score: convert to binary prediction.
    pred_df['pred_bin'] = pred_df['pred'] > 0

    # Get a balanced subset if requested.
    if balanced:
        # Find the most frequent label
        counts = pred_df['label'].value_counts()
        mfl = counts.index[0]

        # Subsample to match the number of the other label.
        ind = np.random.permutation(counts.max())[:counts.min()]
        mfl_pred_df = pred_df[pred_df['label'] == mfl]
        lfl_pred_df = pred_df[~(pred_df['label'] == mfl)]
        pred_df = (mfl_pred_df.iloc[ind]).append(lfl_pred_df)

    # Compute metrics.
    metrics = {}
    y_true = pred_df['label'].values
    y_pred_raw = pred_df['pred'].values
    y_pred_bin = pred_df['pred_bin'].values

    results = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred_bin)

    metrics['results_df'] = pd.DataFrame(
        np.array(results).T,
        columns=[['precision', 'recall', 'f1-score', 'support']],
        index=[['False', 'True']])

    metrics['mcc'] = sklearn.metrics.matthews_corrcoef(
        y_true, y_pred_bin)

    metrics['accuracy'] = sklearn.metrics.accuracy_score(
        y_true, y_pred_bin)

    metrics['pr_fig'], \
    prec, \
    rec, \
    metrics['ap'] = \
        get_pr_curve(y_true, y_pred_raw, name, with_plot)

    metrics['ap_sklearn'] = sklearn.metrics.average_precision_score(
        y_true, y_pred_raw)

    metrics['roc_fig'], fpr, tpr, metrics['auc'] = \
        get_roc_curve(y_true, y_pred_raw, name, with_plot)

    if with_print:
        print_metrics(metrics, name.format(name))

    return metrics


def multiclass_metrics_feat_comparison(
        preds_panel, source_label_df, pred_prefix, features,
        balanced=False, with_plot=False, with_print=False,
        nice_feat_names=None):
    """
    Multiclass classification metrics for a set of feature channels.

    Parameters
    ----------
    features: sequence of string
        If includes feature 'random', also evaluate chance performance.
    """
    feat_metrics = {}
    for feature in features:
        # To evaluate chance performance, we need to pass any feature
        # channel with the random_preds flag so that they get replaced.
        if feature == 'random':
            actual_feature = features[0]
            random_preds = True
        else:
            actual_feature = feature
            random_preds = False

        # Need the feature channel predictions, and true labels.
        mc_pred_df = preds_panel.minor_xs(actual_feature)

        try:
            mc_pred_df = mc_pred_df.join(source_label_df)
        except ValueError:
            print("Looks like the preds frame already has gt info.")

        if '_split' in mc_pred_df.columns:
            print("Only taking 'test' split predictions.")
            mc_pred_df = mc_pred_df[mc_pred_df['_split'] == 'test']
        elif 'split' in mc_pred_df.columns:
            print("Only taking 'test' split predictions.")
            mc_pred_df = mc_pred_df[mc_pred_df['split'] == 'test']
        else:
            print("WARNING: no split info in the preds panel.")

        print('*' * 20 + feature + '*' * 20)
        feat_metrics[feature] = multiclass_metrics(
            mc_pred_df, pred_prefix, balanced, random_preds,
            with_plot, with_print, feature_name=feature)

    all_metrics = {'feat_metrics': feat_metrics}

    # Across-feature AP comparison.
    ap_df = pd.DataFrame(dict(
        (feature, feat_metrics[feature]['binary_metrics_df']['ap'])
        for feature in features
    ))
    if nice_feat_names is not None:
        ap_df.columns = [
            nice_feat_names[x] if x in nice_feat_names else x
            for x in ap_df.columns
        ]
    all_metrics['ap_fig'] = vislab.results_viz.plot_df_bar(ap_df)
    all_metrics['ap_df'] = ap_df

    # # Across-feature top-k accuracy comparison.
    acc_df = pd.DataFrame(
        [feat_metrics[f]['top_k_accuracies'] for f in features],
        index=features
    ).T
    if nice_feat_names is not None:
        acc_df.columns = [
            nice_feat_names[x] if x in nice_feat_names else x
            for x in acc_df.columns
        ]
    all_metrics['top_k_fig'] = vislab.results_viz.plot_top_k_accuracies(acc_df)
    all_metrics['acc_df'] = acc_df

    return all_metrics


def multiclass_metrics(
        mc_pred_df, pred_prefix, balanced=True, random_preds=False,
        with_plot=False, with_print=False, min_pos=20, feature_name='', seed=42):
    """
    Multiclass classification metrics for a single set of predictions.

    Parameters
    ----------
    mc_pred_df: pandas.DataFrame
        Has columns for True/False labels, and same columns,
        with a prefix, with float pred values.
    pred_prefix: string
        The prefix before the name of a pred-containing column.
    balanced: bool [False]
        If True, the number if instances of each class is equalized.
    random_preds: bool [False]
        If True, preds are replaced with random values.
        (For seeing random performance).
    with_plot: bool [False]
        If True, also plot and return figures.
    with_print: bool [False]
        If True, print the metrics.
    min_pos: int [20]
        Minimum number of positive examples needed to evaluate metrics.

    feature_name: just for debug purposes
    """
    metrics = {}

    # Get the list of labels.
    pred_cols, label_cols = map(list, zip(*[
        (col, col.replace(pred_prefix + '_', ''))
        for col in mc_pred_df.columns
        if col.startswith(pred_prefix)
    ]))

    # Make two dataframes: for labels and predictions.
    label_df = mc_pred_df[label_cols]
    pred_df = mc_pred_df[pred_cols]

    # Get rid of those labels with less than min_pos examples
    good_cols = label_df.sum(0) >= min_pos
    good_cols = good_cols[good_cols].index.tolist()
    good_pred_cols = [pred_prefix + '_' + x for x in good_cols]
    label_df = label_df[good_cols]
    pred_df = pred_df[good_pred_cols]
    label_cols = label_df.columns.tolist()

    # Drop those rows that don't have a single positive label.
    ind = label_df.sum(1) > 0
    label_df = label_df[ind]
    pred_df = pred_df[ind]

    assert np.all(label_df.sum(1) > 0)
    assert np.all(label_df.sum(1) == 1)

    # Get vector of multi-class labels.
    # This deals with multiple things being true by picking just one.
    # NOTE: assumes that labels are 0-index ints
    # NOTE: if random is unseeded, this can lead to slight change in
    # printed numbers, as different bases of support are used.
    y_true = []
    np.random.seed(seed)
    for row in label_df.values:
        ind = np.where(row)[0]
        y_true.append(ind[np.random.randint(len(ind))])
        # y_true.append(ind[0])
    y_true = np.array(y_true)

    # Balance the labels.
    if balanced:
        np.random.seed(seed)
        counts = label_df.sum(0).astype(int)
        min_count = counts[counts.argmin()] + 1
        permutation = lambda N, K: np.random.permutation(N)[:K]
        selected_ind = np.unique(np.concatenate([
            np.where(label_df[label])[0][permutation(count, min_count)]
            for label, count in counts.iteritems()
        ]))
        y_true = y_true[selected_ind]
        pred_df = pred_df.iloc[selected_ind]
        label_df = label_df.iloc[selected_ind]
        # np.save('/export/home/asanakoy/workspace/wikiart/karayev/test_balanced_seed{}.npy'
        #   .format(seed), label_df.index.values)

    if random_preds:
        np.random.seed(None)
        rand_values = np.random.rand(*pred_df.values.shape)
        pred_df = pd.DataFrame(rand_values, pred_df.index, pred_df.columns)

    # Get binary metrics for all classes.
    all_binary_metrics = {}
    for i, label in enumerate(label_cols):
        pdf = pd.DataFrame(
            {
                'pred': pred_df['{}_{}'.format(pred_prefix, label)],
                'label': label_df[label]
            },
            pred_df.index
        )
        all_binary_metrics[label] = binary_metrics(
            pdf, label, False, False, False)
    # bin_df will AP, ROC-AUC - adequately
    # accuracy, mcc and several other metrics for unbalanced binary problem with threshold 0.
    bin_df = pd.DataFrame(all_binary_metrics).T

    # Add mean of the numeric metrics.
    mean_df = pd.DataFrame(
        bin_df[['accuracy', 'ap', 'auc', 'mcc']].mean().to_dict(),
        index=['_mean']
    )
    bin_df = bin_df.append(mean_df)

    metrics['binary_metrics_df'] = bin_df

    # # Plot binary metrics.
    # metrics['binary_metrics_fig'] = None
    # if with_plot:
    #     metrics['binary_metrics_fig'] = vislab.results_viz.plot_df_bar(
    #         metrics['binary_metrics_df'], ['ap', 'mcc'])

    # Get vector of multi-class preds.
    y_pred = pred_df.values.argmax(1)

    # Construct the confusion matrix.
    conf_df = pd.DataFrame(
        sklearn.metrics.confusion_matrix(y_true, y_pred),
        columns=label_cols, index=label_cols)

    # Confusion matrix and accuracy.
    metrics['conf_df'] = conf_df
    total = conf_df.sum().sum()
    metrics['conf_df_n'] = conf_df.astype(float) / total
    # overall accuracy for all classes jointly
    metrics['overall_accuracy'] = np.diagonal(conf_df).sum().astype(float) / total

    # multiclass P-R-F1-Support table.
    results = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred)
    metrics['results_df'] = pd.DataFrame(
        np.array(results).T,
        columns=[['precision', 'recall', 'f1-score', 'support']],
        index=label_cols)

    # Plot confusion table.
    metrics['confusion_table_fig'] = None
    if with_plot:
        metrics['confusion_table_fig'] = \
            vislab.dataset_viz.plot_conditional_occurrence(
                metrics['conf_df_n'], sort_by_prior=False)

    # Compute top-k accuracies.
    ordered_preds = np.argsort(-pred_df.values, axis=1)
    Y = np.cumsum(ordered_preds == y_true[:, np.newaxis], axis=1)
    metrics['top_k_accuracies'] = pd.Series(
        [
            round(float(Y[:, k].sum()) / Y.shape[0], 3)
            for k in range(Y.shape[1])
        ],
        np.arange(Y.shape[1]) + 1
    )

    # # Plot top-k accuracies.
    # top_k = min(5, Y.shape[1])
    # metrics['top_k_accuracies_fig'] = None
    # if with_plot:
    #     metrics['top_k_accuracies_fig'] = \
    #         vislab.results_viz.plot_top_k_accuracies(
    #             metrics['top_k_accuracies'], top_k)

    # Print report.
    if with_print:
        print 'Multiclass metrics'
        name = '{} balanced' if balanced else '{}'
        print_metrics(metrics, name)

    return metrics


def print_metrics(metrics, name):
    print('-' * 60)
    if len(name) > 0:
        name = 'on {}'.format(name)
    print("Classification metrics {}".format(name))
    for metric_name, value in metrics.iteritems():
        if metric_name == 'results_df':
            print(value.to_string())
        metrics_to_print = [
            'overall_accuracy', 'mcc', 'ap', 'ap_sklearn'
        ]
        if metric_name in metrics_to_print:
            print('{}: {}'.format(metric_name, value))
    print('')


def get_roc_curve(y_true, y_score, title=None, with_plot=True):
    """
    Plot the [Receiver Operating Characteristic][roc] curve of the given
    true labels and confidence scores.

    [roc]: http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    auc = np.trapz(tpr, fpr)
    fig = None
    if with_plot:
        fig = vislab.results_viz.plot_curve_with_area(
            fpr, tpr, auc, 'False Positive Rate', 'True Positive Rate', 'AUC')
        ax = fig.get_axes()[0]
        ax.plot([0, 1], [0, 1], 'k--')
        if title is not None:
            ax.set_title(title)
    return fig, fpr, tpr, auc


def get_pr_curve(y_true, y_score, title=None, with_plot=True):
    """
    Plot Precision-Recall curve of the true labels and confidence scores
    and return the precision and recall vectors and the average
    precision.

    Returns
    -------
    fig: plt.Figure
    prec: ndarray
    rec: ndarray
    ap: float
    """
    prec, rec, thresh = sklearn.metrics.precision_recall_curve(y_true, y_score)
    # Make sure prec is non-increasing (prec is in reverse order)
    # WARNING: This is not right! Karayev cheated!
    # for i in range(len(prec) - 1):
    #     prec[i + 1] = max(prec[i + 1], prec[i])
    ap = np.trapz(-prec, rec)
    fig = None
    if with_plot:
        fig = vislab.results_viz.plot_curve_with_area(
            rec, prec, ap, 'Recall', 'Precision', 'AP', title)
    return fig, prec, rec, ap
