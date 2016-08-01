
# coding: utf-8

# In[1]:

# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
import sys
sys.path.append('/export/home/asanakoy/workspace/vislab')

import re
import sklearn.metrics
import pandas as pd
import numpy as np

import vislab
import vislab.results
import vislab._results
import vislab.datasets


DATA_NAME = 'data'
is_force = True
is_balanced_test = True
# In[2]:

label_df = vislab.datasets.wikipaintings.get_style_df()


# In[3]:
db_name = 'predict'
print 'Current collections in DB.{}: {}'\
    .format(db_name, vislab.util.get_mongodb_client()[db_name].collection_names())
collection_name = 'wikipaintings_mar23'
c = vislab.util.get_mongodb_client()[db_name][collection_name]
# if c.find({'features': 'noise'}).count() > 0:
#     c.remove({'features': 'noise'})
print 'Current documents in DB.{}.{}:'.format(db_name, collection_name)
print pd.DataFrame([x for x in c.find()])


# In[4]:

results_dirname = vislab.util.makedirs(vislab.config['paths'][DATA_NAME] + '/results')
assert DATA_NAME != 'shared_data' or not is_force
results_df, preds_panel = vislab._results.load_pred_results(
    'wikipaintings_mar23', results_dirname, multiclass=True, force=is_force)
pred_prefix = 'pred'

if DATA_NAME == 'shared_data':
    preds_panel.minor_axis = [_.replace('caffe', 'alexnet') for _ in preds_panel.minor_axis]
print preds_panel.minor_axis


# In[5]:

cache_filename = '{}/{}_thresholds_and_accs.h5'.format(results_dirname, collection_name)
# get per_class_acc_df for some random balanced subset
threshold_df, per_class_acc_df = vislab.results.learn_accuracy_thresholds_for_preds_panel(
    preds_panel, cache_filename, force=is_force, balanced=True)
# del per_class_acc_df['noise None vw']


# In[6]:

per_class_acc_df.index = [_.replace('style_', '') for _ in per_class_acc_df.index]
per_class_acc_df.columns = [_.replace(' ', '_') for _ in per_class_acc_df.columns]
# acc_df.sort('MC-bit accuracy')
mean_per_class_acc_df = pd.DataFrame(per_class_acc_df.mean().to_dict(), index=['_mean'])
per_class_acc_df = per_class_acc_df.append(mean_per_class_acc_df)

# In[7]:
print '=================================='
print 'per class Accuracy on random binary balanced subset'
print per_class_acc_df
# print acc_df.sort('MC-bit accuracy').to_latex()


# In[8]:

feat_to_evaluate = [
    # u'noise None vw',
    u'rs_balance_iter191268_2_sum_pool None vw',
    u'rs_balance_iter191268_2_max_pool None vw',
    u'rs_balance_iter191268 None vw',
    u'rs_iter131934 None vw',
    u'alexnet_fc6 None vw',
    u'alexnet_fc7 None vw',
    u'alexnet_prob None vw',
    u'mc_bit None vw',
    u'decaf_imagenet None vw',
    ###
    u'decaf_fc6 None vw',
    u'decaf_tuned_fc6 False vw',
    u'decaf_fc6,pascal_mc_for_decaf_fc6 pd vw',
    u'decaf_tuned_fc6_flatten False vw',
    u'decaf_tuned_fc6_ud None vw',
    u'fusion_wikipaintings_oct25 None vw',
    u'fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw'
]
preds_panel = preds_panel.select(lambda x: x in feat_to_evaluate, 'minor')


# In[9]:

nice_feat_names = {
    'rs_balance_iter191268 None vw': 'rs_balance_iter191268',
    'rs_balance_iter191268_2_sum_pool None vw': 'rs_balance_iter191268_2_sum_pool',
    'rs_balance_iter191268_2_max_pool None vw': 'rs_balance_iter191268_2_max_pool',
    'rs_iter131934 None vw': 'rs_iter131934',
    'alexnet_fc6 None vw': 'alexnet_FC6',
    'alexnet_fc7 None vw': 'alexnet_FC7',
    'alexnet_prob None vw': 'alexnet_Prob',
    'mc_bit None vw': 'MC-bit',
    'decaf_imagenet None vw': 'ImageNet',
    ###
    'decaf_fc6 None vw': 'DeCAF_6',
    'decaf_fc6_flatten False vw': 'DeCAF_5',
    'decaf_tuned_fc6 False vw': 'Fine-tuned DeCAF_6',
    'fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw': 'Late-fusion x Content'
}

mc_metrics = vislab.results.multiclass_metrics_feat_comparison(
    preds_panel, label_df, pred_prefix,
    # features=['random'],
    features=preds_panel.minor_axis.tolist(),
    balanced=is_balanced_test, with_plot=False, with_print=True, nice_feat_names=nice_feat_names)


# In[13]:

# conf_df = mc_metrics['feat_metrics']['fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw']['conf_df'].astype(float)
# conf_df.index = [x.replace('style_', '') for x in conf_df.index]
# conf_df.columns = [x.replace('style_', '') for x in conf_df.columns]
# fig = vislab.dataset_viz.plot_conditional_occurrence(conf_df, sort_by_prior=False, font_size=16)
# fig.savefig('/Users/sergeyk/work/aphrodite-writeup/figures/evaluation/wikipaintings_conf.pdf', bbox_inches='tight')


# In[14]:

per_class_acc_df = mc_metrics['acc_df']
fig = vislab.results_viz.plot_top_k_accuracies(per_class_acc_df, font_size=16)
fig.savefig(vislab.config['paths'][DATA_NAME] + '/results/figures/wikipaintings_top_k.pdf', bbox_inches='tight')


# In[15]:

ap_df = mc_metrics['ap_df']
if 'noise None vw' in ap_df:
    del ap_df['noise None vw']
column_order = ap_df.columns[(-ap_df.ix['_mean']).argsort().values]
ap_df.index = [x.replace('style_', '') for x in ap_df.index]
ap_df = ap_df.reindex_axis(column_order, axis=1)
# ap_df.to_csv(vislab.config['paths'][DATA_NAME] + '/results/figures/wikipaintings_ap_df.csv')
# fig = vislab.results_viz.plot_df_bar(ap_df, fontsize=14)
# fig.savefig(vislab.config['paths'][DATA_NAME] + '/results/figures/wikipaintings_ap_barplot.pdf', bbox_inches='tight')

# In[16]:

print '=================================='
print 'Average Precision {}'.format('BALANCED' if is_balanced_test else '')
print ap_df


# In[10]:

# del ap_df['random']
# print ap_df.to_latex(float_format=lambda x: '%.3f'%x if not np.isnan(x) else '-')


# In[ ]:



