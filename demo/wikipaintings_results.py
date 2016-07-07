
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


# In[2]:

label_df = vislab.datasets.wikipaintings.get_style_df()


# In[3]:

print vislab.util.get_mongodb_client()['predict'].collection_names()
c = vislab.util.get_mongodb_client()['predict']['wikipaintings_mar23']
# if c.find({'features': 'noise'}).count() > 0:
#     c.remove({'features': 'noise'})
pd.DataFrame([x for x in c.find()])


# In[4]:

results_dirname = vislab.util.makedirs(vislab.config['paths']['shared_data'] + '/results')
# results_dirname = vislab.util.makedirs(vislab.config['paths']['data'] + '/results')
results_df, preds_panel = vislab._results.load_pred_results(
    'wikipaintings_mar23', results_dirname, multiclass=True, force=False)
pred_prefix = 'pred'
print preds_panel.minor_axis


# In[5]:

collection_name = 'wikipaintings_mar23'
cache_filename = '{}/{}_thresholds_and_accs.h5'.format(results_dirname, collection_name)
threshold_df, acc_df = vislab.results.learn_accuracy_thresholds_for_preds_panel(
    preds_panel, cache_filename)
del acc_df['noise None vw']
acc_df.columns = ['MC-bit accuracy']


# In[6]:

acc_df.index = [_.replace('style_', '').replace('_', ' ') for _ in acc_df.index]
acc_df.sort('MC-bit accuracy')


# In[7]:

print acc_df.sort('MC-bit accuracy').to_latex()


# In[8]:

feat_to_evaluate = [
    u'noise None vw',
    u'decaf_fc6 False vw',
    u'caffe_fc7 None vw',
    #u'decaf_fc6,pascal_mc_for_decaf_fc6 pd vw',
    u'decaf_tuned_fc6 False vw',
    #u'decaf_tuned_fc6_flatten False vw',
    #u'decaf_tuned_fc6_ud None vw',
    #u'fusion_wikipaintings_oct25 None vw',
    u'fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw',
    u'mc_bit False vw',
    u'decaf_imagenet None vw'
]
preds_panel = preds_panel.select(lambda x: x in feat_to_evaluate, 'minor')


# In[9]:

nice_feat_names = {
    'decaf_fc6 False vw': 'DeCAF_6',
    'caffe_fc7 None vw': 'alexnet_FC7',
    'decaf_fc6_flatten False vw': 'DeCAF_5',
    'decaf_tuned_fc6 False vw': 'Fine-tuned DeCAF_6',
    'fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw': 'Late-fusion x Content',
    'mc_bit False vw': 'MC-bit',
    'decaf_imagenet None vw': 'ImageNet'
}

mc_metrics = vislab.results.multiclass_metrics_feat_comparison(
    preds_panel, label_df, pred_prefix,
    # features=['caffe_fc7 None vw'] + ['random_{}'.format(i) for i in xrange(2, 7)],
    features=preds_panel.minor_axis.tolist() + ['random'],
    # features=['caffe_fc7 None vw'],
    balanced=True, with_plot=False, with_print=True, nice_feat_names=nice_feat_names)


# In[13]:

# conf_df = mc_metrics['feat_metrics']['fusion_wikipaintings_oct25,pascal_mc_for_fusion_wikipaintings_oct25 fp vw']['conf_df'].astype(float)
# conf_df.index = [x.replace('style_', '') for x in conf_df.index]
# conf_df.columns = [x.replace('style_', '') for x in conf_df.columns]
# fig = vislab.dataset_viz.plot_conditional_occurrence(conf_df, sort_by_prior=False, font_size=16)
# fig.savefig('/Users/sergeyk/work/aphrodite-writeup/figures/evaluation/wikipaintings_conf.pdf', bbox_inches='tight')


# In[14]:

acc_df = mc_metrics['acc_df']
fig = vislab.results_viz.plot_top_k_accuracies(acc_df, font_size=16)
fig.savefig(vislab.config['paths']['shared_data'] + '/results/figures/evaluation/wikipaintings_top_k.pdf', bbox_inches='tight')


# In[15]:

ap_df = mc_metrics['ap_df']
column_order = ap_df.columns[(-ap_df.ix['_mean']).argsort().values]
ap_df.index = [x.replace('style_', '') for x in ap_df.index]
ap_df = ap_df.reindex_axis(column_order, axis=1)
ap_df.to_csv(vislab.config['paths']['shared_data'] + '/results/figures/wikipaintings_ap_df.csv')
fig = vislab.results_viz.plot_df_bar(ap_df, fontsize=14)
fig.savefig(vislab.config['paths']['shared_data'] + '/results/figures/wikipaintings_ap_barplot.pdf', bbox_inches='tight')


# In[16]:

print ap_df


# In[10]:

# del ap_df['random']
# print ap_df.to_latex(float_format=lambda x: '%.3f'%x if not np.isnan(x) else '-')


# In[ ]:



