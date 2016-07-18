#!/bin/bash -x
export PYTHONPATH=PYTHONPATH:/export/home/asanakoy/workspace/vislab-repo
source /export/home/asanakoy/workspace/vislab-repo/venv/bin/activate
PYTHON_BIN='python -u'
# $PYTHON_BIN vislab/feature.py compute --dataset=wikipaintings --features=alexnet_fc7 --num_workers=1 --cpus_per_task=1 --mem=9000
# $PYTHON_BIN vislab/feature.py cache_to_h5 --dataset=wikipaintings --features=alexnet_fc7 --force_features
# $PYTHON_BIN vislab/feature.py cache_to_vw --dataset=wikipaintings --features=alexnet_fc7 --force_features
$PYTHON_BIN vislab/predict.py predict --dataset=wikipaintings --collection_name=wikipaintings_mar23 --prediction_label="style_*" --features=alexnet_fc7 --mem=9000 --num_workers=8 --force_predict
