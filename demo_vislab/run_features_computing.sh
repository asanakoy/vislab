# pushd .
# cd ~/workspace/vlg_extractor/data
# # bash data/download_parameters.sh
# unzip parameters_1.1.zip
# popd
export PYTHONPATH=PYTHONPATH:/export/home/asanakoy/workspace/vislab-repo
source /export/home/asanakoy/workspace/vislab-repo/venv/bin/activate
PYTHON_BIN='python -u'
$PYTHON_BIN vislab/feature.py compute --dataset=wikipaintings --features=mc_bit --num_workers=2 --cpus_per_task=1 --mem=9000
$PYTHON_BIN vislab/feature.py cache_to_h5 --dataset=wikipaintings --features=mc_bit --force_features
$PYTHON_BIN vislab/feature.py cache_to_vw --dataset=wikipaintings --features=mc_bit --force_features
#$PYTHON_BIN vislab/predict.py predict --dataset=wikipaintings --collection_name=wikipaintings_mar23 --prediction_label="style_*" --features=mc_bit --mem=9000 --num_workers=8
