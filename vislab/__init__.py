import os
import json
repo_dirname = os.path.abspath(os.path.dirname(__file__))

# Load config file.
with open(repo_dirname + '/config.json', 'r') as f:
    config = json.load(f)
for path_name, path in config['paths'].iteritems():
    config['paths'][path_name] = os.path.expanduser(path)

# make peredict_temp dir process specific
config['paths']['predict_temp'] += '_{}'.format(os.getpid())

import sys
sys.path.append(os.path.join(config['paths']['caffe'], 'python'))

import util
# import feature
