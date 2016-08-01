import vislab
import vislab.dataset
import requests
import os
import time
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'basic':
        df = vislab.datasets.wikipaintings.get_basic_df(force=True)
        # print vislab.datasets.wikipaintings._get_names_from_url('http://www.wikiart.org/en/artists-by-century/8/0')
        # print vislab.datasets.wikipaintings.fetch_basic_dataset()
    elif sys.argv[1] == 'detailed':
        args = {
            'force_dataset': False,
            'force_basic' : False,
            'num_workers': 1, 'mem': 2000,
            'cpus_per_task': 1, 'async': True,
            'chunk_size': 1
        }
        df = vislab.datasets.wikipaintings.get_df(force=True, args=args)
        print 'Num entries:', len(df)
    else:
        raise ValueError('unknown arg!')
