"""
Copyright Sergey Karayev - 2013.

Scrape wikipaintings.org to construct a dataset.
"""
import os
import pandas as pd
import vislab
import vislab.dataset
import vislab.util
from wikiart_scraping import fetch_basic_dataset, fetch_detailed_dataset

DB_NAME = 'wikipaintings'

_DIRNAME = vislab.config['paths']['shared_data']
_SHARED_KARAYEV_DIR = vislab.config['paths']['shared_data_karayev']
BASIC_DF_FILENAME = _DIRNAME + '/wikipaintings_basic_info.h5'
DETAILED_DF_FILENAME = _DIRNAME + '/wikipaintings_detailed_info.h5'
KARAYEV_DETAILED_DF_FILENAME = _SHARED_KARAYEV_DIR + '/wikipaintings_detailed_info.h5'
URL_DF_FILENAME = _DIRNAME + '/wikipaintings_urls.h5'

underscored_style_names = [
    'style_Abstract_Art',
    'style_Abstract_Expressionism',
    'style_Art_Informel',
    'style_Art_Nouveau_(Modern)',
    'style_Baroque',
    'style_Color_Field_Painting',
    'style_Cubism',
    'style_Early_Renaissance',
    'style_Expressionism',
    'style_High_Renaissance',
    'style_Impressionism',
    'style_Magic_Realism',
    'style_Mannerism_(Late_Renaissance)',
    'style_Minimalism',
    'style_Nave_Art_(Primitivism)',
    'style_Neoclassicism',
    'style_Northern_Renaissance',
    'style_Pop_Art',
    'style_Post-Impressionism',
    'style_Realism',
    'style_Rococo',
    'style_Romanticism',
    'style_Surrealism',
    'style_Symbolism',
    'style_Ukiyo-e'
]


def get_image_url_for_id(image_id):
    filename = URL_DF_FILENAME
    if not os.path.exists(filename):
        df = get_df()
        dfs = df['image_url']
        dfs.to_hdf(filename, 'df', mode='w')
    else:
        dfs = pd.read_hdf(filename, 'df')
    return dfs.ix[image_id]


def get_basic_df(force=False, args=None):
    """
    Return DataFrame of image_id -> detailed artwork info, including
    image URLs.
    """
    return vislab.util.load_or_generate_df(
        BASIC_DF_FILENAME, fetch_basic_dataset, force, args)


def get_df(force=False, args=None):
    """
    Return DataFrame of image_id -> detailed artwork info, including
    image URLs.
    Only keep those artworks with listed style and genre.
    """
    df = vislab.util.load_or_generate_df(
        DETAILED_DF_FILENAME, fetch_detailed_dataset, force=force, args=args)
    return df


def get_style_karayev_df():
    """
    Load detailed df that was used by karayev in his experiments for
    style classification with 25 classes and >= 1000 samples in each of them.
    (Wikipainting from Sep 2013)
    """
    df = pd.read_hdf(KARAYEV_DETAILED_DF_FILENAME, 'df')
    df = df.dropna(how='any', subset=['genre', 'style'])
    return _get_column_label_df(df, 'style', min_positive_examples=1000)


def get_style_df(min_positive_examples=1000, force=False):
    df = get_df(force)
    return _get_column_label_df(df, 'style', min_positive_examples)


def get_genre_df(min_positive_examples=1000, force=False):
    df = get_df(force)
    return _get_column_label_df(df, 'genre', min_positive_examples)


def get_artist_df(min_positive_examples=200, force=False):
    df = get_df(force)
    df['artist'] = df['artist_slug']
    return _get_column_label_df(df, 'artist', min_positive_examples)


def _get_column_label_df(
        df, column_name, min_positive_examples):
    bool_df = vislab.dataset.get_bool_df(
        df, column_name, min_positive_examples)
    bool_df['_split'] = vislab.dataset.get_train_test_split(bool_df)
    bool_df['image_url'] = df['image_url']
    return bool_df
