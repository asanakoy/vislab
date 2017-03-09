import requests
import time
import vislab.util
import vislab.utils.distributed
import vislab.datasets
import os
import sys
import pandas as pd
import numpy as np
import bs4
from unidecode import unidecode
import deepdish.io as dio


def _get_response(url, retry_timeout=1, retry_counts=2):
    """
    :param url:
    :param retry_timeout: retry timeout will be increased by 2 each next trial
    :param retry_counts:
    :return:
    """
    assert retry_timeout >= 0
    try:
        r = requests.get(url, headers=vislab.util.get_mozila_request_header())
        r.raise_for_status()
        return r
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        if retry_counts > 0:
            print '{}.\nRetrying after {} s timeout. {} trials left.' \
                .format(e, retry_timeout, retry_counts)
            time.sleep(retry_timeout)
            return _get_response(url, retry_timeout=2 * retry_timeout,
                                 retry_counts=retry_counts - 1)
        else:
            sys.stderr.write('Exception during get request {}\n{}'.format(url, e))
            raise e
    except Exception as e:
        sys.stderr.write('Exception during get request {}\n{}'.format(url, e))
        print 'Exception during get request {}\n{}'.format(url, e)


def get_list_of_items(strings):
    items = map(lambda s: s.strip(','), strings)
    items = [x for x in items if len(x) > 0]  # filter out empty strings
    return items


def fetch_artist_info(page_url):
    """
    Scrape the artwork info page for relevant properties to return dict.
    """
    r = _get_response(page_url)
    if isinstance(r.text, basestring):
        # convert to unicode if not unicode
        if not isinstance(r.text, unicode):
            r.text = unicode(r.text, encoding='utf-8')

    soup = bs4.BeautifulSoup(r.text, "lxml")
    info = {}
    itemprop_names = ['deathDate', 'birthPlace', 'deathPlace', 'deathDate', 'name', 'additionalName']
    for tag in soup.find_all(lambda _tag: _tag.has_attr('itemprop')):
        itemprop_name = tag.attrs['itemprop']

        strings = [x.lower() for x in tag.stripped_strings]
        if len(strings) == 0:
            continue
        if itemprop_name in itemprop_names:
            info[itemprop_name] = '$'.join(strings)
        elif itemprop_name not in ['sameAs', 'image']:
            info[itemprop_name] = tag.text.strip().lower()

    info_names = ['nationality:', 'art movement:', 'painting school:', 'genre:', 'field:',
                     'influenced by:', 'influenced on:', 'art institution:',
                     'friends and co-workers:', 'wikipedia:']
    for tag in soup.find_all('div', attrs={'class': 'info-line'}):
        strings = [x.lower() for x in tag.stripped_strings]
        if len(strings) == 0:
            continue
        if strings[0] in info_names:
            prop_name = strings[0].strip(':').replace(' ', '_')
            info[prop_name] = get_list_of_items(strings[1:])

    return info


def get_artist_info(artist_slug):
    url_prefix = u'https://www.wikiart.org/en/'
    return (artist_slug, fetch_artist_info(url_prefix + artist_slug))


if __name__ == '__main__':
    info = pd.read_hdf('/export/home/asanakoy/workspace/wikiart/info/info.hdf5')
    artist_names = np.unique(info['artist_slug'])
    artist_names = [y if isinstance(y, unicode) else unicode(y, encoding='utf-8')
                    for y in artist_names]
    print 'Artists num:', len(artist_names)

    # artist_info = list()
    # save_every = 100
    # url_prefix = u'https://www.wikiart.org/en/'
    args_list = [(artist_slug,) for artist_slug in artist_names]

    redis_host = '0.0.0.0'
    redis_port = 6379
    artist_infos = vislab.utils.distributed.map_through_rq(
            'vislab.datasets.wikiart_scrap_artist_info.get_artist_info',
            args_list,
            'wikiart_artists_info',
            num_workers=4, async=True,
            host=redis_host, port=redis_port,
            aggregate=True)

    print 'Saving'

    dio.save('/export/home/asanakoy/workspace/wikiart/info/artists_info_dict.hdf5',
             {'info_lsit': artist_infos})
    artists_df = pd.DataFrame.from_dict(artist_infos, orient='index')
    artists_df.to_hdf('/export/home/asanakoy/workspace/wikiart/info/artists_info.hdf5')
