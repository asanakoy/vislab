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


def fetch_basic_dataset(args=None):
    """
    Fetch basic info and page urls of all artworks by crawling search
    results.  Results are returned as a DataFrame.

    Not parallelized.
    """
    _args = dict(request_interval=0.3)
    if args is not None:
        _args.update(args)

    artist_slugs_with_cent_df = _fetch_all_artist_slugs()
    # TODO: For debug only
    artist_slugs_with_cent_df.to_hdf(os.path.expanduser(
        '~/tmp/wikiart/wikiart_artist_slugs.hdf5'), 'df', mode='w')

    artist_slugs = artist_slugs_with_cent_df.index.values
    print 'Fetching paintings urls'
    all_links = []
    for artist_idx, slug in enumerate(artist_slugs):
        sys.stdout.write('\rArtist {:04d}/{}'.format(artist_idx, len(artist_slugs)))
        sys.stdout.flush()

        relative_page_urls = \
                _get_paintings_relative_urls_by_artist_broot(artist_idx, len(artist_slugs),
                                                     slug, _args['request_interval'])
        all_links.extend(relative_page_urls)
        time.sleep(_args['request_interval'])

        # TODO: for debug only. REMOVE
        if artist_idx % 200 == 0:
            print 'Saving df snapshot'
            tmp_df = _slugs_to_df(all_links, artist_slugs_with_cent_df)
            tmp_df.to_hdf(os.path.expanduser('~/tmp/wikiart/wikiart_basic_info_{}_artists.hdf5'
                                             .format(artist_idx)), 'df', mode='w')
    print ''
    # remove duplicates
    all_links = list(set(all_links))

    # Turn URLs into image ids and get other basic info.
    df = _slugs_to_df(all_links, artist_slugs_with_cent_df)
    return df


def fetch_detailed_dataset(args=None):
    """
    Fetch detailed info by crawling the detailed artwork pages, using
    the links from the basic dataset.

    Parallelized with vislab.utils.distributed.map_through_rq.
    """

    print("Fetching detailed Wikipaintings dataset by scraping artwork pages.")

    if args is None:
        args = {
            'force_dataset': False,
            'force_basic': False,
            'num_workers': 1, 'mem': 2000,
            'cpus_per_task': 1, 'async': True,
            'chunk_size': 10
        }

    basic_df = vislab.datasets.wikipaintings.get_basic_df(force=args.pop('force_basic'), args=args)

    db = vislab.util.get_mongodb_client()[vislab.datasets.wikipaintings.DB_NAME]
    collection = db['image_info']
    print("Old collection size: {}".format(collection.count()))

    force = args['force_dataset']
    if not force:
        # Exclude ids that were already computed.
        image_ids = basic_df.index.tolist()
        image_ids = vislab.util.exclude_ids_in_collection(
            image_ids, collection)
        basic_df = basic_df.loc[image_ids]

    # Chunk up the rows.
    rows = [row.to_dict() for ind, row in basic_df.iterrows()]
    chunk_size = args['chunk_size']
    num_chunks = len(rows) / chunk_size
    if num_chunks == 0 and len(rows) > 0:
        num_chunks = 1

    if num_chunks == 0:
        chunks = []
    else:
        chunks = np.array_split(rows, num_chunks)
    args_list = [(chunk.tolist(), force) for chunk in chunks]

    # Work the jobs.
    vislab.utils.distributed.map_through_rq(
        vislab.datasets.wikiart_scraping._fetch_artwork_infos,
        args_list, 'wikipaintings_info',
        num_workers=args['num_workers'], mem=args['mem'],
        cpus_per_task=args['cpus_per_task'], async=args['async'])
    print("Final collection size: {}".format(collection.count()))

    # Assemble into DataFrame to return.
    # Drop artworks without an image.
    orig_df = pd.DataFrame([doc for doc in collection.find()])
    df = orig_df.dropna(subset=['image']).copy()

    # Rename some columns and add an index.
    new_column_names = {'image': 'image_url',
                        'locationCreated': 'location_created',
                        'dateCreated': 'date'}
    df.columns = [new_column_names.pop(col, col) for col in df.columns]

    print '---- Columns: ', df.columns.values
    df.index = pd.Index(df['image_id'], name='image_id')

    # Only take useful columns.
    columns_to_take = [
        'image_id', 'artist_slug', 'artwork_slug', 'date',
        'genre', 'style', 'keywords', 'name',
        'page_url', 'image_url', 'description',
        'location_created', 'media', 'location'
    ]
    df = df[columns_to_take]
    print '---- Columns took: ', df.columns.values

    #  NOTE: some image urls can be unicode!

#    # Drop artworks with messed up image urls
#    good_inds = []
#    for ind, row in df.iterrows():
#        try:
#            str(row['image_url'])
#            good_inds.append(ind)
#        except:
#            pass
#    df = df.ix[good_inds]
#    df['image_url'] = df['image_url'].apply(lambda x: str(x))

    return df


def _image_url_2_page_relative_url(url, artist_slug):
    """
    :param url: Full url to the image
    :return: relative url of the painting page '/en/artist_slug/painting_slug'
    """
    import re
    res = re.match('http://uploads\d+.wikiart.org/images/.+/(.*)\.jpg!.*', url)
    if res is not None:
        url = '/en/' + artist_slug + '/' + res.groups()[0]
    else:
        print '\nWARNING! url does not match pattern:', url
        url = ''
    return url


def _get_paintings_relative_urls_by_artist_broot(artist_idx, total_num_artists, artist_slug,
                                                 request_interval):
    """
    Find all paintings urls looking through all possible pages with paintings
    on the artist's web-page.
    """
    all_links = []
    artist_url = 'http://www.wikiart.org/en/{}/mode/all-paintings?json=2&page={:d}'
    max_pages_num = 3000
    page_num = 0
    while True:
        if page_num == max_pages_num:
            sys.stderr.write('Breaking for artist {} due to reaching max page number: {}'
                             .format(artist_slug, max_pages_num))
            print 'Max page number {} reached. Break'.format(max_pages_num)
            break

        sys.stdout.write(
            '\r{:04d}/{} : page {}'.format(artist_idx, total_num_artists, page_num))
        sys.stdout.flush()
        json = _get_response(artist_url.format(artist_slug, page_num)).json()
        if len(json['Paintings']) == 0:
            # last page reached
            break

        relative_page_urls = _get_artworks_links_from_json(json)
        all_links.extend(relative_page_urls)
        page_num += 1
        time.sleep(request_interval)
    print ''
    return all_links


def _get_paintings_relative_urls_by_artist(artist_slug):
    # WARNING: Give not correct results! painting-slug is different from image name!
    raise Exception('deprecated!')
    request_url = 'http://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl={}&json=2'
    json = vislab.datasets.wikiart_scraping._get_response(request_url.format(artist_slug)).json()
    urls = [_image_url_2_page_relative_url(json[i]['image'], artist_slug) for i in xrange(len(json))]
    urls = [x for x in urls if len(x) > 0]
    return urls


def _get_artist_slugs(url):
    json = _get_response(url).json()
    # slugs for the pages of the authors
    urls = [json[i]['url'].lower() for i in xrange(len(json))]
    urls = [x for x in urls if len(x) > 0]
    return urls


def _fetch_all_artist_slugs():
    century_url = 'http://www.wikiart.org/en/artists-by-century/{}/0'
    json_list_url = 'http://www.wikiart.org/en/App/Artist/AlphabetJson?inPublicDomain={}'
    urls = [(century_url.format(century), century) for century in xrange(8, 22)]
    urls.extend([(json_list_url.format(public_domain), None) for public_domain in ['true', 'false']])

    all_artists = []
    centuries = dict()
    for url, century in urls:
        print 'Fetching url: {}'.format(url)
        try:
            slugs = _get_artist_slugs(url)
            all_artists.extend(slugs)
            if century is not None:
                for slug in slugs:
                    centuries[slug] = century
        except Exception as e:
            sys.stderr.write('EXCEPTION: {}'.format(e))
            print e
        time.sleep(0.6)

    all_artists = list(set(all_artists))
    df = pd.DataFrame(index=all_artists, data={'century': [centuries.get(x, 'None') for x in all_artists]})
    return df


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


def _slugs_to_df(all_slugs, artist_slugs_with_cent_df):
    """
    Turn URLs' slugs into image ids and get other basic info.
    """
    df = pd.DataFrame([
                          {
                              'page_url': 'http://www.wikiart.org' + slug,
                              'image_id': slug.replace('/en/', '').replace('/', '_'),
                              'artist_slug': slug.split('/')[-2],
                              'artwork_slug': slug.split('/')[-1]
                          } for slug in all_slugs
                          ])
    df['image_id'] = df['image_id'].apply(unidecode)
    df['artist_slug'] = df['artist_slug'].apply(unidecode)
    df['artwork_slug'] = df['artwork_slug'].apply(unidecode)
    df.index = pd.Index(df['image_id'], name='image_id')

    df.join(artist_slugs_with_cent_df, how='left')
    return df


def _get_artworks_links_from_json(json):
    paintings = json['Paintings']
    links = [paintings[i]['paintingUrl'] for i in xrange(len(paintings))]
    return links


def _fetch_artwork_infos(image_ids_and_page_urls, force=False):
    """
    Fetch artwork info, including image url, from the artwork page for
    each of the given image_ids, storing the obtained info to DB.
    """
    collection = vislab.util.get_mongodb_client()[vislab.datasets.wikipaintings.DB_NAME]['image_info']
    collection.ensure_index('image_id')

    for row in image_ids_and_page_urls:
        if not force:
            # Skip if image exists in the database.
            cursor = collection.find({'image_id': row['image_id']})
            if cursor.limit(1).count() > 0:
                continue

        # Get detailed info for the image.
        info = _fetch_artwork_info(row['image_id'], row['page_url'])
        info.update(row)

        collection.update(
            {'image_id': info['image_id']}, info, upsert=True)
        print('inserted {}'.format(info['image_id']))
        time.sleep(0.1)


def _fetch_artwork_info(image_id, page_url):
    """
    Scrape the artwork info page for relevant properties to return dict.
    """
    r = _get_response(page_url)
    soup = bs4.BeautifulSoup(r.text, "lxml")
    info = {}
    for tag in soup.find_all(lambda _tag: _tag.has_attr('itemprop')):
        itemprop_name = tag.attrs['itemprop']
        if itemprop_name == 'keywords':
            keywords = [x.strip().strip(',').lower() for x in tag.strings]
            print keywords
            keywords = [x for x in keywords if x != '']
            info[itemprop_name] = keywords
        elif tag.name == 'img':
            # itemprop='image'
            info[itemprop_name] = tag['src'].split('!')[0]
        else:
            # TODO: parse itemprop='name' differnetly,
            # as threre are 2 names: for artist and for artwork
            info[itemprop_name] = unidecode(tag.text.strip().lower())

    for tag in soup.find_all('div', attrs={'class': 'info-line'}):
        strings = [unidecode(x).lower() for x in tag.stripped_strings]
        if len(strings) == 0:
            continue
        if strings[0] == 'style:':
            info['style'] = '$'.join(strings[1:])
        elif strings[0] == 'media:':
            info['media'] = map(lambda s: s.strip(','), strings[1:])
            info['media'] = [x for x in info['media'] if len(x) > 0]
        elif strings[0] == 'location:':
            info['location'] = '$'.join(strings[1:])

    return info
