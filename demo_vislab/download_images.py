import vislab
import vislab.dataset
import os
import pandas as pd

def replace_old_links():
    df = vislab.datasets.wikipaintings.get_df()
    df['image_url'] = df['image_url'].str.replace('wikipaintings.org', 'wikiart.org', 1)
    out_path = os.path.join(vislab.config['paths']['shared_data'], 'wikipaintings_detailed_info_truncated.hdf5')
    df.to_hdf(out_path, 'df')
    print 'replaced wikipaintings.org -> wikiart.org and stored in {}'.format(out_path)

def main():
    info = pd.read_hdf('/export/home/asanakoy/workspace/wikiart/info/info_joined.hdf5')
    info.index = info['image_id']
    df = info
    # df = vislab.datasets.wikipaintings.get_df()
    # index_to_download = df.index.difference(info.index)
    # df = df.loc[index_to_download]

    print 'Total images in data frame: {}'.format(len(df))
    good_filenames = vislab.dataset.fetch_image_filenames_for_ids(df.index, 'wikipaintings_all')
    print 'Total images in data frame: {}'.format(len(df))
    print 'Total images downloaded: {}'.format(len(good_filenames))

    if len(good_filenames) < len(df):
        print 'Not all the images were downloaded! Try one more time!'

if __name__ == '__main__':
    main()
