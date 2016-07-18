import vislab
import vislab.dataset
import os

def replace_old_links():
    df = vislab.datasets.wikipaintings.get_df()
    df['image_url'] = df['image_url'].str.replace('wikipaintings.org', 'wikiart.org', 1)
    out_path = os.path.join(vislab.config['paths']['shared_data'], 'wikipaintings_detailed_info_truncated.hdf5')
    df.to_hdf(out_path, 'df')
    print 'replaced wikipaintings.org -> wikiart.org and stored in {}'.format(out_path)

def main():
    df = vislab.datasets.wikipaintings.get_df()
    # import pandas as pd
    # df = pd.read_hdf(os.path.join(vislab.config['paths']['shared_data'],
    # 'wikipaintings_detailed_info_truncated_diff.hdf5'))
    good_filenames = vislab.dataset.fetch_image_filenames_for_ids(df.index, 'wikipaintings_all')
    print 'Total images in data frame: {}'.format(len(df))
    print 'Total images downloaded: {}'.format(len(good_filenames))

    if len(good_filenames) < len(df):
        print 'Not all the images were downloaded! Try one more time!'

if __name__ == '__main__':
    main()
