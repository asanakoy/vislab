import copy
import vislab
import vislab.predict
import sys


def correct_columns_order(df):
    df = df.copy()
    style_columns = [u'style_Impressionism', u'style_Realism', u'style_Romanticism',
                     u'style_Post-Impressionism', u'style_Expressionism',
                     u'style_Surrealism', u'style_Art_Nouveau_(Modern)',
                     u'style_Symbolism',
                     u'style_Baroque', u'style_Neoclassicism',
                     u'style_Northern_Renaissance',
                     u'style_Nave_Art_(Primitivism)', u'style_Abstract_Expressionism',
                     u'style_Rococo', u'style_Cubism', u'style_Early_Renaissance',
                     u'style_High_Renaissance', u'style_Minimalism',
                     u'style_Color_Field_Painting', u'style_Mannerism_(Late_Renaissance)',
                     u'style_Ukiyo-e', u'style_Pop_Art', u'style_Abstract_Art',
                     u'style_Magic_Realism', u'style_Art_Informel', u'split']
    assert sorted(df.columns.values) == sorted(style_columns)
    df = df[style_columns]
    return df



# predict --dataset=wikipaintings --source_dataset=wikipaintings --features=noise --collection_name=demo_predict_only --prediction_label="style_*" --num_workers=1
if __name__ == '__main__':
    # Extract hdf used by Karayev as train/test split on wikipaintings
    sys.argv = 'script.py predict --dataset=wikipaintings --source_dataset=wikipaintings --features=noise --collection_name=demo_predict_only --prediction_label="style_*" --num_workers=1'.split()
    args = vislab.utils.cmdline.get_args(
        __file__, 'predict',
        ['dataset', 'prediction', 'processing', 'feature'])

    if args.source_dataset is not None:
        ## Get the source and target datasets as specified in args.

        # First get the source dataset.
        args_copy = copy.deepcopy(args)
        args_copy.dataset = args_copy.source_dataset
        """
        args should contain:
            prediction_label: string
                Can contain a prefix followed by a wildcard *: "style_*".
                In that case, all columns starting with prefix are matched.
        """
        df = vislab.dataset.get_df_with_args(args_copy)
        prefix = 'style_'

        column_names = [col for col in df.columns if col.startswith(prefix)]
        dataset = vislab.predict.get_multiclass_dataset(df, 'wikipaintings', 'styles_ALL',
                                                        column_names, test_frac=.2,
                                                        balanced=False, random_seed=42)

        dataset['train_df']['split'] = 'train'
        dataset['val_df']['split'] = 'val'
        dataset['test_df']['split'] = 'test'
        df = dataset['train_df'].append(dataset['val_df']).append(dataset['test_df'])

        del df['importance']
        del df['label']
        df = correct_columns_order(df)
        print 'Saving split to hdf5'
        df.to_hdf('/net/hciserver03/storage/asanakoy/workspace/'
                  'wikiart/karayev/styles.hdf5', 'df', mode='w')
        print 'SAVED!'
    else:
        raise Exception('source_dataset is None')