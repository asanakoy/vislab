import vislab
from vislab.predict import *
import sys


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
        print 'Saving split to hdf5'
        df.to_hdf('/net/hciserver03/storage/asanakoy/workspace/wikiart/karayev/styles.hdf5',
                  'df', mode='w')
    else:
        raise Exception('source_dataset is None')