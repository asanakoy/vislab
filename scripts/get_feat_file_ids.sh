FEAT_DIR=$HOME/work/vislab-git/data/feats
GZCAT_COMMAND='zcat'
find $FEAT_DIR -name "*.txt.gz" -exec sh -c "echo {} && ${GZCAT_COMMAND} {} | cut -d ' ' -f 2 | cut -c 3- > {}.ids.txt" \;
