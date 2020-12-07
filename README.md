# Dependency parser and POS tagger training code

```
./train_lang.sh \
    fastText-embeddings-binary \
    path-to-ud-treebanks \
    ud-corpus-name \
    output-directory \
    path-to-fastText-sources
```

## Example:

```
./train_lang.sh \
    cc.fr.300.bin \
    ../ud-treebanks-v2.7 \
    French-Sequoia \
    output/french-model \
    ../fastText
```
