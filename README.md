### A parser and exporter for Croatian web corpus hrWaC 2.1

#### About
This python3 script reads, filters and exports the hrWaC 2.1 corpus from `.xml.gz` archives, either from a local copy or directly from the [CLARIN.SI](https://www.clarin.si/repository/xmlui/handle/11356/1064) repository.

The goal is to extract a large, high quality training set to train word and document embeddings such as `word2vec` and `doc2vec`.

Paragraphs are token-filtered according to the parameter settings and exported into a [TaggedLineDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedLineDocument) file where each line contains tokens from one paragraph.

The `read_corpus` function can be also used as an iterator over paragraph tokens from the selected hrWaC `.xml.gz` archive.

#### Training paragraph embeddings
The included script `train_doc2vec.py` can be used to train basic paragraph embeddings models (aka Doc2vec).

For example, to train a 100 dimensional Doc2vec model for 15 epochs on the newly exported hrWaC 2.1 corpus in one-document-per-line format stored in a file `hrwac21.txt` the following command will train the model and save it into the file `hrwac21_doc2vec_d100.model`:

```sh
python train_doc2vec.py -m hrwac_doc2vec_d100.model -d 100 -e 15 hrwac21.txt
```

The exported embeddings model can be used in your code by loading it using Gensim function `gensim.models.doc2vec.Doc2Vec.load`.


#### Requirements
The code works with `python 3.4+` and requires the following python modules (see requirements.txt):

-  `gensim`
-  `smart_open`
-  `langdetect`

#### How to use

The following command will extract paragraph tokens directly from the [CLARIN.SI](https://www.clarin.si/repository/xmlui/handle/11356/1064) hrWaC 2.1 repository using default parameter values and save them into a `hrwac.tld` file:

```sh
python3 process_hrwac.py --web hrwac.tld
```

There is also the `--detect_language` parameter which will instruct the parser to ignore all paragraphs that are not in Croatian. This will make the export **considerably slower** but is recommended for a high quality output because hrWaC contains a small portion of paragraphs in other languages (Slovene, English, ...).  

The parser supports the following parameters:

-  `-f`, `--folder`: input folder with hrWaC .xml.gz files
-  `-m`, `--min_length`, default=50, minimum paragraph length (in tokens)
-  `-M`, `--max_length`, default=unlimited, maximum paragraph length (in tokens)
-  `-D`, `--max_docs`, default=unlimited, maximum number of documents from each archive, useful to extract small samples
-  `--detect_language`, default: false, detect language and filter non `hr` paragraphs (much slower!)
-  `--web`, read corpus directly from the CLARIN.SI repository
-  `output_file`, name of the output file (a positional argument)

Note that the `--folder` and `--web` parameters are mutually exclusive.


#### Author
Vid Podpečan <vid.podpecan@ijs.si>


#### License

MIT
