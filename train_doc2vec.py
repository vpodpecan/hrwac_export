import argparse
import logging

import psutil
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read hrWaC2.1 .xml.gz archive and write a TaggedLineDocument output')
    parser.add_argument('-m', '--model', required=True, help='output model file name')
    parser.add_argument('-d', '--dimensions', type=int, default=300, help='dimensionality of the model')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('corpus', help='hrwac corpus in TaggedLineDocument format')
    args = parser.parse_args()

    model = Doc2Vec(vector_size=args.dimensions,
                    epochs=args.epochs,
                    workers=psutil.cpu_count(logical=False))
    doc_stream = TaggedLineDocument(args.corpus)
    model.build_vocab(documents=doc_stream)
    model.train(documents=doc_stream, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=model.epochs)
    model.save(args.model)
