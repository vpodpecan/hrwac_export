import argparse
import xml
import csv
import io
import time
import os
import sys
import gensim
from gensim.test.utils import datapath
from gensim.models.doc2vec import TaggedLineDocument
import smart_open
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 123

IGNORE_TOKENS = ['(', ')', '[', ']', '{', '}', "'", '"', '»', '«', ';', ':', ',', '!', '/', '\\',
                 '@', '#', '$', '^', '*', '+', '=', '-', '|', '_', '<', '>', '?', '!',
                 '.', ',', '&', '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '..', '...', '....', '.....']


def filter_tokens(tokens, ignorelist=IGNORE_TOKENS, minlen=2, maxlen=20, is_printable=True, is_alphanumeric=True, allow_numbers=False):
    result = [t.strip() for t in tokens if t not in ignorelist and len(t) >= minlen and len(t) <= maxlen]
    if is_printable:
        result = filter(str.isprintable, result)
        # result = [t for t in result if t.isprintable()]
    if is_alphanumeric:
        result = filter(str.isalnum, result)
        # result = [t for t in result if t.isalnum()]
    if not allow_numbers:
        result = [t for t in result if not t.replace('.', '').replace(',', '').replace(':', '').isdigit()]  # catch also real numbers and time measurements
    return result


def read_corpus(filename, tokens_only=False, max_docs=None, min_tokens_per_doc=10, max_tokens_per_doc=None,
                detect_language=True):
    cnt = 0
    with smart_open.open(filename) as fp:
        for event, elt in xml.etree.ElementTree.iterparse(fp, ['end']):
            if elt.tag == 'p':
                par_tokens = []
                for text in elt.itertext():
                    tokens = [row[1].lower() for row in csv.reader(io.StringIO(text), delimiter='\t') if len(row) == 4]
                    par_tokens.extend(filter_tokens(tokens))

                # filter too short and too long
                if len(par_tokens) < min_tokens_per_doc or (max_tokens_per_doc is not None and len(par_tokens) > max_tokens_per_doc):
                    continue

                # try to detect language
                if detect_language:
                    try:
                        par_text = ' '.join(par_tokens)
                        if detect(par_text) != 'hr':
                            continue
                    except Exception as e:
                        pass

                # finally, yield tokens or tagged document
                cnt += 1
                if max_docs is not None and cnt > max_docs:
                    return
                if tokens_only:
                    yield par_tokens
                else:
                    yield gensim.models.doc2vec.TaggedDocument(par_tokens, [cnt])


def to_taglndoc(archives, outfile, **kwargs):
    total_count = 0
    with open(outfile, 'w', encoding='utf-8') as ofp:
        for archive in archives:
            print('Processing {} ...'.format(archive), end=' ', file=sys.stderr)
            cnt = 0
            for doc in read_corpus(archive, **kwargs):
                line = ' '.join(doc.words) + '\n'
                ofp.write(line)
                cnt += 1
            total_count += cnt
            print('{} paragraphs exported.'.format(cnt), file=sys.stderr)
    print('Summary: {} paragraphs exported.'.format(total_count), file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read hrWaC2.1 .xml.gz archive and write a TaggedLineDocument output')
    parser.add_argument('-f', '--folder', help='input folder with hrWaC .xml.gz files')
    parser.add_argument('-m', '--min_length', default=50, type=int, help='minimum paragraph length (in tokens); default: 50')
    parser.add_argument('-M', '--max_length', default=None, type=int, help='maximum paragraph length (in tokens); default: unlimited')
    parser.add_argument('-D', '--max_docs', default=None, type=int, help='maximum number of documents from each archive; default: unlimited')
    parser.add_argument('--detect_language', action='store_true', help='detect language and filter non "hr" paragraphs (much slower!)')
    parser.add_argument('--web', action='store_true', help='read corpus directly from clarin.si repository')
    parser.add_argument('outfile', help='output file')
    args = parser.parse_args()

    if args.web and args.folder:
        print('--folder and --web parameters are mutually exclusive, please set only one')
        exit(1)
    if args.web:
        base = 'https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1064'
        archives = ['{}/hrWaC2.1.{:02d}.xml.gz'.format(base, i) for i in range(1, 15)]
    elif args.folder:
        archives = sorted([entry.path for entry in os.scandir(args.folder) if entry.path.lower().endswith('.xml.gz')])
        if len(archives) == 0:
            print('No .xml.gz archives in folder "{}"!'.format(args.folder))
            exit(1)
    else:
        print('One of these two arguments [folder, web] is required!')
        exit(1)

    to_taglndoc(archives, args.outfile,
                min_tokens_per_doc=args.min_length,
                max_tokens_per_doc=args.max_length,
                max_docs=args.max_docs,
                detect_language=args.detect_language)
