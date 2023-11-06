import argparse
import functions_recommender_bc as frbc

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--stop_words", type=str, nargs='?', default='stop-words-en.txt')
parser.add_argument("-c", "--corpus", type=str, nargs='?', default='corpus-en.txt')
parser.add_argument("-d", "--documents", type=str, nargs='?', default='documents-01.txt')
args = parser.parse_args()

stop_words_file: str = args.stop_words
corpus_file: str = args.corpus
documents_file: str = args.documents

results = frbc.main(stop_words_file, corpus_file, documents_file)
