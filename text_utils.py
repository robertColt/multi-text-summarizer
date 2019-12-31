import os
import re
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import string
import itertools

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


class Document:
    def __init__(self, name):
        self.name = name
        self.text = ''
        self.summary = ''
        self.original_summary_sentences = []
        self.sentences = []
        self.original_sentences = []

    def set_text(self, text):
        self.text = text
        self.original_sentences, self.sentences = self.init_sentences(text)

    def set_text_summary(self, text):
        self.summary = text
        self.original_summary_sentences, self.summary_sentences = self.init_sentences(text)

    def init_sentences(self, text):
        original_sentences = sent_tokenize(text)
        original_sentences = [s.translate(str.maketrans('', '', string.punctuation)).lower() for s in
                              original_sentences]
        original_sentences = list(map(word_tokenize, original_sentences))
        # print('sentences no punctuation\n', self.sentences)
        sentences = list(map(remove_stops, original_sentences))
        sentences = list(map(stem_sentence, sentences))
        # print('\nsentences clean\n', self.sentences)
        return original_sentences, sentences

    def __str__(self):
        return '\n{} \noriginal: {} \nprocessed: {} \nsummary: {}'.format(self.name, self.original_sentences,
                                                                          self.sentences, self.summary_sentences)


def read_docs(doc_dir, doc_dir_summary, n_docs=-1):
    docs = []
    for i, doc_name in enumerate(os.listdir(doc_dir)):
        doc_path = doc_dir + '/' + doc_name
        doc_path_summary = doc_dir_summary + '/' + doc_name
        doc = Document(doc_name)
        with open(doc_path, 'rt') as doc_file:
            lines = doc_file.readlines()[1:]
            doc.set_text(' '.join(map(str.strip, lines)))
        with open(doc_path_summary, 'rt') as doc_file:
            lines = doc_file.readlines()
            text = ' '.join(map(str.strip, lines))
            text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)
            doc.set_text_summary(text)
        if len(doc.sentences) < len(doc.summary_sentences):
            continue
        docs.append(doc)
        if n_docs > 0 and i >= n_docs - 1:
            return docs
    return docs


def stem_sentence(sentence):
    return list(map(stemmer.stem, sentence))


def remove_stops(sentence):
    return [word for word in sentence if word
            not in stop_words and len(word) > 1]


def common_elements(list1, list2):
    return list(set(list1) & set(list2))


def common_tokens_similarity(sentence1, sentence2, verbose):
    from math import log
    words = list(set(sentence1 + sentence2))
    if verbose: print("\n\nAll Words", words, "\n\n")
    similarity = len(common_elements(sentence1, sentence2))
    common_len = len(sentence2) + len(sentence1)
    return similarity / common_len if common_len != 0 else 0


def common_sentences(true_summary, calculcated_summary):
    return sum([1 for sentence in calculcated_summary if sentence in true_summary])


def rouge_1_score(true_summary, calculcated_summary):
    common = common_sentences(true_summary, calculcated_summary)
    return common / len(calculcated_summary)


def replace_with_definitions(sentence):
    definitions_words = []
    for word in sentence:
        syns = wordnet.synsets(word)
        if len(syns) == 0:
            continue
        definition = syns[0].definition().lower()
        definition = word_tokenize(definition, 'english')
        definition = remove_stops(definition)
        definition = stem_sentence(definition)
        definitions_words.extend(definition)

    return definitions_words


def sentences_to_words(sentences_list):
    return list(itertools.chain.from_iterable(sentences_list))