import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import networkx
import string
from typing import List
from abc import ABC, abstractmethod
import itertools
from ontology import Ontology
from text_utils import *

nltk.download('punkt')
nltk.download('stopwords')


def tuples_to_normalized_score(index_score_tuples):
    num_tuples = len(index_score_tuples)
    sorted_tuples = sorted(index_score_tuples, key=lambda tup: tup[1])
    # print(sorted_tuples)
    for i in range(num_tuples):
        sorted_tuples[i][1] = i
    return sorted(sorted_tuples, key=lambda tup: tup[0])


def summary_from_score_tuples(sentence_scores, sentences, n_summary_sentences):
    sorted_sentences = sorted(sentence_scores, key=lambda tup: tup[1], reverse=True)
    # print(sentences, '\n', sorted_sentences)
    summary = []
    for i in range(n_summary_sentences):
        top_sentence_i = sorted_sentences[i][0]
        summary.append(sentences[top_sentence_i])

    return summary


class Summarizer(ABC):
    def __init__(self, docs):
        self.docs: List[Document] = docs
        # self.print_docs()
        self.summaries = []
        self.doc_sentence_scores = []
        self.rouge_1_scores = []

    def print_docs(self):
        for doc in self.docs:
            print(doc)

    def compute_performance_metrics(self, sentence_scores, actual_summary, all_sentences):
        n_sentences_summary = len(actual_summary)
        predicted_summary = summary_from_score_tuples(sentence_scores, all_sentences, n_sentences_summary)
        rouge_1 = rouge_1_score(actual_summary, predicted_summary)
        self.rouge_1_scores.append(rouge_1)

    def performance(self, verbose=True):
        mean_rouge_1 = np.mean(self.rouge_1_scores)
        if verbose:
            print('{}: avg. Rouge-1: {:.2f}'.format(self.__class__.__name__, mean_rouge_1))
        return mean_rouge_1

    @abstractmethod
    def summarize_docs(self):
        pass


class TextRankSummarizer(Summarizer):
    def __init__(self, docs):
        super().__init__(docs)

    def similarity_matrix_common_tokens(self, sentences):
        num_sentences = len(sentences)
        similarity_matrix = np.zeros((num_sentences, num_sentences))

        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:
                    similarity_matrix[i][j] = common_tokens_similarity(sentences[i], sentences[j], verbose=False)
        return similarity_matrix

    def summarize_docs(self):
        for doc in self.docs:
            similarity_matrix = self.similarity_matrix_common_tokens(doc.sentences)
            graph = networkx.from_numpy_matrix(similarity_matrix)
            scores_dict = networkx.pagerank(graph, max_iter=500)
            sentence_scores = [[index, score] for index, score in scores_dict.items()]
            sentence_scores_normalized = tuples_to_normalized_score(sentence_scores)
            self.doc_sentence_scores.append(sentence_scores)
            self.compute_performance_metrics(sentence_scores_normalized, doc.summary_sentences, doc.sentences)
            # print(sentence_scores)


class LeskSummarizer(Summarizer):
    def __init__(self, docs):
        super().__init__(docs)

    def summarize_docs(self):
        for doc in self.docs:
            sentence_scores = []
            for i, (original_sentence, sentence) in enumerate(zip(doc.original_sentences, doc.sentences)):
                sentence_definitions_words = replace_with_definitions(original_sentence)
                other_sentences = doc.sentences[:i] + doc.sentences[i + 1:]
                other_sentences_words = sentences_to_words(other_sentences)
                n_common_words = len(common_elements(sentence_definitions_words, other_sentences_words))
                sentence_scores.append([i, n_common_words])

            sentence_scores_normalized = tuples_to_normalized_score(sentence_scores)
            self.doc_sentence_scores.append(sentence_scores_normalized)
            self.compute_performance_metrics(sentence_scores_normalized, doc.summary_sentences, doc.sentences)
            # print(sorted(sentence_scores_normalized, key=lambda tup: tup[1]))
            # top_sentences = list(reversed(sorted(sentence_scores)))
            # print('TOP sentences Lesk', sentence_scores)


class OntologySummarizer(Summarizer):
    def __init__(self, docs,
                 ontology: Ontology
                 ):
        super().__init__(docs)
        self.ontology = ontology

    def summarize_docs(self):
        for doc in self.docs:

            sentence_scores = []
            doc_words = sentences_to_words(doc.sentences)
            doc_theme = self.ontology.theme_from_words(doc_words)
            for i, sentence in enumerate(doc.sentences):
                sentence_theme = self.ontology.theme_from_words(sentence)
                score = 1 if len(common_elements(sentence_theme, doc_theme)) > 0 else 0
                sentence_scores.append([i, score])
            sentence_scores = sorted(sentence_scores, key=lambda tup: tup[1], reverse=True)
            self.doc_sentence_scores.append(sentence_scores)
            # print(sorted(sentence_scores_normalized, key=lambda tup: tup[1]))
            # top_sentences = list(reversed(sorted(sentence_scores)))
            # print('TOP sentences Lesk', sentence_scores)


class BlendedSummarizer(Summarizer):
    def __init__(self, docs,
                 summarizers: List[Summarizer],
                 summarizers_weights=None
                 ):
        super().__init__(docs)
        self.summarizers = summarizers
        self.combined_scores = []
        self.summarizers_weights = summarizers_weights or [1 / len(summarizers)] * len(summarizers)
        self.combine_scores()

    def summarize_docs(self):
        for doc in self.docs:
            for i, (original_sentence, sentence) in enumerate(zip(doc.original_sentences, doc.sentences)):
                continue

    def combine_scores(self):
        for i, doc in enumerate(self.docs):
            sentence_scores = []
            for summarizer, summarizer_weight in zip(self.summarizers, self.summarizers_weights):
                summarizer_sent_scores = summarizer.doc_sentence_scores[i]
                # print(summarizer, summarizer_sent_scores)
                if len(sentence_scores) == 0:
                    sentence_scores = [[index, score * summarizer_weight] for index, score in summarizer_sent_scores]
                    continue
                for index, sentence_score in summarizer_sent_scores:
                    old_score = sentence_scores[index][1]
                    new_score = old_score + sentence_score * summarizer_weight
                    sentence_scores[index][1] = new_score

            self.compute_performance_metrics(sentence_scores, doc.summary_sentences, doc.sentences)
            # print(sentence_scores)
