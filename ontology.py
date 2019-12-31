import json
from typing import List

from nltk.stem import PorterStemmer
import re
from text_utils import replace_with_definitions,common_elements, stem_sentence


class Node:
    def __init__(self, label, markers=None):
        self.label = label
        self.children: List[Node] = []
        self.markers = markers or []
        self.definitions = replace_with_definitions(self.markers) + stem_sentence(self.markers)

    def __str__(self):
        return '{}: {}\n{}'.format(self.label, self.markers, self.definitions)


class Ontology:
    def __init__(self, ontology_path):
        self.root = Node('root')
        self.init_tree(ontology_path)

    def init_tree(self, json_path):
        with open(json_path, 'rb') as json_file:
            # ontology = json.load(json_file)
            ontology = eval(json_file.read())
            categories = ontology['root']['children']
            for category_label, subcategories in categories.items():
                node = Node(category_label, subcategories)
                self.root.children.append(node)
            self.print_tree()

    def theme_from_words(self, words):
        category_scores = []
        for category in self.root.children:
            n_common_elems = len(common_elements(words, category.markers))
            category_scores.append([n_common_elems, category.label])

        category_scores = list(sorted(category_scores, key=lambda tup: tup[0], reverse=True))
        themes = [category_scores[0][1]]
        for score, category in category_scores[1:]:
            if score == category_scores[0][1]:
                themes.append(category)
        return themes

    def print_tree(self):
        for child in self.root.children:
            print(child)
