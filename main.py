from typing import List

from ontology import Ontology
from summarizers import TextRankSummarizer, LeskSummarizer, OntologySummarizer, BlendedSummarizer
from text_utils import read_docs, Document


ontology = Ontology('ontologies/sport.json')

doc_dir = 'BBC News Summary/News Articles/sport'
doc_dir_summary = 'BBC News Summary/Summaries/sport'
docs: List[Document] = read_docs(doc_dir, doc_dir_summary, n_docs=-1)

print('Read {} docs...'.format(len(docs)))

ontology_summarizer = OntologySummarizer(docs, ontology)
ontology_summarizer.summarize_docs()

textrank_summarizer = TextRankSummarizer(docs)
textrank_summarizer.summarize_docs()
textrank_summarizer.performance()

lesk_summarizer = LeskSummarizer(docs)
lesk_summarizer.summarize_docs()
lesk_summarizer.performance()
#
blended_summarizer = BlendedSummarizer(docs, summarizers=[textrank_summarizer, lesk_summarizer, ontology_summarizer])
blended_summarizer.performance()

