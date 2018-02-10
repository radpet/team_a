# Resolves coreferences in text with representative mentions.
#
# Usage:
# from coref_resolution import resolve_coreferences
# result = resolve_coreferences("text-to-be-coreferenced")
# expects string and returns string
#
# Prerequesites:
# You need to have a stanford corenlp server running
# Recommended version is 3.8, because 3.9 has bugs with the mention annotator
# Version 3.8 can be found here: http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
#
# Command to run corenlp server:
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 150000 -annotators tokenize,ssplit,pos,lemma,ner,parse,coref,mention


import sys
import xml.etree.ElementTree
import nltk

#nltk.download()

from pycorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer


nlp = StanfordCoreNLP('http://localhost:9000')
detokenizer = MosesDetokenizer()


def resolve_coreferences(text):
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,mention,coref',
        'outputFormat': 'json',
    })

    sentence_tokens = []
    sentence_updates = []
    sentence_tags = []
    for sentence in output['sentences']:
        tokens = [token['word'] for token in sentence['tokens']]
        sentence_tokens.append(tokens)
        sentence_updates.append([False] * len(tokens))
        tags = nltk.pos_tag(tokens)
        sentence_tags.append(tags)

    for corefs in output['corefs'].values():
        # skip corefs that do not relate to anything
        if len(corefs) < 2:
            continue

        # find the representative mention
        representativeMention = None
        for coref in corefs:
            if coref['isRepresentativeMention']:
                representativeMention = coref

        # replace all mentions with the representative
        for coref in corefs:
            sent = coref['sentNum'] - 1
            start = coref['startIndex'] - 1
            end = coref['endIndex'] - 1

            should_replace = True
            for i in range(start, end):
                if sentence_updates[sent][i]:
                    should_replace = False
                    break
                sentence_updates[sent][i] = True

            if not should_replace:
                continue

            prefix = sentence_tokens[sent][:start]
            suffix = sentence_tokens[sent][end:]
            replaced = [representativeMention['text']] + [''] * (end - start - 1)
            sentence_tokens[sent] = prefix + replaced + suffix

    resolved_sentences = []
    for tokens in sentence_tokens:

        # detokenize the sentence
        sentence = detokenizer.detokenize(tokens, return_str=True)

        # make sure the first letter is uppercase
        sentence = sentence[0].upper() + sentence[1:]

        # add to text
        resolved_sentences.append(sentence)

    return ' '.join(resolved_sentences)