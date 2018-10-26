import numpy as np
import pandas as pd
import nltk

class np_detector():
    """
    Detect NPs from pandas paragraphs.
    
    Access np_list variable to get the list of detected NPs.
    """
    # The pandas paragraphs or list of paragraphs.
    review_texts = None

    # The detected noun phrase.
    np_list      = []

    # The NP chunk grammar.
    _grammar     = \
r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""

    # The NP chunk parser.
    _cp          = None

    # Whether the review_texts is a list or pandas
    _pandas      = None

    def __init__(self, df_texts, np_grammar=None, pandas=True):
        self.review_texts    = df_texts
        self._pandas         = pandas
        self._cp             = nltk.RegexpParser(self._grammar) if np_grammar == None else nltk.RegexpParser(np_grammar)
        self.np_list         = self._list_nps()

    def _text_chunking(self, text):
        """
        Process a paragraph to an NP chunked paragraph.
        """
        sentences = nltk.sent_tokenize(text)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent)       for sent in sentences]
        sentences = [self._cp.parse(sent)     for sent in sentences]
        return sentences

    def _chunked_texts(self):
        """
        Generator for getting NP chunked paragraphs from the pandas/list of paragraphs.
        """
        if self._pandas:
            for idx, text in self.review_texts.iteritems():
                yield self._text_chunking(text)
        else:
            for text in self.review_texts:
                yield self._text_chunking(text)

    def _leaves(self, tree):
        """
        Generator for getting the subtrees from the NP chunked sentence.
        """
        for subtree in tree.subtrees(filter=lambda t:t.label()=='NP'):
            yield subtree.leaves()

    def _nps(self):
        """
        Generator for getting NPs from the subtree.
        """
        for chunk_sents in self._chunked_texts():
            for chunk_sent in chunk_sents:
                for leaf in self._leaves(chunk_sent):
                    np = []
                    for word, tag in leaf:
                        np.append(word)
                    yield ' '.join(np).lower()

    def _list_nps(self):
        """
        Collect NPs from NP generator.
        """
        np_list = []
        for np in self._nps():
            np_list.append(np)
        return np_list
