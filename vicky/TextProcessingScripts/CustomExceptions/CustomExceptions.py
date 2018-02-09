# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

class SentenceShorterError(Exception):
    def __init__(self):
        Exception.__init__(self,"Sentence is shorter than the chosen sentence length") 

# <codecell>

class SentenceLongerError(Exception):
    def __init__(self):
        Exception.__init__(self,"Sentence is longer than the chosen sentence length") 

# <codecell>

class DocumentShorterError(Exception):
    def __init__(self):
        Exception.__init__(self,"Document is shorter than the chosen document length") 

# <codecell>

class DocumentLongerError(Exception):
    def __init__(self):
        Exception.__init__(self,"Document is longer than the chosen document length") 

# <codecell>


