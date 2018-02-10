Data 
----------------------------------------------------------------
The dataset is about inferring parent-subsidiary relations from text. Hoping for a supervised learning model, 
many annotated examples are needed. Those were automatically obtained by the following process:
- Identify pairs of companies mentioned in news (using Ontotext Named Entity Tagger)
- Ask DBpedia if there is a parent-subsidiary relationship between them.
- If yes, add the example to the training set as positive.
- We automatically generated negatives as well.

For the test set, we kept examples of parent-subsidiaries that are not written in DBpedia. 
Ontotext has those from an acquired dataset (it's expensive, it's not worth buying it for the datathon! :) ).

Files
----------------------------------------------------------------
The train.csv and test.csv data contain the following columns:

-- company1: string, label of the first company1  
-- company2: string, label of the second company1
-- is_parent: boolean, true if company1 is parent of company2; in the test file, this column is not filled (NA values)
-- snippet: string, containing a sentence or two extracting from news mentioning the two companies 


Additional information that could help
----------------------------------------------------------------
If you choose to implement state-of-the-art deep learning approaches, you may want to use word vector representations. 
Some pre-trained wordvectors cn be found here: 
       https://code.google.com/archive/p/word2vec/#Pre-trained_word_and_phrase_vectors
       

References:
----------------------------------------------------------------
Lin et al. 2016,  Neural Relation Extraction with Selective Attention over Instances
Mintz et al. 2009,  Distant supervision for relation extraction without labeled data. 
Miwa et al, 2016, End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures
Zeng et al, 2015, Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks
