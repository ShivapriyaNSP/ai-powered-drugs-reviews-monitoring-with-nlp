import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

import gensim

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords


# Create Dictionary and Corpus
#Create a dictionary and corpus from the lemmatized tokens.

def topic_modelinglda(drug_df):

    # Create dictionary and corpus
    id2word = corpora.Dictionary(drug_df['lemmas_filtered'])
    texts = drug_df['lemmas_filtered']
    corpus = [id2word.doc2bow(text) for text in texts]

    #Build LDA Model
    #Build an LDA model with the specified number of topics.

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    topics = lda_model.print_topics(num_words=5)

      # 1. Interactive visualization
    vis = gensimvis.prepare(lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis, '../output/medical_lda_visualization.html') 
    
    return topics   
     
if __name__ == "__main__":
     print("Testing topic_modeling.py...")