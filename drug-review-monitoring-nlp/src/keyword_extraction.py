import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import spacy

warnings.filterwarnings('ignore')

import gensim
import nltk
import pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pyLDAvis.gensim_models as gensimvis
from pyLDAvis import prepare, save_html
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
 
def tfidfmodel(documents):
    # --- Create TF-IDF vectorizer ---
 vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(2, 2)
    )

           
 tfidf_matrix = vectorizer.fit_transform(documents)

    # --- FIX STARTS HERE: Identify and remove empty documents ---
    # Check which rows (documents) have a sum of zero
 doc_sums = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
    
    # Create masks for keeping valid rows
 non_empty_docs_mask = doc_sums > 0
    
    # Filter the matrix, the documents list, and the document lengths list
 tfidf_matrix_filtered = tfidf_matrix[non_empty_docs_mask, :]
 documents_filtered = [doc for doc, keep in zip(documents, non_empty_docs_mask) if keep]
    
    # Update the doc_lengths using the filtered documents list
 doc_lengths = [len(doc.split()) for doc in documents_filtered]

 print(f"\n🗑️ Removed {len(documents) - len(documents_filtered)} empty documents.")
 print("📊 Filtered TF-IDF Matrix Shape:", tfidf_matrix_filtered.shape)
    # --- FIX ENDS HERE ---

 feature_names = vectorizer.get_feature_names_out()
 words = np.array(feature_names)

    # Get the vocabulary and term frequencies (use filtered matrix)
 vocab = feature_names
 term_frequency = np.asarray(tfidf_matrix_filtered.sum(axis=0)).ravel()

    # (You can keep the top words print section if you like, using the filtered matrix)
    # ...

    # now apply nmf (use filtered matrix for fit_transform)
 nmf = NMF(n_components=10, init='nndsvda', solver="cd", random_state=1)

 doc_topic_dists = nmf.fit_transform(tfidf_matrix_filtered)
 topic_term_dists = nmf.components_
    
    # Normalize the distributions
 topic_term_dists = normalize(topic_term_dists, norm='l1', axis=1)
 doc_topic_dists_normalized = normalize(doc_topic_dists, norm='l1', axis=1)

        # --- ADDED: Explicitly handle NaNs/Infs right before pyLDAvis ---
 topic_term_dists = np.nan_to_num(topic_term_dists, nan=0.0, posinf=0.0, neginf=0.0)
 doc_topic_dists_normalized = np.nan_to_num(doc_topic_dists_normalized, nan=0.0, posinf=0.0, neginf=0.0)
 term_frequency = np.nan_to_num(term_frequency, nan=0.0, posinf=0.0, neginf=0.0)
    # --- END ADDED STEP ---



    # visualise nmf

    # Pass the filtered and normalized components to prepare
 pyLDAvis_data = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists_normalized,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency
    )
    
 pyLDAvis.save_html(pyLDAvis_data, '../output/nmf_visualization.html')
 
 topics_dict = {}
 num_words=5
 for i, topic in enumerate(topic_term_dists):
    # Get top word indices for this topic
        top_word_indices = topic.argsort()[-num_words:][::-1]
        
        # Extract words and their weights
        topic_words = [(feature_names[i], topic[i]) for i in top_word_indices]
        topics_dict[i] = topic_words
        print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in words[topic.argsort()[-10:]]])))
 return topics_dict

 if __name__ == "__main__":
     print("Testing keyword_extraction.py...")