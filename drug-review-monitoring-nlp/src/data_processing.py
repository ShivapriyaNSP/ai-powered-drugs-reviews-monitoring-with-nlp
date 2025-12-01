import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import spacy
import nltk
import gensim
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')

def clean_text(text):
        """Clean review text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs and special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\d+', '', text)  # Remove numbers
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
# Download nltk stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize(text):
        tokens = gensim.utils.simple_preprocess(text, deacc=True)
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

#lemmatize
# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If model not found, download it
    print("📥 Downloading spaCy English model...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


def extract_aspects(text):
    """
    Extracts key aspects (side effects) from a given text using spaCy.
    """
     # Use a smaller model for faster processing
    doc = nlp(text)

    aspects = []
    for token in doc:
        if token.pos_ == "NOUN" and token.dep_ == "dobj":  # Focus on nouns with a direct object (often related to side effects)
            aspects.append(token.text)
    return aspects



def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# Define a list of additional words to remove after lemmatization
additional_words_to_remove = ["say", "new", "days", "use", "think","tennis", "made", "soccer","put", "i", "d", "take", "get","hav", "month", "amp", "amp", "many", "time", ""] # Replace with the actual words you want to remove

# Convert the list to a set for efficient lookup
additional_words_set = set(additional_words_to_remove)

# Function to remove additional words from a list of lemmas
def remove_additional_words(lemmas_list):
    return [lemma for lemma in lemmas_list if lemma not in additional_words_set]


def clean_data(drug_df):
        """Clean and prepare the data"""
        if drug_df is None:
            return
        
        # Convert date
        drug_df['date'] = pd.to_datetime(drug_df['date'], errors='coerce')
        
        # Extract year and month for time analysis
        drug_df['year'] = drug_df['date'].dt.year
        drug_df['month'] = drug_df['date'].dt.month
        
        # Clean review text
        drug_df['clean_review'] = drug_df['review'].apply(clean_text)

        # aspect considerations
        
        drug_df['aspects'] = drug_df['clean_review'].apply(extract_aspects)
      
        # create tokens for LDA model

        drug_df['tokens'] = drug_df['clean_review'].apply(tokenize)

        #create lemmas for creating dictonory and corpus LDA model

        drug_df['lemmas'] = drug_df['tokens'].apply(lemmatize)
        drug_df['lemmas_filtered'] = drug_df['lemmas'].apply(remove_additional_words)
        
        print("✅ Data cleaning completed!")
        return drug_df

if __name__ == "__main__":
    print("Testing data_processing.py...")