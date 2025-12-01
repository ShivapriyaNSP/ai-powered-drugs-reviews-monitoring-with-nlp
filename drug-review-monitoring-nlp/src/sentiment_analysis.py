import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

def sentiment_scores(text):
  sid_obj = SentimentIntensityAnalyzer()
  sentiment_dict = sid_obj.polarity_scores(text)
  #print(f"Sentiment Scores: {sentiment_dict}")
  #print(f"Negative Sentiment: {sentiment_dict['neg']*100}%")
  #print(f"Neutral Sentiment: {sentiment_dict['neu']*100}%")
  #print(f"Positive Sentiment: {sentiment_dict['pos']*100}%")
     
  if sentiment_dict['compound'] >= 0.05:
     return "Positive"
  elif sentiment_dict['compound'] <= -0.05:
     return "Negative"
  else:
      return "Neutral"
                    
def getsentiment_scorescompound(text):
   score_obj = SentimentIntensityAnalyzer()
   sentiment_dict = score_obj.polarity_scores(text)
   return sentiment_dict['compound'] 

def analyze_sentiment(df):
        """Perform sentiment analysis using Vader"""
        if df is None:
            return

        
        df['sentiment_score_compound'] = df['clean_review'].apply(getsentiment_scorescompound)
        df['sentiment_classifiers'] = df['clean_review'].apply(sentiment_scores)
        df['aspect_score'] = df['aspects'].apply(getsentiment_scorescompound)


     
        X= df[['sentiment_score_compound','rating','aspect_score']]
        Y= df['sentiment_classifiers']

        X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, Y, test_size=0.4, random_state=42)

        X_val, X_test, Y_val, Y_test = train_test_split(
            X_holdout, Y_holdout,
            test_size=0.5, # Split the holdout set 50/50 (results in 20% of original data)
            random_state=42, 
            stratify=Y_holdout
        )

        # Initialize and train the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model using all features
        rf_classifier.fit(X_train, Y_train)

        # predict with the model

        predictions= rf_classifier.predict(X_test)

        #predict on validate data

        y_val_pred = rf_classifier.predict(X_val)


        # calculate the performance matrics
        
        accuracy = accuracy_score(Y_test,predictions)

        print( f"accuracy score is {accuracy * 100:.2f}%  " )


        # report generation

        print("detailed report")
        print(classification_report(Y_test,predictions))

        print("confusion matrix")
        print(confusion_matrix(Y_test,predictions))

        print("F1_score :")
        print(f1_score(Y_test,predictions, average='macro'))

        print("F1_score Validate:")
        print(f1_score(Y_val,y_val_pred, average='macro'))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(' Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution (Pie chart)
        sentiment_counts = df['sentiment_classifiers'].value_counts()
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                     autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        ax1.set_title('Review Sentiment Distribution')

        # 2. Rating Distribution (Bar chart)
        rating_counts = df['rating'].value_counts().sort_index()
        ax2.bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
        ax2.set_title('Star Rating Distribution')
        ax2.set_xlabel('Rating (out of 10)')
        ax2.set_ylabel('Number of Reviews')


        # 3. Print sentiment distribution

        print("\n🎭 Sentiment Analysis Results:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment.capitalize()}: {count} reviews ({percentage:.1f}%)")
        return df

if __name__ == "__main__":
    print("Testing sentiment_analysis.py...")