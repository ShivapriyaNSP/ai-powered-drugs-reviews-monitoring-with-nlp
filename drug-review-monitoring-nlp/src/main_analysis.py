# main_analysis.py
import pandas as pd
# Import functions from your other files
from data_processing import clean_data
from topic_modeling import topic_modelinglda
from sentiment_analysis import analyze_sentiment
from keyword_extraction import tfidfmodel

def main():
    print("🎯 STARTING DRUG REVIEW ANALYSIS PIPELINE")
    print("=" * 50)
    
    print("\n1️⃣ STEP 1: Loading and cleaning data...")

    full_path = '../data/drugsComTrain_raw.csv'
    train_df = pd.read_csv(full_path)

    #Testing Data
    test_full_path  = '../data/drugsComTest_raw.csv'
    test_df= pd.read_csv(test_full_path)
    #Combined data

    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"✅ Dataset loaded successfully! Total reviews: {len(df):,}")

    drug_keywords = [
            'voltaren', 'diclofenac', 'diclofenac gel', 'voltaren gel',
            'voltaren emulgel', 'topical diclofenac'
        ]
    # Create a mask for specific drug reviews, this needs to be dynamic in future enhancement
    mask = df['drugName'].str.lower().str.contains('|'.join(drug_keywords), na=False)
       
    drug_df = df[mask].copy()
    print(f"🎯 Found {len(drug_df)} Voltaren/diclofenac reviews")
    if len(drug_df) > 0:
            # Basic stats
        unique_drugs = drug_df['drugName'].value_counts()
        print(f"💊 Voltaren products found:")
        for drug, count in unique_drugs.items():
                print(f"   - {drug}: {count} reviews")
        
    cleaned_df = clean_data(drug_df) 
    

    # Step 2: Topic modeling
    print("\n2️⃣ STEP 2: Finding topics...")
    topics = topic_modelinglda(cleaned_df)  

    # step 3: Keyword extraction 
    print("\n2️⃣ STEP 2: Finding Keywords...")
    keywords = tfidfmodel(cleaned_df['clean_review'])  

    # Step 4: Sentiment analysis  
    print("\n3️⃣ STEP 3: Analyzing sentiment...")
    df = analyze_sentiment(cleaned_df)
    
    
    # Show summary
    print("\n📈 ANALYSIS SUMMARY:")
    print(f"Total reviews processed: {len(df)}")
    print(f"Topics found: {len(topics)}")
    print(f"keywords extracted: {len(keywords)}")
    print(f"Sentiment distribution:")
    print(df['sentiment_classifiers'].value_counts())
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")

# Run the main function
if __name__ == "__main__":
    main()