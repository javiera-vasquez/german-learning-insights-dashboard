import pandas as pd
import json
import numpy as np
import re

def preprocess_german_learning_data(file_path):
    """
    Preprocess the German language learning video data CSV.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Process date columns
    date_columns = ['published_at', 'publication_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Process keywords from string to list
    def parse_keywords(kw_str):
        if not isinstance(kw_str, str) or not kw_str.strip():
            return []
        
        try:
            # Replace single quotes with double quotes for valid JSON
            cleaned_str = kw_str.replace("'", '"')
            return json.loads(cleaned_str)
        except:
            # Fallback in case the JSON parsing fails
            # Strip brackets and split by comma
            cleaned_str = kw_str.strip('[]').split(',')
            return [k.strip(' "\'') for k in cleaned_str if k.strip()]
    
    df['keywords_list'] = df['keywords'].apply(parse_keywords)
    
    # Calculate derived metrics
    if 'start_at' in df.columns and 'end_at' in df.columns:
        df['segment_duration'] = df['end_at'] - df['start_at']
    
    # Ensure numeric columns are properly typed
    numeric_cols = [
        'view_count', 'like_count', 'comment_count', 'video_duration_seconds',
        'likes_per_1000_views', 'comments_per_1000_views', 'start_at', 'end_at'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize language levels
    if 'language_level' in df.columns:
        # Extract the main level (A1, A2, B1, etc.)
        def standardize_level(level):
            if not isinstance(level, str):
                return np.nan
            
            # Find patterns like A1, B2, C1
            match = re.search(r'([ABC][12])', level)
            if match:
                return match.group(1)
            return level
        
        df['language_level'] = df['language_level'].apply(standardize_level)
    
    # Fix missing values in categorical columns
    categorical_cols = [
        'topic_category', 'tone', 'feedback_type', 'is_teaching', 'language_level'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            # Replace empty strings with NaN
            df[col] = df[col].replace('', np.nan)
    
    # Identify segments with high engagement
    if 'likes_per_1000_views' in df.columns:
        likes_threshold = df['likes_per_1000_views'].quantile(0.75)
        df['high_engagement'] = df['likes_per_1000_views'] >= likes_threshold
    
    return df

def generate_sample_data():
    """
    Generate a small sample dataset for testing when the real data isn't available
    """
    # Sample data
    data = {
        'video_id': [f'vid_{i}' for i in range(1, 21)],
        'video_title': [f'German Learning Video {i}' for i in range(1, 21)],
        'segment_title': [f'Segment {i}' for i in range(1, 21)],
        'start_at': np.random.randint(0, 500, 20),
        'end_at': np.random.randint(30, 600, 20),
        'segment_content': [f'Lorem ipsum dolor sit amet {i}' for i in range(1, 21)],
        'view_count': np.random.randint(100, 10000, 20),
        'like_count': np.random.randint(10, 1000, 20),
        'comment_count': np.random.randint(0, 100, 20),
        'video_duration_seconds': np.random.randint(300, 1200, 20),
        'video_duration': [f'{np.random.randint(5, 20)}:{np.random.randint(10, 60):02d}' for _ in range(20)],
        'published_at': pd.date_range(start='2023-01-01', periods=20),
        'publication_date': pd.date_range(start='2023-01-01', periods=20),
        'segment_id': [f'seg_{i}' for i in range(1, 21)],
        'topic_category': np.random.choice(['Expressions', 'Verbs', 'Vocabulary', 'Grammar'], 20),
        'tone': np.random.choice(['Didactic', 'Conversational', 'Formal'], 20),
        'feedback_type': np.random.choice(['Explanation', 'Examples', 'Definition'], 20),
        'is_teaching': np.random.choice(['True', 'False'], 20),
        'language_level': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], 20),
        'confidence': np.random.choice(['High', 'Medium', 'Low'], 20),
        'keywords': [f'["{np.random.choice(["Umgangssprache", "Ausdrücke", "Grammatik", "Verben", "Vokabeln"])}", "{np.random.choice(["Deutsch", "Lernen", "Anfänger", "Fortgeschritten"])}"]' for _ in range(20)]
    }
    
    # Calculate metrics
    df = pd.DataFrame(data)
    df['likes_per_1000_views'] = (df['like_count'] / df['view_count']) * 1000
    df['comments_per_1000_views'] = (df['comment_count'] / df['view_count']) * 1000
    
    return df

if __name__ == "__main__":
    # Test the preprocessor
    try:
        # Try to load and preprocess the actual data
        processed_df = preprocess_german_learning_data('enriched_segments_rieke_neue.csv')
        print(f"Processed actual data: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        
        # Display sample of the processed data
        print("\nSample of processed data:")
        print(processed_df.head())
        
    except FileNotFoundError:
        # Generate sample data if the file doesn't exist
        print("Real data file not found. Generating sample data...")
        sample_df = generate_sample_data()
        
        # Process the sample data
        processed_sample = preprocess_german_learning_data(sample_df)
        print(f"Generated sample data: {processed_sample.shape[0]} rows, {processed_sample.shape[1]} columns")
        
        # Save the sample data
        sample_df.to_csv('sample_german_learning_data.csv', index=False)
        print("Sample data saved to 'sample_german_learning_data.csv'")