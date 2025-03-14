"""
German Language Learning Videos Analytics Dashboard

This Streamlit app visualizes data from German language learning videos,
showing engagement metrics, content performance by topic, language level, etc.

Make sure to place your CSV file ('enriched_segments_rieke_neue.csv')
in the same directory as this script, or use the file uploader in the app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="German Language Learning Videos Analytics",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions - IMPORTANT: These need to be defined BEFORE they are used in load_data
def parse_keywords(kw_str):
    """Parse keywords from string to list, handling various formats."""
    if not isinstance(kw_str, str) or not kw_str.strip():
        return []
    
    try:
        # Try parsing as JSON
        return json.loads(kw_str.replace("'", '"'))
    except json.JSONDecodeError:
        # Fallback method for non-JSON formatted strings
        try:
            # Try handling as a Python literal (list of strings)
            if kw_str.startswith('[') and kw_str.endswith(']'):
                # Remove brackets and split by commas
                items = kw_str.strip('[]').split(',')
                # Clean each item
                return [item.strip().strip('"\'') for item in items if item.strip()]
            else:
                # If it's a single keyword without brackets
                return [kw_str.strip()]
        except:
            # If all else fails, return an empty list
            return []

def extract_all_keywords(df):
    """Extract all keywords from the dataframe and count their frequency."""
    all_keywords = []
    for kw_list in df['keywords_list']:
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list)
        elif isinstance(kw_list, str):
            all_keywords.append(kw_list)
    return Counter(all_keywords)

def generate_color_scale(n):
    """Generate a color scale with n distinct colors"""
    return px.colors.qualitative.G10[:n] if n <= 10 else px.colors.qualitative.Alphabet[:n]

@st.cache_data
def load_data():
    """Load and preprocess the data from CSV file."""
    # Try different possible file names
    possible_filenames = [
        'enriched_segments_rieke_neue.csv',  # With underscore
    ]
    
    for filename in possible_filenames:
        try:
            df = pd.read_csv(filename)
            
            # Clean and convert keywords from string to list
            if 'keywords' in df.columns:
                df['keywords_list'] = df['keywords'].apply(parse_keywords)
            
            # Convert date columns
            try:
                if 'published_at' in df.columns:
                    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
                
                if 'publication_date' in df.columns:
                    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            except Exception as e:
                st.warning(f"Issue converting date columns: {e}")
                
            # Ensure numeric columns are properly typed
            numeric_cols = ['view_count', 'like_count', 'comment_count', 'video_duration_seconds',
                          'likes_per_1000_views', 'comments_per_1000_views']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate segment duration
            if 'start_at' in df.columns and 'end_at' in df.columns:
                df['segment_duration'] = df['end_at'] - df['start_at']
            
            return df, filename
        except FileNotFoundError:
            continue
    
    # If we get here, none of the files were found
    return None, None

# Load data and show toast
df, loaded_filename = load_data()
if loaded_filename:
    st.toast(f"Successfully loaded data from {loaded_filename}", icon="✅")
else:
    st.error("Could not find the data file. Please upload it below.")

# Check if data is available
# ------- SIDEBAR -------
st.sidebar.markdown("## Filters")

# Topic category filter
if 'topic_category' in df.columns:
    topics = sorted(df['topic_category'].dropna().unique().tolist())
    selected_topics = st.sidebar.multiselect("Topic Categories", topics, default=[])

# Is teaching filter
if 'keywords' in df.columns:
    # Flatten the list of keywords and get unique values
    all_keywords = []
    for kw_list in df['keywords_list']:
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list)
    teaching_options = sorted(set(all_keywords))  # Convert to set to get unique values
    selected_teaching = st.sidebar.multiselect("Keyboards", teaching_options, default=[])

# Language level filter
if 'language_level' in df.columns:
    levels = sorted(df['language_level'].dropna().unique().tolist())
    selected_levels = st.sidebar.multiselect("Language Levels", levels, default=[])

# Feedback type filter
if 'feedback_type' in df.columns:
    feedback_types = sorted(df['feedback_type'].dropna().unique().tolist())
    selected_feedback = st.sidebar.multiselect("Feedback Types", feedback_types, default=[])

# Date range filter
if 'publication_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['publication_date']):
    try:
        min_date = df['publication_date'].min().date()
        max_date = df['publication_date'].max().date()
        date_range = st.sidebar.date_input("Publication Date Range", 
                                            value=(min_date, max_date),
                                            min_value=min_date,
                                            max_value=max_date)
    except (AttributeError, ValueError):
        st.sidebar.warning("Publication date format issues detected. Date filter disabled.")
        date_range = None
else:
    st.sidebar.info("Publication date not available in the dataset.")
    date_range = None

# Apply filters
filtered_df = df.copy()

if 'topic_category' in filtered_df.columns and selected_topics:
    filtered_df = filtered_df[filtered_df['topic_category'].isin(selected_topics)]

if 'language_level' in filtered_df.columns and selected_levels:
    filtered_df = filtered_df[filtered_df['language_level'].isin(selected_levels)]
    
if 'feedback_type' in filtered_df.columns and selected_feedback:
    filtered_df = filtered_df[filtered_df['feedback_type'].isin(selected_feedback)]
    
if 'is_teaching' in filtered_df.columns and selected_teaching:
    filtered_df = filtered_df[filtered_df['is_teaching'].isin(selected_teaching)]

if 'publication_date' in filtered_df.columns and date_range and len(date_range) == 2:
    try:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['publication_date'].dt.date >= start_date) & 
                                (filtered_df['publication_date'].dt.date <= end_date)]
    except:
        st.warning("Could not apply date filter due to date format issues.")

# ------- MAIN DASHBOARD -------
st.subheader('Rieke Segment Analytics')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Videos", filtered_df['video_id'].nunique())
    
with col2:
    st.metric("Total Segments", filtered_df.shape[0])
    
with col3:
    # Count unique keywords by flattening the list first
    all_keywords = []
    for kw_list in filtered_df['keywords_list']:
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list)
    unique_keywords_count = len(set(all_keywords))
    st.metric("Total Keywords", unique_keywords_count)

with col4:
    if 'likes_per_1000_views' in filtered_df.columns:
        avg_likes = filtered_df['likes_per_1000_views'].mean()
        st.metric("Avg. Likes per 1000 Views", f"{avg_likes:.2f}", delta="1")
    else:
        st.metric("Avg. Likes per 1000 Views", "N/A")
    
with col5:
    if 'comments_per_1000_views' in filtered_df.columns:
        avg_comments = filtered_df['comments_per_1000_views'].mean()
        st.metric("Avg. Comments per 1000 Views", f"{avg_comments:.2f}", delta="1")
    else:
        st.metric("Avg. Comments per 1000 Views", "N/A")

# Top performing content
st.subheader('Top Performing Content')

metrics_tab1, metrics_tab2 = st.tabs(["By Likes", "By Comments"])

with metrics_tab1:
    if 'likes_per_1000_views' in filtered_df.columns and 'segment_title' in filtered_df.columns:
        top_likes_df = filtered_df.sort_values('likes_per_1000_views', ascending=False).head(10)
        
        fig = px.bar(
            top_likes_df,
            x='likes_per_1000_views',
            y='segment_title',
            orientation='h',
            title='Top 10 Segments by Likes per 1000 Views',
            labels={'likes_per_1000_views': 'Likes per 1000 Views', 'segment_title': 'Segment Title'},
            color='likes_per_1000_views',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required columns for this visualization are not available in the dataset.")

with metrics_tab2:
    if 'comments_per_1000_views' in filtered_df.columns and 'segment_title' in filtered_df.columns:
        top_comments_df = filtered_df.sort_values('comments_per_1000_views', ascending=False).head(10)
        
        fig = px.bar(
            top_comments_df,
            x='comments_per_1000_views',
            y='segment_title',
            orientation='h',
            title='Top 10 Segments by Comments per 1000 Views',
            labels={'comments_per_1000_views': 'Comments per 1000 Views', 'segment_title': 'Segment Title'},
            color='comments_per_1000_views',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required columns for this visualization are not available in the dataset.")

# Content analysis by category
st.subheader('Content Analysis')

analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["By Topic", "By Language Level", "By Content Type"])

with analysis_tab1:
    if 'topic_category' in filtered_df.columns and 'likes_per_1000_views' in filtered_df.columns:
        topic_metrics = filtered_df.groupby('topic_category').agg({
            'likes_per_1000_views': 'mean',
            'comments_per_1000_views': 'mean',
            'segment_id': 'count'
        }).reset_index()
        
        topic_metrics.rename(columns={'segment_id': 'segment_count'}, inplace=True)
        topic_metrics.sort_values('likes_per_1000_views', ascending=False, inplace=True)
        
        fig = px.bar(
            topic_metrics,
            x='topic_category',
            y=['likes_per_1000_views', 'comments_per_1000_views'],
            title='Engagement by Topic Category',
            labels={
                'topic_category': 'Topic Category',
                'value': 'Engagement per 1000 Views',
                'variable': 'Metric'
            },
            barmode='group',
            hover_data=['segment_count']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required columns for this visualization are not available in the dataset.")
        
with analysis_tab2:
    if 'language_level' in filtered_df.columns and 'likes_per_1000_views' in filtered_df.columns:
        # Define the correct order of language levels
        level_order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        
        # Filter to only include valid levels and create the aggregation
        level_df = filtered_df[filtered_df['language_level'].isin(level_order)]
        
        if not level_df.empty:
            level_metrics = level_df.groupby('language_level').agg({
                'likes_per_1000_views': 'mean',
                'comments_per_1000_views': 'mean',
                'segment_id': 'count'
            }).reset_index()
            
            # Create a categorical type with our custom ordering
            level_metrics['language_level'] = pd.Categorical(
                level_metrics['language_level'], 
                categories=level_order, 
                ordered=True
            )
            
            level_metrics.rename(columns={'segment_id': 'segment_count'}, inplace=True)
            level_metrics.sort_values('language_level', inplace=True)
            
            fig = px.line(
                level_metrics,
                x='language_level',
                y=['likes_per_1000_views', 'comments_per_1000_views'],
                title='Engagement by Language Level',
                labels={
                    'language_level': 'Language Level',
                    'value': 'Engagement per 1000 Views',
                    'variable': 'Metric'
                },
                markers=True,
                hover_data=['segment_count']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for standard language levels (A1, A2, B1, B2, C1, C2).")
    else:
        st.info("Required columns for this visualization are not available in the dataset.")

with analysis_tab3:
    content_cols = ['tone', 'feedback_type', 'is_teaching']
    available_cols = [col for col in content_cols if col in filtered_df.columns]
    
    if available_cols and 'likes_per_1000_views' in filtered_df.columns:
        selected_dimension = st.selectbox("Select Content Dimension", available_cols)
        
        content_metrics = filtered_df.groupby(selected_dimension).agg({
            'likes_per_1000_views': 'mean',
            'comments_per_1000_views': 'mean',
            'segment_id': 'count'
        }).reset_index()
        
        content_metrics.rename(columns={'segment_id': 'segment_count'}, inplace=True)
        content_metrics.sort_values('likes_per_1000_views', ascending=False, inplace=True)
        
        fig = px.bar(
            content_metrics,
            x=selected_dimension,
            y=['likes_per_1000_views', 'comments_per_1000_views'],
            title=f'Engagement by {selected_dimension}',
            labels={
                selected_dimension: selected_dimension.replace('_', ' ').title(),
                'value': 'Engagement per 1000 Views',
                'variable': 'Metric'
            },
            barmode='group',
            hover_data=['segment_count']
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required columns for this visualization are not available in the dataset.")

# Keyword analysis
st.subheader('Keyword Analysis')

# Check if keywords column exists
if 'keywords' in filtered_df.columns:
    # Ensure keywords_list exists in the dataframe
    if 'keywords_list' not in filtered_df.columns:
        # Process keywords if needed
        filtered_df['keywords_list'] = filtered_df['keywords'].apply(parse_keywords)
    
    # Get keyword frequencies
    keyword_counter = extract_all_keywords(filtered_df)
    
    if keyword_counter:
        top_keywords = dict(keyword_counter.most_common(20))
        
        # Create keyword frequency chart
        fig = px.bar(
            x=list(top_keywords.keys()),
            y=list(top_keywords.values()),
            title='Top 20 Keywords by Frequency',
            labels={'x': 'Keyword', 'y': 'Frequency'},
            color=list(top_keywords.values()),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword engagement analysis
        st.markdown("### Keyword Engagement Impact")
        
        # Get all unique keywords
        all_unique_keywords = sorted(set(keyword_counter.keys()))
        
        if all_unique_keywords:
            # Option to select specific keywords to analyze
            selected_keyword = st.selectbox(
                "Select a keyword to analyze its impact on engagement", 
                all_unique_keywords
            )
            
            if selected_keyword:
                # Create mask for segments containing the selected keyword
                keyword_mask = filtered_df['keywords_list'].apply(lambda x: selected_keyword in x if isinstance(x, list) else False)
                
                if keyword_mask.any():  # Check if any rows match the keyword
                    # Prepare data for comparison
                    comparison_data = pd.DataFrame({
                        'Category': ['With Keyword', 'Without Keyword'],
                        'Likes per 1000 Views': [
                            filtered_df[keyword_mask]['likes_per_1000_views'].mean(),
                            filtered_df[~keyword_mask]['likes_per_1000_views'].mean()
                        ],
                        'Comments per 1000 Views': [
                            filtered_df[keyword_mask]['comments_per_1000_views'].mean(),
                            filtered_df[~keyword_mask]['comments_per_1000_views'].mean()
                        ],
                        'Segment Count': [
                            keyword_mask.sum(),
                            (~keyword_mask).sum()
                        ]
                    })
                    
                    # Create the visualization
                    fig = px.bar(
                        comparison_data,
                        x='Category',
                        y=['Likes per 1000 Views', 'Comments per 1000 Views'],
                        title=f'Engagement Impact of Keyword: "{selected_keyword}"',
                        barmode='group',
                        text='Segment Count'
                    )
                    
                    fig.update_traces(texttemplate='%{text} segments', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate percentage difference
                    likes_diff_pct = ((comparison_data.loc[0, 'Likes per 1000 Views'] / 
                                    comparison_data.loc[1, 'Likes per 1000 Views']) - 1) * 100
                    
                    comments_diff_pct = ((comparison_data.loc[0, 'Comments per 1000 Views'] / 
                                        comparison_data.loc[1, 'Comments per 1000 Views']) - 1) * 100
                    
                    # Display insights box
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>Keyword Impact Analysis</h4>
                        <p>Segments containing the keyword "<span class="highlight">{selected_keyword}</span>" show 
                        <span class="highlight">{likes_diff_pct:.1f}%</span> {'higher' if likes_diff_pct > 0 else 'lower'} likes per 1000 views
                        and <span class="highlight">{comments_diff_pct:.1f}%</span> {'higher' if comments_diff_pct > 0 else 'lower'} comments per 1000 views
                        compared to segments without this keyword.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No segments found containing the keyword '{selected_keyword}'.")
        else:
            st.info("No keywords found in the dataset.")
    else:
        st.info("No keywords frequency data available.")
else:
    st.info("Keywords data is not available in the dataset.")

# Video duration vs. engagement analysis
st.subheader('Duration vs. Engagement Analysis')

if 'segment_duration' in filtered_df.columns and 'likes_per_1000_views' in filtered_df.columns:
    # Segment duration histogram with engagement overlay
    duration_fig = px.scatter(
        filtered_df,
        x='segment_duration',
        y='likes_per_1000_views',
        color='language_level' if 'language_level' in filtered_df.columns else None,
        size='view_count' if 'view_count' in filtered_df.columns else None,
        hover_name='segment_title' if 'segment_title' in filtered_df.columns else None,
        hover_data=['comments_per_1000_views', 'topic_category'] if 'topic_category' in filtered_df.columns else ['comments_per_1000_views'],
        title='Segment Duration vs. Engagement',
        labels={
            'segment_duration': 'Segment Duration (seconds)',
            'likes_per_1000_views': 'Likes per 1000 Views'
        }
    )
    
    # Add trendline
    duration_fig.update_layout(height=600)
    st.plotly_chart(duration_fig, use_container_width=True)
else:
    st.info("Required columns for duration analysis are not available in the dataset.")

# Segment data explorer
st.subheader('Segment Data Explorer')

# Group by video_id and aggregate data
grouped_df = df.groupby('video_id').agg({
    'video_title': 'first',  # Take the first title
    # You can add other aggregations here, like:
    # 'view_count': 'sum',
    # 'likes_per_1000_views': 'mean',
    # 'comments_per_1000_views': 'mean'
}).reset_index()

# Add count of entries per video_id
grouped_df['entry_count'] = df.groupby('video_id').size().values

# Define display columns based on what's available
all_possible_cols = ['video_id', 'video_title', 'entry_count', 'segment_title', 
                        'topic_category', 'language_level', 'tone', 'view_count', 
                        'likes_per_1000_views', 'comments_per_1000_views',
                        'segment_duration', 'feedback_type', 'is_teaching']

display_cols = [col for col in all_possible_cols if col in grouped_df.columns]

if display_cols:
    # Create the explorer dataframe with available columns
    explorer_df = grouped_df[display_cols]
    
    # Sort by engagement if available
    if 'likes_per_1000_views' in explorer_df.columns:
        explorer_df = explorer_df.sort_values('likes_per_1000_views', ascending=False)
    elif 'entry_count' in explorer_df.columns:
        explorer_df = explorer_df.sort_values('entry_count', ascending=False)
    
    st.subheader("Videos Grouped by ID")
    st.dataframe(explorer_df, use_container_width=True)
    
    # Add feature to view details for a selected video
    st.subheader("View Detailed Entries")
    selected_video = st.selectbox("Select a video ID to view all entries:", 
                                    options=grouped_df['video_id'].tolist(),
                                    format_func=lambda x: f"{x} - {grouped_df[grouped_df['video_id']==x]['video_title'].values[0]}")
    
    if selected_video:
        video_entries = df[df['video_id'] == selected_video]
        st.dataframe(video_entries, use_container_width=True)
    
    # Download option
    csv = grouped_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='german_learning_filtered_data.csv',
        mime='text/csv',
    )
