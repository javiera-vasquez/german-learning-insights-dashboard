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

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E3A8A;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.8rem;
    color: #1E3A8A;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
}
.insight-box {
    background-color: #e6f3ff;
    border-left: 5px solid #2778c4;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
}
.highlight {
    color: #2778c4;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    df = pd.read_csv('enriched_segments_rieke_neue.csv')
    
    # Clean and convert keywords from string to list
    df['keywords_list'] = df['keywords'].apply(lambda x: 
        json.loads(x.replace("'", '"')) if isinstance(x, str) and x.strip() else [])
    
    # Convert date columns
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'])
    
    if 'publication_date' in df.columns:
        df['publication_date'] = pd.to_datetime(df['publication_date'])
        
    # Ensure numeric columns are properly typed
    numeric_cols = ['view_count', 'like_count', 'comment_count', 'video_duration_seconds',
                   'likes_per_1000_views', 'comments_per_1000_views']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate segment duration
    df['segment_duration'] = df['end_at'] - df['start_at']
    
    return df

def extract_all_keywords(df):
    all_keywords = []
    for kw_list in df['keywords_list']:
        all_keywords.extend(kw_list)
    return Counter(all_keywords)

def generate_color_scale(n):
    """Generate a color scale with n distinct colors"""
    return px.colors.qualitative.G10[:n] if n <= 10 else px.colors.qualitative.Alphabet[:n]

# Load the data
try:
    df = load_data()
    
    # ------- SIDEBAR -------
    st.sidebar.markdown("## Filters")
    
    # Topic category filter
    topics = sorted(df['topic_category'].dropna().unique().tolist())
    selected_topics = st.sidebar.multiselect("Topic Categories", topics, default=[])
    
    # Language level filter
    levels = sorted(df['language_level'].dropna().unique().tolist())
    selected_levels = st.sidebar.multiselect("Language Levels", levels, default=[])
    
    # Tone filter
    tones = sorted(df['tone'].dropna().unique().tolist())
    selected_tones = st.sidebar.multiselect("Content Tone", tones, default=[])
    
    # Feedback type filter
    feedback_types = sorted(df['feedback_type'].dropna().unique().tolist())
    selected_feedback = st.sidebar.multiselect("Feedback Types", feedback_types, default=[])
    
    # Is teaching filter
    teaching_options = sorted(df['is_teaching'].dropna().unique().tolist())
    selected_teaching = st.sidebar.multiselect("Teaching Content", teaching_options, default=[])
    
    # Date range filter
    if 'publication_date' in df.columns and not df['publication_date'].isna().all():
        min_date = df['publication_date'].min().date()
        max_date = df['publication_date'].max().date()
        date_range = st.sidebar.date_input("Publication Date Range", 
                                           value=(min_date, max_date),
                                           min_value=min_date,
                                           max_value=max_date)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_topics:
        filtered_df = filtered_df[filtered_df['topic_category'].isin(selected_topics)]
    
    if selected_levels:
        filtered_df = filtered_df[filtered_df['language_level'].isin(selected_levels)]
        
    if selected_tones:
        filtered_df = filtered_df[filtered_df['tone'].isin(selected_tones)]
        
    if selected_feedback:
        filtered_df = filtered_df[filtered_df['feedback_type'].isin(selected_feedback)]
        
    if selected_teaching:
        filtered_df = filtered_df[filtered_df['is_teaching'].isin(selected_teaching)]
    
    if 'publication_date' in df.columns and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['publication_date'].dt.date >= start_date) & 
                                (filtered_df['publication_date'].dt.date <= end_date)]
    
    # ------- MAIN DASHBOARD -------
    st.markdown('<div class="main-header">German Language Learning Videos Analytics</div>', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown('<div class="sub-header">Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Videos", filtered_df['video_id'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Segments", filtered_df.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_likes = filtered_df['likes_per_1000_views'].mean()
        st.metric("Avg. Likes per 1000 Views", f"{avg_likes:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        avg_comments = filtered_df['comments_per_1000_views'].mean()
        st.metric("Avg. Comments per 1000 Views", f"{avg_comments:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top performing content
    st.markdown('<div class="sub-header">Top Performing Content</div>', unsafe_allow_html=True)
    
    metrics_tab1, metrics_tab2 = st.tabs(["By Likes", "By Comments"])
    
    with metrics_tab1:
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
    
    with metrics_tab2:
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
    
    # Content analysis by category
    st.markdown('<div class="sub-header">Content Analysis</div>', unsafe_allow_html=True)
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["By Topic", "By Language Level", "By Content Type"])
    
    with analysis_tab1:
        if 'topic_category' in filtered_df.columns:
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
            
    with analysis_tab2:
        if 'language_level' in filtered_df.columns:
            # Define the correct order of language levels
            level_order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            
            # Filter to only include valid levels and create the aggregation
            level_df = filtered_df[filtered_df['language_level'].isin(level_order)]
            
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
    
    with analysis_tab3:
        content_cols = ['tone', 'feedback_type', 'is_teaching']
        selected_dimension = st.selectbox("Select Content Dimension", content_cols)
        
        if selected_dimension in filtered_df.columns:
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
    
    # Keyword analysis
    st.markdown('<div class="sub-header">Keyword Analysis</div>', unsafe_allow_html=True)
    
    # Get keyword frequencies
    keyword_counter = extract_all_keywords(filtered_df)
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
    
    # Option to select specific keywords to analyze
    all_unique_keywords = list(set([kw for sublist in filtered_df['keywords_list'] for kw in sublist]))
    all_unique_keywords.sort()
    
    selected_keyword = st.selectbox(
        "Select a keyword to analyze its impact on engagement", 
        all_unique_keywords
    )
    
    if selected_keyword:
        # Create mask for segments containing the selected keyword
        keyword_mask = filtered_df['keywords_list'].apply(lambda x: selected_keyword in x)
        
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
    
    # Video duration vs. engagement analysis
    st.markdown('<div class="sub-header">Duration vs. Engagement Analysis</div>', unsafe_allow_html=True)
    
    # Segment duration histogram with engagement overlay
    duration_fig = px.scatter(
        filtered_df,
        x='segment_duration',
        y='likes_per_1000_views',
        color='language_level' if 'language_level' in filtered_df.columns else None,
        size='view_count',
        hover_name='segment_title',
        hover_data=['comments_per_1000_views', 'topic_category'],
        title='Segment Duration vs. Engagement',
        labels={
            'segment_duration': 'Segment Duration (seconds)',
            'likes_per_1000_views': 'Likes per 1000 Views'
        }
    )
    
    # Add trendline
    duration_fig.update_layout(height=600)
    st.plotly_chart(duration_fig, use_container_width=True)
    
    # Segment data explorer
    st.markdown('<div class="sub-header">Segment Data Explorer</div>', unsafe_allow_html=True)
    
    display_cols = ['video_title', 'segment_title', 'topic_category', 'language_level', 'tone', 
                   'view_count', 'likes_per_1000_views', 'comments_per_1000_views']
    
    explorer_df = filtered_df[display_cols].sort_values('likes_per_1000_views', ascending=False)
    
    st.dataframe(explorer_df, use_container_width=True)
    
    # Download option
    st.download_button(
        label="Download Filtered Data as CSV",
        data=explorer_df.to_csv(index=False).encode('utf-8'),
        file_name='german_learning_filtered_data.csv',
        mime='text/csv',
    )
    
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure the CSV file 'enriched_segments_rieke_neue.csv' is in the same directory as this script.")