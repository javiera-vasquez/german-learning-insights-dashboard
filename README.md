# German Learning Insights Dashboard

An interactive analytics dashboard for German language learning video content, providing insights into engagement metrics and content performance.

![Dashboard Screenshot](https://via.placeholder.com/800x450?text=German+Learning+Insights+Dashboard)

## Features

- **Key Metrics Overview**: Total videos, segments, and average engagement metrics
- **Top Performing Content Analysis**: Most engaging segments by likes and comments
- **Content Breakdown Analysis**: Performance by topic, language level, and content type
- **Keyword Impact Analysis**: Understand which keywords drive higher engagement
- **Duration vs. Engagement Analysis**: Visualize how segment length affects viewer engagement
- **Interactive Filtering**: Filter data by multiple dimensions (topic, language level, tone, etc.)
- **Data Explorer**: Browse and search through the detailed segment data
- **Export Functionality**: Download filtered data for further analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/javiera-vasquez/german-learning-insights-dashboard.git
   cd german-learning-insights-dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it (MacOS/Linux)
   source venv/bin/activate
   
   # Activate it (Windows)
   # venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your CSV data file in the project directory:
   - The file should be named `enriched_segments_rieke neue.csv`
   - Alternatively, you can upload the file through the dashboard's UI

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

6. Access the dashboard in your web browser at:
   ```
   http://localhost:8501
   ```

## Data Format

The dashboard expects a CSV file with the following columns:
- `video_id`: Unique identifier for videos
- `video_title`: Title of the video
- `segment_title`: Title of the segment
- `segment_content`: Transcribed content of the segment
- `view_count`: Number of views
- `like_count`: Number of likes
- `comment_count`: Number of comments
- `likes_per_1000_views`: Normalized engagement metric
- `comments_per_1000_views`: Normalized engagement metric
- `topic_category`: Category of content (e.g., Expressions, Verbs, Vocabulary)
- `language_level`: German proficiency level (A1-C2)
- `tone`: Content tone (e.g., Didactic, Conversational)
- `feedback_type`: Type of feedback provided
- `is_teaching`: Whether the segment is teaching content
- `keywords`: JSON array of keywords for the segment

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
5. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Data source: Rieke Neue German Learning Videos
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)