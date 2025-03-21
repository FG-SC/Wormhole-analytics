import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from textblob import TextBlob  # For sentiment analysis

# Set page config
st.set_page_config(page_title="Twitter Analytics", layout="wide")

# Load data with caching
#@st.cache_data
#def load_data():
 #   tweets = pd.read_csv('account_analytics_content_2024-11-21_2025-02-19.csv')
  #  account = pd.read_csv('account_overview_analytics.csv')
   # return tweets, account

#tweets_sheet, account_analytics = load_data()

tweets_sheet = st.file_uploader("Upload your account_analytics_content data CSV", type=["csv"])

account_analytics = st.file_uploader("Upload your account_overview_analytics performance CSV", type=["csv"])

# Preprocess data
if account_analytics is not None:
    account_analytics= pd.read_csv(account_analytics)

    account_analytics['Date'] = pd.to_datetime(account_analytics['Date'])
    account_analytics = account_analytics.sort_values(by='Date')

    account_analytics['followers'] = (account_analytics['New follows'] - account_analytics['Unfollows']).cumsum()

if tweets_sheet is not None:
    tweets_sheet = pd.read_csv(tweets_sheet)

    tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Date'])
    tweets_sheet['tweet_length'] = tweets_sheet['Post text'].apply(lambda x: len(str(x)))

    tweets_sheet = tweets_sheet.sort_values(by='Date')

if (account_analytics is not None) and (tweets_sheet is not None) :
        
    # Main content
    st.title("Twitter Analytics Dashboard")

    # Account Overview Section
    st.header("Account Overview")

    # Follower growth chart
    fig_followers = go.Figure()

    fig_followers.add_trace(go.Scatter(x=account_analytics['Date'], y=account_analytics['followers'],
                                    mode='lines', name='Followers'))
    fig_followers.update_layout(title="Follower Growth Over Time",
                            xaxis_title="Date", yaxis_title="Followers")
    st.plotly_chart(fig_followers, use_container_width=True)

    # Correlation heatmap using Plotly
    corr_matrix = account_analytics[['Impressions', 'Likes', 'Engagements', 'followers']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                        x=corr_matrix.columns, y=corr_matrix.columns)
    st.plotly_chart(fig_corr, use_container_width=True)


    # Engagement Analysis Section
    st.header("Engagement Analysis")

    # Daily aggregated engagement metrics
    daily_engagement = tweets_sheet.set_index('Date').resample('D').agg({
        'Likes': 'sum',
        'Reposts': 'sum',
        'Replies': 'sum',
        'Bookmarks': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Create cumulative engagement
    for metric in ['Likes', 'Reposts', 'Replies', 'Bookmarks']:
        daily_engagement[f'cum_{metric}'] = daily_engagement[metric].cumsum()

    # Time series of engagement metrics
    fig_engagement = go.Figure()
    metrics = ['Likes', 'Reposts', 'Replies', 'Bookmarks']
    colors = ['#FF006E', '#8338EC', '#3A86FF', '#FFBE0B']

    for metric, color in zip(metrics, colors):
        fig_engagement.add_trace(go.Scatter(
            x=daily_engagement['Date'], 
            y=daily_engagement[metric],
            name=metric,
            line=dict(color=color),
            stackgroup='one'
        ))

    fig_engagement.update_layout(title="Daily Engagement Metrics",
                            xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig_engagement, use_container_width=True)

    # Engagement Rate Analysis
    st.subheader("Engagement Rate Analysis")
    tweets_sheet['engagement_rate'] = tweets_sheet['Engagements']/tweets_sheet['Impressions']

    # Calculate mean engagement rate
    mean_engagement_rate = tweets_sheet['engagement_rate'].mean()

    # Create the histogram with Plotly
    fig_engagement_rate = go.Figure()

    # Add histogram trace
    fig_engagement_rate.add_trace(go.Histogram(
        x=tweets_sheet['engagement_rate'],
        nbinsx=50,
        name='Engagement Rate',
        marker_color='#1f77b4'
    ))

    # Add mean line
    fig_engagement_rate.add_vline(
        x=mean_engagement_rate,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_engagement_rate:.2%}",
        annotation_position="top right"
    )

    # Update layout
    fig_engagement_rate.update_layout(
        title="Distribution of Engagement Rates",
        xaxis_title="Engagement Rate",
        yaxis_title="Count",
        showlegend=False
    )

    st.plotly_chart(fig_engagement_rate, use_container_width=True)
    # Content Analysis Section
    st.header("Content Analysis")

    # Extract mentions and hashtags
    def extract_features(text, pattern):
        return [mention for mention in re.findall(pattern, str(text))]

    # Update the extract_features function to handle case sensitivity
    def extract_features2(text, pattern):
        return [mention.lower() for mention in re.findall(pattern, str(text))]

    tweets_sheet['mentions'] = tweets_sheet['Post text'].apply(
        lambda x: extract_features2(x, r'@(\w+)'))
    tweets_sheet['hashtags'] = tweets_sheet['Post text'].apply(
        lambda x: extract_features(x, r'#(\w+)'))


    # Top mentions analysis
    mentions_counts = pd.Series([mention for sublist in tweets_sheet['mentions'] for mention in sublist]).value_counts().head(20)

    fig_mentions = px.bar(mentions_counts, 
                        labels={'index': 'Mention', 'value': 'Count'},
                        title="Top 20 Mentions")
    st.plotly_chart(fig_mentions, use_container_width=True)

    # Impact of mentions on impressions
    mention_effect = []
    for mention in mentions_counts.index:
        avg_impression = tweets_sheet[tweets_sheet['mentions'].apply(lambda x: mention in x)]['Impressions'].mean()
        mention_effect.append({'Mention': mention, 'Avg Impressions': avg_impression})

    mention_effect_df = pd.DataFrame(mention_effect)


    fig_mention_impact = px.bar(mention_effect_df, x='Mention', y='Avg Impressions',
                            title="Average Impressions by Mention")
    st.plotly_chart(fig_mention_impact, use_container_width=True)

    # Tweet Length vs Engagement Analysis
    fig_length = px.scatter(tweets_sheet, x='tweet_length', y='Likes',
                        trendline="lowess",
                        title="Tweet Length vs Likes",
                        labels={'tweet_length': 'Tweet Length (characters)'})
    st.plotly_chart(fig_length, use_container_width=True)

    # Sentiment Analysis (requires TextBlob)
    tweets_sheet['sentiment'] = tweets_sheet['Post text'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity)
    fig_sentiment = px.histogram(tweets_sheet, x='sentiment',
                                title="Sentiment Distribution",
                                nbins=20)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Top Performing Tweets
    st.subheader("Top Performing Tweets")
    top_tweets = tweets_sheet.sort_values('Likes', ascending=False).head(5)[['Date', 'Post text', 'Likes', 'Reposts']]
    st.dataframe(top_tweets, use_container_width=True)
