import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from textblob import TextBlob  # For sentiment analysis

# Set page config
st.set_page_config(page_title="Twitter Analytics", layout="wide")

# Load data with caching
@st.cache_data
def load_data():
    tweets = pd.read_csv('account_analytics_content_2024-11-21_2025-02-19.csv')
    account = pd.read_csv('account_overview_analytics.csv')
    return tweets, account

tweets_sheet, account_analytics = load_data()

# Preprocess data
account_analytics['followers'] = (account_analytics['New follows'] - account_analytics['Unfollows']).cumsum()
tweets_sheet['Date'] = pd.to_datetime(tweets_sheet['Date'])
tweets_sheet['tweet_length'] = tweets_sheet['Post text'].apply(lambda x: len(str(x)))

# Sidebar controls
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Select date range", 
                                 [tweets_sheet['Date'].min(), tweets_sheet['Date'].max()])

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

# Time series of engagement metrics
fig_engagement = go.Figure()
metrics = ['Likes', 'Reposts', 'Replies', 'Bookmarks']
colors = ['#FF006E', '#8338EC', '#3A86FF', '#FFBE0B']

for metric, color in zip(metrics, colors):
    fig_engagement.add_trace(go.Scatter(
        x=tweets_sheet['Date'], 
        y=tweets_sheet[metric],
        name=metric,
        line=dict(color=color),
        stackgroup='one'  # Remove for individual lines
    ))

fig_engagement.update_layout(title="Engagement Metrics Over Time",
                           xaxis_title="Date", yaxis_title="Count")
st.plotly_chart(fig_engagement, use_container_width=True)

# Content Analysis Section
st.header("Content Analysis")

# Extract mentions and hashtags
def extract_features(text, pattern):
    return re.findall(pattern, str(text))

tweets_sheet['mentions'] = tweets_sheet['Post text'].apply(
    lambda x: extract_features(x, r'@(\w+)'))
tweets_sheet['hashtags'] = tweets_sheet['Post text'].apply(
    lambda x: extract_features(x, r'#(\w+)'))

# Top mentions analysis
mentions_counts = pd.Series([mention for sublist in tweets_sheet['mentions'] for mention in sublist]).value_counts().head(10)
fig_mentions = px.bar(mentions_counts, 
                     labels={'index': 'Mention', 'value': 'Count'},
                     title="Top 10 Mentions")
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