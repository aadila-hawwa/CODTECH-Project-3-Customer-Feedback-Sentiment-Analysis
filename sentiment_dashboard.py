import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Feedback Sentiment Dashboard", layout="wide")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("cleaned_feedback_data.csv")

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Customer Feedback Sentiment Analysis Dashboard")

total_reviews = len(df)
positive_reviews = len(df[df["Sentiment"] == 1])
negative_reviews = len(df[df["Sentiment"] == 0])
positive_percentage = positive_reviews / total_reviews * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", total_reviews)
col2.metric("Positive Reviews", positive_reviews)
col3.metric("Negative Reviews", negative_reviews)
col4.metric("Positive %", f"{positive_percentage:.2f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Overview",
    "Word Analysis",
    "Prediction",
    "Data Preview"
])

with tab1:
    sentiment_count = df["Sentiment_Label"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]

    fig1 = px.pie(
        sentiment_count,
        values="Count",
        names="Sentiment",
        title="Positive vs Negative Feedback"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        title="Sentiment Count"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    selected_sentiment = st.selectbox(
        "Select Sentiment",
        ["Positive", "Negative"]
    )

    if selected_sentiment == "Positive":
        reviews = df[df["Sentiment"] == 1]["Cleaned_Review"]
    else:
        reviews = df[df["Sentiment"] == 0]["Cleaned_Review"]

    reviews = reviews.dropna().astype(str)
    text = " ".join(reviews)

    if text.strip() == "":
        st.warning("No text available for word cloud.")
    else:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
with tab3:
    st.subheader("Predict Sentiment for New Feedback")

    user_review = st.text_area("Enter customer feedback")

    if st.button("Predict Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter feedback text.")
        else:
            cleaned_review = clean_text(user_review)
            vectorized_review = vectorizer.transform([cleaned_review])
            prediction = model.predict(vectorized_review)[0]
            probability = model.predict_proba(vectorized_review)[0]

            if prediction == 1:
                st.success(f"Predicted Sentiment: Positive")
                st.write(f"Confidence: {probability[1] * 100:.2f}%")
            else:
                st.error(f"Predicted Sentiment: Negative")
                st.write(f"Confidence: {probability[0] * 100:.2f}%")

with tab4:
    st.dataframe(df, use_container_width=True)