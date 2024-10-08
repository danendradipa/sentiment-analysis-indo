import streamlit as st
import joblib

# Load the trained pipeline
pipeline = joblib.load('sentiment_week7/sentiment_pipeline.joblib')

def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                # Predict sentiment of the entire text using the pipeline
                sentiment = pipeline.predict([raw_text])[0]
                st.write(f"Predicted Sentiment: {sentiment}")

                # Emoji based on prediction
                if sentiment == "positive":
                    st.markdown("# Positive 😃")
                elif sentiment == "negative":
                    st.markdown("# Negative 😡")
                else:
                    st.markdown("# Neutral 😐")

            with col2:
                st.info("Token Sentiment")
                # Analyze sentiment for individual tokens
                token_sentiments = analyze_token_sentiment(raw_text, pipeline)
                st.write(token_sentiments)

    else:
        st.subheader("About")

def analyze_token_sentiment(docx, pipeline):
    pos_list = []
    neg_list = []
    neu_list = []

    for word in docx.split():
        # Predict sentiment for each token using the trained pipeline
        prediction = pipeline.predict([word])[0]

        # Check if the pipeline supports predict_proba
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba([word])
            proba_max = proba.max()  # Get the highest probability
        else:
            proba_max = None  # If predict_proba is not available, return None

        if prediction == 'positive':
            pos_list.append([word, proba_max])
        elif prediction == 'negative':
            neg_list.append([word, proba_max])
        else:
            neu_list.append(word)
    
    result = {
        'positives': pos_list,
        'negatives': neg_list,
        'neutral': neu_list,
    }
    return result

if __name__ == '__main__':
    main()
