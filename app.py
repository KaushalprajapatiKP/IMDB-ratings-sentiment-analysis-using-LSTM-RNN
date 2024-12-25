import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model from the .keras file
model = load_model('imdb_sentiment_annalysis.keras')

# Load the tokenizer from the .pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess and predict sentiment
def predict_sentiment(review):
    # Tokenize and pad the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    
    # Predict sentiment
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Streamlit UI with classy styling
st.markdown("""
    <style>
        /* Body styling */
        body {
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
            margin: 0;
            padding: 0;
        }
        
        /* Title section */
        .title {
            text-align: center;
            color: #ffffff;
            font-size: 48px;
            font-weight: 700;
            padding: 40px;
            margin-top: 30px;
            background-color: #333333;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        }
        
        /* Main container */
        .container {
            text-align: center;
            background-color: #252525;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            margin-top: 100px;  /* Increased margin-top for more gap */
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Text area styling */
        .textarea {
            width: 80%;
            height: 180px;
            border-radius: 8px;
            padding: 15px;
            font-size: 18px;
            border: 1px solid #444444;
            background-color: #333333;
            color: #f0f0f0;
            margin-bottom: 30px;
            transition: border-color 0.3s ease;
        }
        
        .textarea:focus {
            border-color: #ff6f61;
            outline: none;
        }
        
        /* Button styling */
        .button {
            background-color: #ff6f61;
            color: white;
            padding: 15px 30px;
            font-size: 20px;
            border-radius: 8px;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        
        .button:hover {
            background-color: #ff3b2f;
            transform: scale(1.05);
        }
        
        /* Result styling */
        .result {
            font-size: 24px;
            font-weight: 600;
            margin-top: 30px;
        }
        
        .positive {
            color: #32cd32;  /* Lime green for positive sentiment */
        }
        
        .negative {
            color: #ff6347;  /* Tomato red for negative sentiment */
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            font-size: 14px;
            color: #aaa;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">IMDB Sentiment Analysis</div>', unsafe_allow_html=True)

# Add extra spacing after the title to increase the gap
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# Main container for input and prediction
with st.container():
    # Input text from user
    user_review = st.text_area("Enter your review:", key="review", height=180, max_chars=1000, placeholder="Write your movie review here...", label_visibility="collapsed")

    # When the button is clicked, predict sentiment
    if st.button("Predict Sentiment", key="predict", use_container_width=True):
        if user_review:
            sentiment = predict_sentiment(user_review)
            sentiment_class = "positive" if sentiment == "positive" else "negative"
            st.markdown(f'<div class="result {sentiment_class}">The sentiment of the review is: <span>{sentiment}</span></div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to predict sentiment.")

# Footer
st.markdown('<div class="footer">Made  by Kaushal Prajapati | Your feedback is valuable!</div>', unsafe_allow_html=True)
