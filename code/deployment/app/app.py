import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Art Auction Price Predictor", page_icon="🎨")

st.title("🎨 Art Auction Price Prediction")
st.write("Predict the auction price of artworks based on their features")

# Input form
with st.form("prediction_form"):
    st.subheader("Artwork Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox("Period", ["Contemporary", "Post-War", "Modern", "19th Century"])
        movement = st.selectbox("Art Movement", ["Abstract", "Baroque", "Surrealism", "Expressionism"])
        size_category = st.selectbox("Size Category", ["Small", "Medium", "Large"])
    
    with col2:
        creation_year = st.number_input("Creation Year", min_value=1700, max_value=2025, value=2000)
        signed = st.selectbox("Signed", ["Yes", "No"])
    
    signed_binary = 1 if signed == "Yes" else 0
    
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Prepare data for API
        features = {
            "period": period,
            "movement": movement,
            "size_category": size_category,
            "creation_year": creation_year,
            "signed_binary": signed_binary
        }
        
        try:
            # Make prediction request
            response = requests.post("http://api:8000/predict", json=features)
            
            if response.status_code == 200:
                result = response.json()
                if "predicted_price_usd" in result:
                    predicted_price = result["predicted_price_usd"]
                    st.success(f"Predicted Auction Price: **${predicted_price:,.2f}**")
                else:
                    st.error(f"Prediction error: {result.get('error', 'Unknown error')}")
            else:
                st.error("Failed to connect to prediction API")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

st.markdown("---")
st.subheader("How to use:")
st.write("1. Fill in the artwork details")
st.write("2. Click 'Predict Price'")
st.write("3. View the predicted auction price")

st.subheader("Sample Predictions:")
sample_data = pd.DataFrame({
    "Feature": ["Contemporary Abstract (Medium, Signed, 2020)", "Post-War Surrealism (Large, Not Signed, 1950)"],
    "Typical Price Range": ["$50 - $500", "$1,000 - $10,000"]
})
st.table(sample_data)