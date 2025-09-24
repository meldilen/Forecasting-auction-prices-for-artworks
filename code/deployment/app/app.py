import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Art Auction Price Predictor",
    layout="wide"
)

st.title("Art Auction Price Predictor")
st.markdown(
    "Predict the auction price of artworks based on their characteristics")

if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = False

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    st.session_state.api_healthy = health_response.status_code == 200
except:
    st.session_state.api_healthy = False

if st.session_state.api_healthy:
    st.success("✅ API is connected and healthy")
else:
    st.error("❌ API is not available. Please check if the API container is running.")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Artwork Details")
        creation_year = st.number_input(
            "Creation Year", min_value=1500, max_value=2024, value=2000)
        signed_binary = st.selectbox(
            "Signed", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
        avg_dimension_cm = st.number_input(
            "Average Dimension (cm)", min_value=1.0, value=50.0)
        size_category = st.selectbox("Size Category", options=[
                                     "Small", "Medium", "Large"])

    with col2:
        st.subheader("Artist Information")
        lifespan = st.number_input(
            "Artist Lifespan (years)", min_value=0, value=70)
        years_since_death = st.number_input(
            "Years Since Death", min_value=0, value=20)
        paintings = st.number_input(
            "Number of Paintings by Artist", min_value=1, value=100)
        moma_artwork_count = st.number_input(
            "MoMA Artwork Count", min_value=0, value=5)
        is_living = st.checkbox("Artist is Living", value=False)

    period = st.selectbox("Art Period", options=[
                          "Renaissance", "Baroque", "Modern", "Contemporary", "Impressionism"])
    movement = st.selectbox("Art Movement", options=[
                            "Realism", "Impressionism", "Expressionism", "Cubism", "Abstract", "Pop Art"])

    submitted = st.form_submit_button("Predict Auction Price")

if submitted:
    features = {
        "creation_year": creation_year,
        "signed_binary": signed_binary,
        "avg_dimension_cm": avg_dimension_cm,
        "lifespan": lifespan,
        "years_since_death": years_since_death,
        "paintings": paintings,
        "moma_artwork_count": moma_artwork_count,
        "is_living": is_living,
        "period": period,
        "movement": movement,
        "size_category": size_category
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=features)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed successfully!")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Auction Price",
                    value=f"${result['prediction']:,.2f}",
                    help="Estimated price in USD"
                )
            
            with col2:
                st.metric(
                    label="Confidence Level",
                    value=f"{result['confidence']*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Price Range",
                    value=f"${result['prediction']*0.8:,.0f} - ${result['prediction']*1.2:,.0f}",
                    help="Estimated price range (±20%)"
                )
            
            with st.expander("Features Used for Prediction"):
                st.json(result['features_used'])
                
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

with st.expander("Model Information"):
    try:
        model_info = requests.get(f"{API_URL}/model-info").json()
        st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.write("**Expected Features:**")
        st.json(model_info.get('features_expected', []))
    except:
        st.write("Unable to retrieve model information")