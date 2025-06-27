import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import joblib

# Set page config
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler (replace with your actual loading code)
@st.cache_resource
def load_assets():
    # In a real app, you would load your trained model and scaler here
    # model = load_model('weather_model.h5')
    # scaler = joblib.load('scaler.pkl')
    # features = ['tp', 'ssrd', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sp']
    # return model, scaler, features
    
    # For demo purposes, we'll return None and use dummy data
    return None, None, ['tp', 'ssrd', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sp']

model, scaler, features = load_assets()

# App title
st.title("🌦️ Advanced Weather Prediction Dashboard")

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    lookback_days = st.slider("Lookback Period (days)", 1, 14, 7)
    forecast_days = st.slider("Forecast Period (days)", 1, 7, 3)
    selected_feature = st.selectbox("Feature to Visualize", features)
    
    st.markdown("---")
    st.markdown("""
    ### About
    This dashboard predicts weather parameters using an LSTM-Attention model.
    - **Lookback**: Historical data used for prediction
    - **Forecast**: Future period to predict
    """)

# Generate sample data (replace with your actual data loading)
def generate_sample_data():
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='H')
    data = pd.DataFrame({
        'valid_time': dates,
        'tp': np.random.uniform(0, 5, 1000),
        'ssrd': np.random.uniform(0, 300, 1000),
        'u10': np.random.uniform(-20, 20, 1000),
        'v10': np.random.uniform(-20, 20, 1000),
        'd2m': np.random.uniform(250, 320, 1000),
        't2m': np.random.uniform(250, 320, 1000),
        'msl': np.random.uniform(98000, 105000, 1000),
        'sp': np.random.uniform(98000, 105000, 1000)
    })
    return data

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Historical Analysis", "ℹ️ Model Info"])

with tab1:
    st.header("Weather Forecast")
    
    # Load data (replace with your actual data)
    data = generate_sample_data()
    
    # Display raw data
    with st.expander("Show Raw Data"):
        st.dataframe(data.tail(24*7))  # Show last week of data
    
    # Create plot
    fig = px.line(
        data.tail(24*lookback_days),
        x='valid_time',
        y=selected_feature,
        title=f"Historical {selected_feature} (Last {lookback_days} days)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction placeholder (replace with actual prediction)
    st.subheader("Predicted Values")
    future_dates = pd.date_range(
        start=data['valid_time'].iloc[-1] + timedelta(hours=1),
        periods=24*forecast_days,
        freq='H'
    )
    pred_data = pd.DataFrame({
        'valid_time': future_dates,
        selected_feature: np.random.uniform(
            data[selected_feature].min(),
            data[selected_feature].max(),
            24*forecast_days
        )
    })
    
    combined = pd.concat([
        data[['valid_time', selected_feature]].tail(24*3),
        pred_data
    ])
    
    fig2 = px.line(
        combined,
        x='valid_time',
        y=selected_feature,
        color=combined.index > len(data.tail(24*3)),
        title=f"Historical vs Predicted {selected_feature}",
        labels={'color': 'Type'}
    )
    fig2.update_layout(showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Historical Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Feature 1", features, index=0)
    with col2:
        feature2 = st.selectbox("Feature 2", features, index=1)
    
    fig3 = px.scatter(
        data.tail(24*30),
        x=feature1,
        y=feature2,
        color='t2m',
        title=f"{feature1} vs {feature2} (Last 30 days)"
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.header("Model Information")
    st.markdown("""
    ### LSTM-Attention Weather Prediction Model
    
    This model uses:
    - **LSTM layers** to capture temporal patterns
    - **Attention mechanism** to focus on important time steps
    - **Robust scaling** to handle outliers
    
    Model architecture:
    """)
    
    # Placeholder for model summary
    st.code("""
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 24, 128)           70656     
    _________________________________________________________________
    dropout (Dropout)            (None, 24, 128)           0         
    _________________________________________________________________
    multi_head_attention (MultiH (None, 24, 128)           66048     
    _________________________________________________________________
    layer_normalization (LayerNo (None, 24, 128)           256       
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 64)                49408     
    _________________________________________________________________
    dense (Dense)                (None, 72)                4680      
    _________________________________________________________________
    reshape (Reshape)            (None, 24, 3)             0         
    =================================================================
    Total params: 191,048
    Trainable params: 191,048
    Non-trainable params: 0
    _________________________________________________________________
    """)