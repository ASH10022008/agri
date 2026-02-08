import streamlit as st
import joblib
import numpy as np

# --- 1. LOAD THE BRAIN ---
# This looks for the files you created in Phase 1
try:
    model = joblib.load('crop_model.pkl')
    encoder = joblib.load('crop_encoder.pkl')
except:
    st.error("Error: 'crop_model.pkl' or 'crop_encoder.pkl' not found! Run your training script first.")

# --- 2. DESIGN THE INTERFACE ---
st.set_page_config(page_title="AgroPredict AI", page_icon="🌾")

st.title("🌾 AgroPredict: Smart Yield Engine")
st.markdown("Enter your farm details below to get an AI-driven yield forecast.")

# Sidebar for organization
st.sidebar.header("Farm Configuration")

# Get the list of crops from our encoder so the dropdown is accurate
crop_list = encoder.classes_
selected_crop = st.sidebar.selectbox("Select Crop Type", crop_list)

# Sliders for easy input
rain = st.sidebar.slider("Annual Rainfall (mm)", 200, 3000, 1000)
temp = st.sidebar.slider("Average Temperature (°C)", 10, 50, 25)

# --- 3. THE PREDICTION LOGIC ---
if st.button("Calculate Predicted Yield"):
    # Convert the selected crop name back into a number for the AI
    crop_num = encoder.transform([selected_crop])[0]
    
    # Arrange inputs in the EXACT order your model was trained on: [crop, rain, temp]
    input_data = np.array([[crop_num, rain, temp]])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Show results with style
    st.balloons()
    st.success(f"### Estimated Yield: {prediction:.2f} Tons per Hectare")
    
    # Practical advice (Great for winning points!)
    st.write("---")
    st.subheader("💡 Farmer Advisory")
    if temp > 35:
        st.warning("High temperature detected. Ensure adequate irrigation to prevent crop wilting.")
    elif rain < 500:
        st.warning("Low rainfall predicted. Consider drought-resistant farming techniques.")
    else:
        st.info("Weather conditions look optimal for the selected crop.")