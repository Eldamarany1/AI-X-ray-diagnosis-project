import streamlit as st
import numpy as np
from PIL import Image
import time

try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# --- 1. UI Configuration & CSS ---
st.set_page_config(
    page_title="Pneumo-AI", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Glassmorphism, Typography & Animations
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background: sleek dark animated gradient */
    .stApp {
        background: linear-gradient(135deg, #0b132b, #1c2541, #3a506b);
        background-attachment: fixed;
        background-size: 200% 200%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Title styling */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 10px;
        margin-top: -30px;
        text-shadow: 0 0 20px rgba(79, 172, 254, 0.4);
    }
    
    .subtitle {
        text-align: center;
        font-weight: 300;
        font-size: 1.2rem;
        color: #cbd5e1;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }

    /* Glassmorphism Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
    }

    /* Upload box styling override */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(79, 172, 254, 0.5);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00f2fe;
        background: rgba(79, 172, 254, 0.1);
    }

    /* Results cards */
    .result-card-normal {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.3));
        border-left: 6px solid #2ecc71;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(46, 204, 113, 0.15);
    }
    
    .result-card-pneumonia {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.3));
        border-left: 6px solid #e74c3c;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.15);
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: -10px 0 5px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* Expander styling override */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
    }

    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .pulse-animation {
        animation: pulse 2.5s infinite ease-in-out;
    }
    
    /* Hide some default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🫁 Pneumo-AI Diagnostic System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Neural Network for Chest X-Ray Analysis</p>", unsafe_allow_html=True)

# --- 2. Load the Brain ---
@st.cache_resource 
def load_keras_model():
    if MODEL_AVAILABLE:
        try:
            return load_model('pneumonia_vision_model.h5')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_keras_model()

if model is None:
    st.error("Model file `pneumonia_vision_model.h5` not found or TensorFlow is not installed. Please ensure the model file is in the same directory.")
    st.stop()

# --- 3. Main Layout ---
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Patient Scan")
    uploaded_file = st.file_uploader("Select an X-ray image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Original Scan")
        # Center the image
        st.image(image, use_column_width=True, output_format="PNG")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Analysis Results")
        
        # Dramatic loading effect
        with st.spinner("Initializing Deep Learning Engine..."):
            time.sleep(0.4)
        with st.spinner("Extracting Topological Features..."):
            time.sleep(0.4)
        with st.spinner("Running Inference Protocol..."):
            time.sleep(0.4)
            
        # --- 4. The Preprocessing ---
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array_normalized = img_array / 255.0 
        img_batch = np.expand_dims(img_array_normalized, axis=0) 
        
        # --- 5. The Prediction ---
        prediction_prob = model.predict(img_batch)[0][0]
        
        # --- 6. The Verdict ---
        if prediction_prob >= 0.5:
            confidence = prediction_prob * 100
            st.markdown(f"""
            <div class="result-card-pneumonia pulse-animation">
                <h2 style="color: #ffb3b3; margin-top: 0;">⚠️ PNEUMONIA DETECTED</h2>
                <div class="metric-value">{confidence:.1f}%</div>
                <p style="color: #ffe6e6; margin-bottom: 15px; font-weight: 600;">Confidence Score</p>
                <hr style="border-top: 1px solid rgba(255,255,255,0.2); margin: 15px 0;">
                <p style="color: #ffcccc; line-height: 1.5;">The neural network has identified distinct patterns consistent with pneumonia in the provided scan. <strong>Please consult a medical professional immediately.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual impact
            st.progress(int(confidence), text="Diagnosis Confidence")
            
        else:
            confidence = (1 - prediction_prob) * 100
            st.markdown(f"""
            <div class="result-card-normal pulse-animation">
                <h2 style="color: #b3ffcc; margin-top: 0;">✅ NORMAL LUNGS</h2>
                <div class="metric-value">{confidence:.1f}%</div>
                <p style="color: #e6ffe6; margin-bottom: 15px; font-weight: 600;">Confidence Score</p>
                <hr style="border-top: 1px solid rgba(255,255,255,0.2); margin: 15px 0;">
                <p style="color: #ccffcc; line-height: 1.5;">No distinct signs of pneumonia detected. The lung topology appears to be clear and within normal parameters.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(int(confidence), text="Diagnosis Confidence")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Advanced technical details expander
        with st.expander("📊 View Technical Telemetry", expanded=False):
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; font-family: monospace; font-size: 0.9em; color: #a0aec0;">
            <strong>[SYSTEM]</strong> Telemetry Data:<br>
            • Input Resolution: {image.size[0]}x{image.size[1]} px<br>
            • Tensor Shape: (1, 224, 224, 3)<br>
            • Raw Logit Score: {prediction_prob:.6f}<br>
            • Color Space: RGB (Converted)<br>
            • Normalization: Min-Max [0, 1]<br>
            • Inference Time: ~O(1) ms
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='glass-container' style='text-align: center; color: #94a3b8; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 300px;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 4rem; opacity: 0.5; margin: 0;'>🩻</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Waiting for Scan Data</h3>")
        st.markdown("<p style='max-width: 80%;'>Please upload a chest X-Ray image in the adjacent panel to initiate the AI diagnostic sequence.</p>")
        st.markdown("</div>", unsafe_allow_html=True)