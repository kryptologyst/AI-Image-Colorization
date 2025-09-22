import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Import our colorization model
from modern_colorization import ColorizationUNet, lab_to_rgb

# Page configuration
st.set_page_config(
    page_title="🎨 AI Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the colorization model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ColorizationUNet()
    
    # Try to load pre-trained weights
    if os.path.exists('best_colorization_model.pth'):
        model.load_state_dict(torch.load('best_colorization_model.pth', map_location=device))
        st.success("✅ Pre-trained model loaded successfully!")
    else:
        st.warning("⚠️ No pre-trained model found. Using random weights.")
    
    model.eval()
    return model.to(device)

def preprocess_image(image):
    """Preprocess uploaded image for colorization"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGB to LAB
    lab_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Extract L channel and normalize
    L = lab_image[:, :, 0].astype(np.float32) / 100.0
    
    # Resize to model input size (256x256)
    L_resized = cv2.resize(L, (256, 256))
    
    # Convert to tensor
    L_tensor = torch.tensor(L_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return L_tensor, img_array, lab_image

def colorize_image(model, L_tensor, device):
    """Colorize grayscale image using the model"""
    with torch.no_grad():
        L_tensor = L_tensor.to(device)
        AB_pred = model(L_tensor)
        
        # Convert back to numpy
        L_np = L_tensor[0, 0].cpu().numpy()
        AB_np = AB_pred[0].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        L_np = L_np * 100
        AB_np[:, :, 0] = AB_np[:, :, 0] * 128 + 128
        AB_np[:, :, 1] = AB_np[:, :, 1] * 128 + 128
        
        # Convert LAB to RGB
        colorized_rgb = lab_to_rgb(L_np, AB_np)
        
        return colorized_rgb

def calculate_metrics(original, colorized):
    """Calculate colorization quality metrics"""
    # Convert to grayscale for comparison
    orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    color_gray = cv2.cvtColor((colorized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate metrics
    mse = np.mean((orig_gray - color_gray) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Colorfulness metric
    colorfulness = np.std(colorized) * np.sqrt(3)
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'Colorfulness': colorfulness
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Colorization</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Choose Model",
        ["Modern U-Net", "Legacy OpenCV", "GAN-based (Coming Soon)"],
        index=0
    )
    
    # Advanced settings
    st.sidebar.subheader("🔧 Advanced Settings")
    
    enhance_colors = st.sidebar.checkbox("Enhance Colors", value=True)
    color_saturation = st.sidebar.slider("Color Saturation", 0.5, 2.0, 1.0, 0.1)
    
    # Load model
    if model_option == "Modern U-Net":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model()
        st.sidebar.success(f"Using device: {device}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color image to colorize"
        )
        
        # Sample images
        st.subheader("🖼️ Sample Images")
        sample_images = [
            "https://via.placeholder.com/300x200/cccccc/666666?text=Sample+1",
            "https://via.placeholder.com/300x200/cccccc/666666?text=Sample+2",
            "https://via.placeholder.com/300x200/cccccc/666666?text=Sample+3"
        ]
        
        selected_sample = st.selectbox("Choose a sample image", ["None"] + [f"Sample {i+1}" for i in range(len(sample_images))])
        
        if selected_sample != "None":
            sample_idx = int(selected_sample.split()[-1]) - 1
            st.image(sample_images[sample_idx], caption=f"Sample {sample_idx + 1}", use_column_width=True)
    
    with col2:
        st.subheader("🎨 Colorization Results")
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            
            # Display original
            st.subheader("Original Image")
            st.image(image, caption="Original", use_column_width=True)
            
            # Colorize button
            if st.button("🎨 Colorize Image", type="primary"):
                with st.spinner("Colorizing image..."):
                    # Preprocess
                    L_tensor, original_array, lab_image = preprocess_image(image)
                    
                    # Colorize
                    start_time = time.time()
                    colorized_rgb = colorize_image(model, L_tensor, device)
                    processing_time = time.time() - start_time
                    
                    # Apply enhancements
                    if enhance_colors:
                        colorized_rgb = np.clip(colorized_rgb * color_saturation, 0, 1)
                    
                    # Display results
                    st.subheader("Colorized Image")
                    st.image(colorized_rgb, caption="Colorized", use_column_width=True)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(original_array, colorized_rgb)
                    
                    # Display metrics
                    st.subheader("📊 Quality Metrics")
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                    
                    with col_metric2:
                        st.metric("MSE", f"{metrics['MSE']:.2f}")
                    
                    with col_metric3:
                        st.metric("Colorfulness", f"{metrics['Colorfulness']:.3f}")
                    
                    st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    
                    # Download button
                    colorized_pil = Image.fromarray((colorized_rgb * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    colorized_pil.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 Download Colorized Image",
                        data=byte_im,
                        file_name="colorized_image.png",
                        mime="image/png"
                    )
        
        else:
            st.info("👆 Please upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Powered by PyTorch & Streamlit | Built with modern deep learning techniques</p>
        <p>💡 Upload any grayscale or color image to see AI-powered colorization in action!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
