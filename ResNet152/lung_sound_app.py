import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import tempfile
import os
import pandas as pd

# Set page title
st.set_page_config(page_title="Lung Sound Classifier", layout="wide")

# Add a sidebar for debug mode
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Load your trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('lung_sound_model1.h5')
        if debug_mode:
            st.sidebar.success("Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# Load label encoder
@st.cache_resource
def load_encoder():
    try:
        import pickle
        with open('label_encoder1.pkl', 'rb') as f:
            encoder = pickle.load(f)
        if debug_mode:
            st.sidebar.success("Label encoder loaded successfully")
            st.sidebar.write(f"Classes: {encoder.classes_}")
        return encoder
    except Exception as e:
        st.sidebar.error(f"Error loading encoder: {e}")
        return None

# Preprocessing function
def preprocess_audio(file_path, is_temp=False):
    # Load audio
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        if debug_mode:
            st.write(f"✅ Audio loaded successfully: {len(audio)} samples, {sr}Hz")
    except Exception as e:
        st.error(f"❌ Error loading audio: {e}")
        return None, None, None
    
    # Display audio waveform
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(np.arange(len(audio)) / sr, audio)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    st.pyplot(fig)
    
    # Extract mel spectrogram
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if debug_mode:
            st.write(f"✅ Spectrogram created: shape {mel_spec_db.shape}")
    except Exception as e:
        st.error(f"❌ Error creating spectrogram: {e}")
        return None, None, None
    
    # Original spectrogram for visualization
    original_spec = mel_spec_db.copy()
    
    # Reshape to 224x224
    target_shape = (224, 224)
    resized = np.zeros(target_shape)
    h = min(mel_spec_db.shape[0], target_shape[0])
    w = min(mel_spec_db.shape[1], target_shape[1])
    resized[:h, :w] = mel_spec_db[:h, :w]
    
    # Check if resizing worked
    if np.all(resized == 0):
        st.warning("⚠️ Resized spectrogram is all zeros!")
    
    # Normalize the data
    min_val = np.min(resized)
    max_val = np.max(resized)
    if max_val > min_val:
        resized = (resized - min_val) / (max_val - min_val)
        if debug_mode:
            st.write(f"✅ Normalized range: [{np.min(resized):.4f}, {np.max(resized):.4f}]")
    else:
        st.error("❌ Cannot normalize: min and max values are equal")
        return None, None, None
    
    # Convert to RGB
    mel_spec_rgb = np.stack((resized,)*3, axis=-1)
    
    # Add batch dimension
    mel_spec_rgb = np.expand_dims(mel_spec_rgb, axis=0)
    if debug_mode:
        st.write(f"✅ Final input shape: {mel_spec_rgb.shape}")
    
    return mel_spec_rgb, original_spec, resized

# Load model and encoder
model = load_model()
label_encoder = load_encoder()

# Check if model and encoder loaded correctly
if model is None or label_encoder is None:
    st.error("Failed to load model or label encoder. Please check the files exist and are not corrupted.")
    st.stop()

# Streamlit App
st.title("Lung Sound Classification App 🎧")
st.write("Upload a lung sound .wav file for classification")

# Create tabs for different functions
tab1, tab2 = st.tabs(["Classifier", "Model Diagnostics"])

with tab1:
    uploaded_file = st.file_uploader("Upload a lung sound .wav file", type=["wav"])
    
    if uploaded_file is not None:
        # Play the audio
        st.audio(uploaded_file, format="audio/wav")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        st.write("### Processing Audio")
        
        # Process the audio file
        input_data, original_spec, resized_spec = preprocess_audio(temp_path, is_temp=True)
        
        if input_data is not None:
            # Visualize spectrograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Mel Spectrogram")
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(original_spec, x_axis='time', y_axis='mel', sr=22050, ax=ax)
                plt.colorbar(img, ax=ax, format="%+2.f dB")
                st.pyplot(fig)
            
            with col2:
                st.write("Processed Spectrogram (Model Input)")
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(resized_spec, x_axis='time', y_axis='linear', sr=22050, ax=ax)
                plt.colorbar(img, ax=ax, format="%+2.f dB")
                st.pyplot(fig)
            
            # Make prediction
            st.write("### Prediction Results")
            with st.spinner("Running model prediction..."):
                try:
                    prediction = model.predict(input_data)
                    
                    if debug_mode:
                        st.write("Raw prediction output:")
                        st.write(prediction)
                    
                    # Display prediction results
                    predicted_class = np.argmax(prediction, axis=1)
                    
                    # IMPORTANT FIX: Ensure we're correctly transforming the predicted class
                    predicted_label = label_encoder.inverse_transform(predicted_class)
                    
                    if debug_mode:
                        st.write(f"Predicted class index: {predicted_class[0]}")
                        st.write(f"Available classes: {label_encoder.classes_}")
                        st.write(f"Transformed to label: {predicted_label[0]}")
                    
                    # Show all class probabilities
                    st.write("#### Class Probabilities:")
                    
                    # IMPORTANT FIX: Make sure we're creating the dataframe correctly
                    class_names = label_encoder.classes_
                    probs = prediction[0] * 100
                    probs_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probs
                    }).sort_values('Probability', ascending=False)
                    
                    if debug_mode:
                        st.write("DataFrame created for visualization:")
                        st.write(probs_df)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(probs_df['Class'], probs_df['Probability'])
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('Prediction Probabilities')
                    
                    # Highlight the predicted class
                    for i, bar in enumerate(bars):
                        if probs_df.iloc[i]['Class'] == predicted_label[0]:
                            bar.set_color('red')
                    
                    st.pyplot(fig)
                    
                    # Show final prediction
                    st.success(f"Prediction: **{predicted_label[0]}**")
                    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    if debug_mode:
                        import traceback
                        st.code(traceback.format_exc())
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

with tab2:
    st.write("### Model Diagnostics")
    
    # Show model summary
    st.write("#### Model Architecture")
    
    # Get model info
    model_input_shape = model.input_shape
    model_output_shape = model.output_shape
    
    st.write(f"Input shape: {model_input_shape}")
    st.write(f"Output shape: {model_output_shape}")
    st.write(f"Number of classes: {len(label_encoder.classes_)}")
    st.write(f"Classes: {label_encoder.classes_}")
    
    # Test with random data
    st.write("#### Test with Random Data")
    if st.button("Test model with random input"):
        random_input = np.random.rand(1, 224, 224, 3)
        with st.spinner("Running prediction on random data..."):
            try:
                random_prediction = model.predict(random_input)
                
                st.write("Random input prediction (raw):")
                st.write(random_prediction)
                
                random_class = np.argmax(random_prediction, axis=1)
                random_label = label_encoder.inverse_transform(random_class)
                
                st.write(f"Predicted class index: {random_class[0]}")
                st.write(f"Predicted class: {random_label[0]}")
                
                # Show all class probabilities for random input
                st.write("#### Class Probabilities for Random Input:")
                rand_probs_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Probability': random_prediction[0] * 100
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(rand_probs_df)
                
            except Exception as e:
                st.error(f"Error during random prediction: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())