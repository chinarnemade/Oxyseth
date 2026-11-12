🚑 Oxyseth - Respiratory Sound Classification 🩺
Oxysteth is an AI-powered diagnostic tool designed to classify lung sounds into three primary categories: Normal, Crackles, and Wheezes. It leverages deep learning techniques like Convolutional Neural Networks (CNNs) and Transfer Learning (ResNet152) on lung sound spectrograms to assist healthcare professionals—especially in low-resource settings—with fast, reliable, and automated respiratory sound analysis.

📚 Table of Contents
🎯 Motivation

❗ Problem Statement

✨ Key Features

🏗️ System Architecture

📁 Dataset

🛠️ Technologies Used

📊 Results

🚀 Future Work

⚙️ How to Run

👥 Team

🔗 Live Preview

🎯 Motivation
Respiratory illnesses such as asthma, pneumonia, and COPD are among the leading causes of death worldwide. Diagnosing these conditions in resource-constrained environments is challenging due to limited access to skilled clinicians and diagnostic equipment.
Oxysteth addresses this by enabling early detection and monitoring through a deep learning-based classification of lung sounds.

❗ Problem Statement
Design and implement a deep learning model that can accurately classify lung sounds into:

Normal

Crackles

Wheezes

The model should:

Handle varying environmental noise and recording conditions

Be scalable for real-time, mobile, and telemedicine deployments

✨ Key Features
🎧 Converts lung audio into spectrograms for visual analysis

🔍 Uses ResNet152 with Transfer Learning for advanced feature extraction

🎯 Employs differential learning rates for effective fine-tuning

📈 Delivers high accuracy, sensitivity, and specificity

🌐 Supports real-time, cloud-based, and mobile deployment

🏗️ System Architecture

Audio Input  
   ↓  
Preprocessing (Spectrogram Generation)  
   ↓  
Feature Extraction (ResNet152)  
   ↓  
Classification Layers  
   ↓  
Output: Normal / Crackles / Wheezes
🔄 Data Augmentation Techniques:

Time Stretching

Pitch Shifting

Noise Injection

📊 Performance Metrics:

Accuracy

Sensitivity

Specificity

F1-score

📁 Dataset
Source: ICBHI 2017 Respiratory Sound Database

Real-world lung sound recordings labeled by medical experts

Categorized into: Normal, Crackles, Wheezes, and Both

🛠️ Technologies Used
Language: Python 🐍

Deep Learning: TensorFlow / Keras

Audio Processing: Librosa, OpenCV

Model Evaluation: Scikit-learn

Data Visualization: Matplotlib, Seaborn

📊 Results 📈
Model	Accuracy	Sensitivity	Specificity
ResNet50	74.68%	0.68	0.83
ResNet101	~	~	~
ResNet152	89.19%	0.91	0.97

✅ F1-score and Precision-Recall Curves were also analyzed.
✅ Model maintained robustness under noisy conditions.

🚀 Future Work
Incorporate additional classes like Stridor, Rhonchi

Deploy as a mobile app and integrate with IoT stethoscopes

Explore real-time inference on edge devices

Enhance explainability using Grad-CAM or SHAP

⚙️ How to Run 🛠️
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/Oxysteth.git
cd Oxysteth
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare the dataset:

Place the audio files in the specified folder structure.

Train the model:

bash
Copy
Edit
python train.py
Predict using a sample audio file:

bash
Copy
Edit
python predict.py --audio your_audio_file.wav

👥 Team
👨‍💻 Kunal Nemade
👨‍💻 Sarvesh Kadam
👨‍💻 Yahya Moosa
👨‍💻 Ikjot Singh

🔗 Live Preview

📎 Open on Desktop Browser
🔗 Click here to view - https://drive.google.com/file/d/1FbiqbhGL4_17x7o1NGgn4knqxemzrd5u/view?usp=sharing&t=4 
