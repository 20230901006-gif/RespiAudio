# 🫁 Smart Auscultation: Enhancing Respiratory Disease Diagnosis Using Audio Analysis and AI

## Overview

Smart Auscultation is an AI-powered respiratory disease diagnosis system that analyzes lung sound recordings to assist in the detection of respiratory disorders. The project combines digital signal processing, machine learning, and explainable artificial intelligence (XAI) to provide accurate and interpretable predictions.

Respiratory audio recordings are preprocessed, relevant acoustic features are extracted, and machine learning models are trained to classify respiratory conditions. To improve transparency and trust in AI-assisted diagnosis, SHAP (SHapley Additive exPlanations) is used to explain model predictions.

This project demonstrates how artificial intelligence can support healthcare professionals by enabling efficient, non-invasive, and accessible respiratory disease screening.

## Features

- Respiratory audio preprocessing
- Noise reduction and signal normalization
- Acoustic feature extraction using Librosa
- Machine learning-based respiratory disease classification
- Explainable AI using SHAP
- Interactive prediction interface
- Probability-based disease prediction
- Visualization of model performance

## Dataset

This project uses the **ICBHI 2017 Respiratory Sound Database**, which contains lung sound recordings collected from patients with various respiratory conditions.

The dataset includes recordings from multiple respiratory disease categories, including healthy individuals and patients with respiratory disorders.

Due to licensing restrictions, the dataset is not included in this repository. Please obtain the dataset from the original source before running the notebooks.

## Methodology
## Methodology

The Smart Auscultation pipeline follows a systematic workflow for respiratory disease classification using respiratory sound recordings and machine learning.

### 1. Data Collection
- Respiratory sound recordings are obtained from the ICBHI 2017 Respiratory Sound Database.
- Audio recordings are labeled according to respiratory disease categories.

### 2. Audio Preprocessing
The audio signals undergo preprocessing to improve signal quality:
- Noise reduction
- Signal normalization
- Resampling
- Silence removal (if applicable)

### 3. Feature Extraction
Acoustic features are extracted using the Librosa library, including:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Chroma STFT
- Mel Spectrogram
- Spectral Contrast
- Spectral Centroid
- Spectral Bandwidth
- Spectral Roll-off
- Zero Crossing Rate
- Root Mean Square (RMS) Energy
- Tonnetz Features
- Spectral Flatness
- Tempogram (if applicable)

### 4. Machine Learning
Extracted features are used to train and evaluate machine learning models for respiratory disease classification.

The evaluated models include:
- Random Forest
- LightGBM

### 5. Explainable AI
SHAP (SHapley Additive exPlanations) is used to interpret model predictions by identifying the contribution of individual acoustic features.

### 6. Prediction
The trained model predicts respiratory disease classes from unseen respiratory sound recordings and provides probability scores for each prediction.

## Results

The Smart Auscultation pipeline generates several outputs for evaluating model performance and respiratory disease classification.

### Generated Outputs

- Disease prediction
- Prediction probabilities
- Confusion Matrix
- ROC Curve
- Feature Importance
- SHAP Summary Plot
- SHAP Force Plot
- Model evaluation metrics

The Explainable AI module improves transparency by highlighting the acoustic features that contribute most to each prediction.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RespiAudio.git
cd RespiAudio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the notebooks

Execute the notebooks in sequence to reproduce the complete respiratory disease diagnosis pipeline.
```

## Repository Structure

```
RespiAudio/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
├── models/
├── figures/
├── audio/
├── app/
└── data/
```

## Future Work

- Develop deep learning models using CNNs and Transformer-based architectures.
- Integrate with digital stethoscopes for real-time respiratory sound analysis.
- Expand the system to classify additional respiratory diseases.
- Develop a web-based or mobile application for clinical use.
- Validate the system using larger and more diverse clinical datasets.

## Citation

If you use this repository in your research or academic work, please cite this repository or contact the repository author for citation information.

## License

This project is licensed under the MIT License.
