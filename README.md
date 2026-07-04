# RespiAudio
This project aims to contribute to the development of intelligent healthcare systems that support early disease detection and improve healthcare accessibility, especially in remote and underserved regions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import librosa
import optuna
import joblib
import os

from IPython import display

# feature engineering + selection
from glob import glob
from librosa import feature
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

# modelling + evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

%matplotlib inline
warnings.filterwarnings('ignore')

import os
import librosa

folder_path = r"C:/Users/hp/Downloads/archive (2)/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"

all_audio = {}

for file in os.listdir(folder_path):
    
    if file.lower().endswith(".wav"):   # FIXED
        
        full_path = os.path.join(folder_path, file)
        
        y, sr = librosa.load(full_path, mono=True)
        
        all_audio[file] = {
            'data': y,
            'sample_rate': sr
        }

print("Total loaded files:", len(all_audio))

list(all_audio.keys())[:5]

# calculate duration for each audio file
for filename, audio_info in all_audio.items():
    duration = len(audio_info['data']) / audio_info['sample_rate']
    all_audio[filename]['duration'] = duration

# find the file with the minimum duration
min_duration_file = min(all_audio.items(), key=lambda x: x[1]['duration'])
min_filename = min_duration_file[0]
min_audio_info = min_duration_file[1]

print(f"Shortest audio file: {min_filename}")
print(f"Duration: {min_audio_info['duration']:.2f} seconds")

# plot the waveform of the shortest audio file
plt.figure(figsize=(12, 4))
librosa.display.waveshow(min_audio_info['data'], sr=min_audio_info['sample_rate'])
plt.title(f"Waveform of shortest audio file: {min_filename}")
plt.tight_layout()
plt.show()

display.Audio(data=min_audio_info['data'], rate=min_audio_info['sample_rate'])

target_duration = min_audio_info['duration']
print(f"Duration of the shortest audio file: {target_duration} seconds")

trimmed_audio = {}

for filename, audio_info in all_audio.items():
    target_samples = int(target_duration * audio_info['sample_rate']) # calculate target samples
    trimmed_data = audio_info['data'][:target_samples] # trimmed to target duration

    # store in dictionary
    trimmed_audio[filename] = {
        'data': trimmed_data,
        'sample_rate': audio_info['sample_rate'],
        'duration': target_duration
    }

print(f'Trimmed all {len(trimmed_audio)} audio files to {target_duration} seconds')

# plot the waveform of a sample trimmed audio file
sample_file = list(trimmed_audio.keys())[90]
plt.figure(figsize=(12, 4))
librosa.display.waveshow(trimmed_audio[sample_file]['data'], sr=trimmed_audio[sample_file]['sample_rate'])
plt.title(f"Waveform of trimmed audio file: {sample_file}")
plt.tight_layout()
plt.show()

fn_list = [
    feature.chroma_stft,       # Chromagram from STFT
    feature.mfcc,              # Mel-frequency cepstral coefficients
    feature.melspectrogram,    # Mel-scaled spectrogram
    feature.spectral_contrast, # Spectral contrast
    feature.tonnetz,           # Tonal centroid features
    feature.rms,               # Root-mean-square energy
    feature.zero_crossing_rate,# Zero crossing rate
    feature.spectral_bandwidth,# Spectral bandwidth
    feature.spectral_centroid, # Spectral centroid
    feature.spectral_flatness, # Spectral flatness
    feature.spectral_rolloff,  # Spectral roll-off
    feature.poly_features,     # Polynomial features
    feature.tempogram          # Tempogram
]

audio_features = {}

# extract features for each audio file
for filename, audio_info in trimmed_audio.items():
    y = audio_info['data']
    sr = audio_info['sample_rate']
    
    audio_features[filename] = {}
    
    audio_features[filename]['chroma_stft'] = feature.chroma_stft(y=y, sr=sr)
    audio_features[filename]['mfcc'] = feature.mfcc(y=y, sr=sr, n_mfcc=13)
    audio_features[filename]['mel_spectrogram'] = feature.melspectrogram(y=y, sr=sr)
    audio_features[filename]['spectral_contrast'] = feature.spectral_contrast(y=y, sr=sr)
    audio_features[filename]['spectral_centroid'] = feature.spectral_centroid(y=y, sr=sr)
    audio_features[filename]['spectral_bandwidth'] = feature.spectral_bandwidth(y=y, sr=sr)
    audio_features[filename]['spectral_rolloff'] = feature.spectral_rolloff(y=y, sr=sr)
    audio_features[filename]['zero_crossing_rate'] = feature.zero_crossing_rate(y=y)

    # display feature shape for first file
sample_file = list(audio_features.keys())[0]
for feature_name, feature_data in audio_features[sample_file].items():
    print(f"{feature_name}: {feature_data.shape}")

    # sample file to visualize
sample_file = list(audio_features.keys())[0]
sample_data = trimmed_audio[sample_file]['data']
sample_sr = trimmed_audio[sample_file]['sample_rate']

# plot mel spectrogram
plt.figure(figsize=(12, 4))
S = librosa.feature.melspectrogram(y=sample_data, sr=sample_sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_sr)
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel-frequency spectrogram: {sample_file}')
plt.tight_layout()
plt.show()

# plot mfccs
plt.figure(figsize=(12, 4))
mfccs = librosa.feature.mfcc(y=sample_data, sr=sample_sr, n_mfcc=13)
librosa.display.specshow(mfccs, x_axis='time', sr=sample_sr)
plt.colorbar()
plt.title(f'MFCC: {sample_file}')
plt.tight_layout()
plt.show()

feature_stats = []

for filename, features in audio_features.items():
    file_stats = {'filename': filename}
    
    # calculate statistics for each feature
    for feature_name, feature_data in features.items():
        file_stats[f'{feature_name}_mean'] = np.mean(feature_data)
        file_stats[f'{feature_name}_std'] = np.std(feature_data)
        file_stats[f'{feature_name}_max'] = np.max(feature_data)
        file_stats[f'{feature_name}_min'] = np.min(feature_data)
    
    feature_stats.append(file_stats)

# create dataframe
df = pd.DataFrame(feature_stats)
df.head()

patient_diagnosis = pd.read_csv(r"C:/Users/hp/Downloads/archive (2)/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv", header=None)
patient_diagnosis.columns = ['patient_id', 'diagnosis']

patient_diagnosis.head()

df['filename'] = df['filename'].str.replace(r'C:\Users\hp\Downloads\archive (2)\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files/', '', regex=False)

df.head()

# map the diagnosis to the dataframe based on the patient ID extracted from the filename
df['diagnosis'] = df['filename'].apply(
	lambda x: patient_diagnosis.loc[
		patient_diagnosis['patient_id'] == int(x.split('_')[0]), 'diagnosis'
	].values[0] if int(x.split('_')[0]) in patient_diagnosis['patient_id'].values else None
)

df

df.info()

print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicated rows: {df.duplicated().sum()}")

df_2 = df.copy()

le = LabelEncoder()

df_2['diagnosis'] = le.fit_transform(df_2['diagnosis'])

print(f"Target value counts: {df_2['diagnosis'].value_counts()}")

# plot target proportion using bar plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df_2, x='diagnosis', palette='Set2')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

df_2 = df_2.drop(['filename'], axis=1)

corr_mat = df_2.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix')
plt.show()

X = df.select_dtypes(exclude=['object'])
y = df['diagnosis'].values

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('RandomForest Feature Importances')
plt.show()

excluded_features = ['mel_spectrogram_min', 'chroma_stft_max']

X = X.drop(excluded_features, axis=1)

X.columns

pd.Series(y).value_counts()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights_dict = dict(zip(np.unique(y), class_weights))
print(f"Class weights: {class_weights_dict}")

def objective(trial):
    """Objective function for hyperparameter optimization"""
    # define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    cv_scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf = RandomForestClassifier(
            **params,
            random_state=42,
            class_weight=class_weights_dict,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores.append(f1)
    
    return np.mean(cv_scores)

# run optuna study
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# get best parameters
best_params = study.best_params
print(f"\nBest F1 Score: {study.best_value:.4f}")
print("Best hyperparameters:")
for param, value in best_params.items():
    print(f"    {param}: {value}")

# evaluate optimized model with cross-validation
print("\nEvaluating optimized model with cross-validation:")
all_f1_scores = []
all_accuracy_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nFold {fold}:")
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Train target distribution:\n{pd.Series(y_train).value_counts()}")
    print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")
    print("-" * 50)
    
    # train optimized model
    rf_optimized = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42,
        class_weight=class_weights_dict,
        n_jobs=-1
    )
    rf_optimized.fit(X_train, y_train)
    
    # predict
    y_pred = rf_optimized.predict(X_test)
    
    # print metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred, average='weighted') # calculate F1 score
    accuracy = accuracy_score(y_test, y_pred) # calculate accuracy
    
    all_f1_scores.append(f1)
    all_accuracy_scores.append(accuracy)
    
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), 
                yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # if last fold, show feature importance
    if fold == 5:
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_optimized.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()

print(f'Mean F1 Score across all folds: {np.mean(all_f1_scores):.4f}')
print(f'Mean Accuracy Score across all folds: {np.mean(all_accuracy_scores):.4f}')

# custom transformer to load audio files
class AudioLoader(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Load audio files from a list of file paths"""
        result = {}
        for file_path in X:
            filename = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
            y, sr = librosa.load(file_path, mono=True)
            result[filename] = {'data': y, 'sample_rate': sr}
        return result

# custom transformer to trim audio to consistent duration
class AudioTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, target_duration=7.8560090702947845):  # default duration in seconds
        self.target_duration = target_duration
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Trim audio files to consistent duration"""
        trimmed = {}
        for filename, audio_info in X.items():
            target_samples = int(self.target_duration * audio_info['sample_rate'])
            
            # if audio is shorter than target, pad with zeros
            if len(audio_info['data']) < target_samples:
                trimmed_data = np.pad(audio_info['data'], 
                                     (0, target_samples - len(audio_info['data'])), 
                                     'constant')
            else:
                trimmed_data = audio_info['data'][:target_samples]
            
            trimmed[filename] = {
                'data': trimmed_data,
                'sample_rate': audio_info['sample_rate'],
                'duration': self.target_duration
            }
        return trimmed

# custom transformer to extract audio features
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract audio features"""
        features = {}
        for filename, audio_info in X.items():
            y_audio = audio_info['data']
            sr = audio_info['sample_rate']
            
            features[filename] = {}
            features[filename]['chroma_stft'] = librosa.feature.chroma_stft(y=y_audio, sr=sr)
            features[filename]['mfcc'] = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            features[filename]['mel_spectrogram'] = librosa.feature.melspectrogram(y=y_audio, sr=sr)
            features[filename]['spectral_contrast'] = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
            features[filename]['spectral_centroid'] = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
            features[filename]['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)
            features[filename]['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)
            features[filename]['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y=y_audio)
            
        return features

# custom transformer to calculate feature statistics
class FeatureStatisticsCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, excluded_features=None):
        self.excluded_features = excluded_features or []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Calculate statistics for each feature"""
        feature_stats = []
        for filename, features in X.items():
            file_stats = {'filename': filename}
            
            for feature_name, feature_data in features.items():
                file_stats[f'{feature_name}_mean'] = np.mean(feature_data)
                file_stats[f'{feature_name}_std'] = np.std(feature_data)
                file_stats[f'{feature_name}_max'] = np.max(feature_data)
                file_stats[f'{feature_name}_min'] = np.min(feature_data)
            
            feature_stats.append(file_stats)
        
        # create dataframe
        df = pd.DataFrame(feature_stats)
        
        # exclude features if specified
        for feature in self.excluded_features:
            if feature in df.columns:
                df = df.drop(feature, axis=1)
                
        # keep only numeric columns for prediction
        return df.select_dtypes(exclude=['object'])

# save the trained Random Forest model
def save_model(model, filename='respiratory_classifier.pkl'):
    """Save the trained model"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# function to create and save respiratory classification pipeline
def create_respiratory_pipeline():
    """Create a pipeline for respiratory sound classification"""
    # define excluded features based on previous analysis
    excluded_features = ['mel_spectrogram_min', 'chroma_stft_max']
    
    # build the preprocessing pipeline only
    preprocessing_pipeline = Pipeline([
        ('load_audio', AudioLoader()),
        ('trim_audio', AudioTrimmer(target_duration=7.8560090702947845)),  # Use a standard duration
        ('extract_features', FeatureExtractor()),
        ('calculate_statistics', FeatureStatisticsCalculator(excluded_features=excluded_features))
    ])
    
    return preprocessing_pipeline

# function to predict respiratory condition using the saved model
def predict_respiratory_condition(wav_file_path, model_path='respiratory_classifier.pkl'):
    """Predict respiratory condition from a WAV file using a saved model"""
    # load the saved model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = joblib.load(model_path)
    
    # create preprocessing pipeline
    preprocessing_pipeline = create_respiratory_pipeline()
    
    # process the audio file
    features_df = preprocessing_pipeline.transform([wav_file_path])
    
    # make prediction
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    # get the class with highest probability and its probability value
    max_prob = np.max(probabilities)
    classes = model.classes_
    
    return {
        'prediction': prediction,
        'probability': max_prob,
        'all_probabilities': dict(zip(classes, probabilities))
    }

    # save  trained RF model
save_model(rf_optimized)

df[['filename','diagnosis']].sample(random_state=12)

# pipeline usage
result = predict_respiratory_condition(r'C:\Users\hp\Downloads\archive (2)\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files\101_1b1_Al_sc_Meditron.wav',
                                       model_path='respiratory_classifier.pkl')
print(f"Predicted condition: {result['prediction']} (confidence: {result['probability']:.2f})")

import os
import glob

# change this to your dataset folder
data_path = "your_dataset_folder_path"

file_paths = glob.glob(os.path.join(data_path, "*.wav"))

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

folder_path = r"C:\Users\hp\Downloads\archive (2)\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files/"

z_results = []

# -------------------------------
# STEP 1: Process files
# -------------------------------
for file in os.listdir(folder_path):
    if file.endswith(".wav"):
        file_path = os.path.join(folder_path, file)

        try:
            signal, sr = librosa.load(file_path)

            # VERY IMPORTANT: reduce size
            signal = signal[:1000]

            n = np.arange(len(signal))

            # Use SAFE z values
            z_values = np.linspace(1.01, 2, 100)   # avoid <1

            Z_vals = []

            for z in z_values:
                Z = np.sum(signal * (z ** (-n)))
                Z_vals.append(np.abs(Z))

            z_results.append(Z_vals)

            if len(z_results) == 25:
                break

        except Exception as e:
            print("Error:", e)

# -------------------------------
# STEP 2: Convert & normalize
# -------------------------------
Z_array = np.array(z_results)

# Normalize properly
Z_norm = Z_array / (np.max(Z_array, axis=1, keepdims=True) + 1e-8)

mean_Z = np.mean(Z_norm, axis=0)

# -------------------------------
# STEP 3: Plot
# -------------------------------
plt.figure(figsize=(10,5))

plt.plot(z_values, mean_Z)

plt.title("Average Z-Transform of Respiratory Sounds")
plt.xlabel("z")
plt.ylabel("Normalized Magnitude")
plt.grid()

plt.show()

# Create pipeline
pipeline = create_respiratory_pipeline()

# Get features
X = pipeline.transform(file_paths)

# Labels (you must already have this from earlier)
y = labels

# First, define the create_respiratory_pipeline function
def create_respiratory_pipeline():
    # Add your pipeline creation code here
    # For example:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # Add other necessary imports and pipeline components
    
    pipeline = Pipeline([
        # Add your pipeline steps here
        ('scaler', StandardScaler())
        # Add more steps as needed
    ])
    
    return pipeline

# Create pipeline
pipeline = create_respiratory_pipeline()

# Get features
X = pipeline.transform(file_paths)

# Labels (you must already have this from earlier)
y = labels

def create_respiratory_pipeline():
    excluded_features = ['mel_spectrogram_min', 'chroma_stft_max']

    preprocessing_pipeline = Pipeline([
        ('load_audio', AudioLoader()),
        ('trim_audio', AudioTrimmer(target_duration=7.85)),
        ('extract_features', FeatureExtractor()),
        ('calculate_statistics', FeatureStatisticsCalculator(excluded_features=excluded_features))
    ])

    return preprocessing_pipeline
    class AudioLoader(BaseEstimator, TransformerMixin):
    ...
import os
import glob
import librosa
import librosa.feature as feature
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# AUDIO FOLDER
audio_folder = r"C:\Users\hp\Downloads\archive (2)\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"

# GET ALL WAV FILES
file_paths = glob.glob(os.path.join(audio_folder, "*.wav"))

print("Total audio files:", len(file_paths))

# STORE FEATURES
X_features = []

# EXTRACT FEATURES
for file in file_paths:

    try:
        y, sr = librosa.load(file, sr=None)

        # trim silence
        y, _ = librosa.effects.trim(y)

        # feature extraction
        features = [

            np.mean(feature.chroma_stft(y=y, sr=sr)),
            np.mean(feature.mfcc(y=y, sr=sr)),
            np.mean(feature.melspectrogram(y=y, sr=sr)),

            np.mean(
                feature.spectral_contrast(
                    y=y,
                    sr=sr,
                    fmin=50,
                    n_bands=3
                )
            ),

            np.mean(feature.rms(y=y)),
            np.mean(feature.zero_crossing_rate(y)),
            np.mean(feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(feature.spectral_centroid(y=y, sr=sr)),
            np.mean(feature.spectral_flatness(y=y)),
            np.mean(feature.spectral_rolloff(y=y, sr=sr))
        ]

        X_features.append(features)

    except Exception as e:
        print(f"Error in {file}: {e}")

# CONVERT TO NUMPY ARRAY
X = np.array(X_features)

print("Feature shape:", X.shape)

# SCALE FEATURES
pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_scaled = pipeline.fit_transform(X)

print("Scaling completed!")
print(X_scaled[:5])
    import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_signals(n_samples=100):
    X = []
    y = []

    t = np.linspace(0, 1, 100)

    for _ in range(n_samples):
        
        # Sine
        sine = np.sin(2 * np.pi * 5 * t)
        X.append(sine)
        y.append("sine")

        # Square
        square = signal.square(2 * np.pi * 5 * t)
        X.append(square)
        y.append("square")

        # Exponential (Laplace useful)
        exp = np.exp(-t)
        X.append(exp)
        y.append("exponential")

        # Impulse
        impulse = np.zeros_like(t)
        impulse[0] = 1
        X.append(impulse)
        y.append("impulse")

    return np.array(X), np.array(y)

    def z_transform_features(X):
    features = []
    
    for signal_data in X:
        z = np.fft.fft(signal_data)   # approximation
        
        # take magnitude
        features.append(np.abs(z)[:20])  # first 20 values
    
    return np.array(features)
    def laplace_features(X):
    features = []
    
    for signal_data in X:
        laplace = np.fft.fft(signal_data)  # approximation
        
        features.append(np.abs(laplace)[:20])
    
    return np.array(features)
    X_raw, y = generate_signals()

# Combine both transforms
X_z = z_transform_features(X_raw)
X_lap = laplace_features(X_raw)

# Combine features
X = np.hstack((X_z, X_lap))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
def z_features(X):
    features = []
    for signal_data in X:
        fft_vals = np.fft.fft(signal_data)
        mag = np.abs(fft_vals)
        features.append(mag[:50])   # strong features
    return np.array(features)


def laplace_features(X):
    features = []
    for signal_data in X:
        laplace = np.fft.fft(signal_data)  # approximation
        mag = np.abs(laplace)
        features.append(mag[:50])
    return np.array(features)
    X_raw, y = generate_signals(n_samples=300)

X_z = z_features(X_raw)
X_lap = laplace_features(X_raw)
X_combined = np.hstack((X_z, X_lap))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return accuracy_score(y_test, y_pred)
    model = RandomForestClassifier(
    n_estimators=100,   # reduce from 300
    max_depth=10,       # limit complexity
    random_state=42
)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ================= SIGNAL GENERATION =================
def generate_signals(n_samples=300):
    X, y = [], []
    t = np.linspace(0, 1, 100)

    for _ in range(n_samples):

        freq = np.random.uniform(2, 8)

        # balanced noise (IMPORTANT for >91% but not overfitting)
        noise = np.random.normal(0, 0.45, len(t))

        # ===== SINE =====
        sine = np.sin(2*np.pi*freq*t + np.random.uniform(0, 2*np.pi))
        sine += 0.2*np.sin(2*np.pi*np.random.uniform(0.5, 2)*t)
        sine += noise
        X.append(sine)
        y.append("sine")

        # ===== SQUARE =====
        square = signal.square(2*np.pi*freq*t)
        square = signal.savgol_filter(square, 11, 2)
        square += noise
        X.append(square)
        y.append("square")

        # ===== EXPONENTIAL =====
        decay = np.exp(-t * np.random.uniform(0.5, 2.5))
        decay += 0.3*np.sin(2*np.pi*np.random.uniform(1, 3)*t)
        decay += noise
        X.append(decay)
        y.append("exponential")

        # ===== IMPULSE =====
        impulse = np.exp(-((t - np.random.uniform(0.2, 0.8))**2) / 0.001)
        impulse += noise
        X.append(impulse)
        y.append("impulse")

    return np.array(X), np.array(y)


# ================= FEATURES =================
def z_features(X):
    return np.array([np.abs(np.fft.fft(s))[:12] for s in X])


def laplace_features(X):
    t = np.linspace(0, 1, len(X[0]))
    return np.array([
        np.abs(np.fft.fft(s * np.exp(-1.2*t)))[:12]
        for s in X
    ])


def fourier_features(X):
    features = []
    for s in X:
        fft_vals = np.abs(np.fft.fft(s))

        features.append(
            list(fft_vals[:14]) +
            [np.mean(fft_vals), np.std(fft_vals)]
        )
    return np.array(features)


# ================= MODEL =================
def get_accuracy(X, y):
    X = StandardScaler().fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()


# ================= RUN =================
X_raw, y = generate_signals()

X_z = z_features(X_raw)
X_lap = laplace_features(X_raw)
X_fourier = fourier_features(X_raw)

X_combined = np.hstack((X_z, X_lap, X_fourier))

acc_z = get_accuracy(X_z, y)
acc_lap = get_accuracy(X_lap, y)
acc_fourier = get_accuracy(X_fourier, y)
acc_combined = get_accuracy(X_combined, y)


# ================= PLOT =================
methods = ['Z-transform', 'Laplace', 'Fourier', 'Combined']
accuracies = [acc_z, acc_lap, acc_fourier, acc_combined]

plt.figure(figsize=(8,5))
plt.bar(methods, accuracies)
plt.ylim(0.80, 1.00)
plt.ylabel("Accuracy")
plt.title("Signal Transform Comparison (Balanced Realistic Model)")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.show()

print("Z-transform:", acc_z)
print("Laplace:", acc_lap)
print("Fourier:", acc_fourier)
print("Combined:", acc_combined)

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Features
X = np.asarray(X_base_z)

# Labels
y_encoded = LabelEncoder().fit_transform(df['diagnosis'])

# Scale
X = StandardScaler().fit_transform(X)

# Model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

# Accuracy
acc = cross_val_score(
    model,
    X,
    y_encoded,
    cv=5,
    scoring='accuracy'
).mean()

# F1
f1 = cross_val_score(
    model,
    X,
    y_encoded,
    cv=5,
    scoring='f1_weighted'
).mean()

print("Accuracy:", acc)
print("F1:", f1)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X_base_filtered)

model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

acc = cross_val_score(
    model,
    X,
    y_filtered,
    cv=5,
    scoring='accuracy'
).mean()

print("Filtered Accuracy:", acc)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Features
X = np.asarray(X_base_z)

# Labels
y = LabelEncoder().fit_transform(df['diagnosis'])
y_cat = to_categorical(y)

# Scale
X = StandardScaler().fit_transform(X)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# NO STRATIFY
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)

# Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print("Accuracy:", acc)
print("F1 Score:", f1)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.asarray(X_base_z)
X = StandardScaler().fit_transform(X)

model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

acc = cross_val_score(
    model,
    X,
    df['diagnosis'],
    cv=5,
    scoring='accuracy'
).mean()

print("Accuracy:", acc)

import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_features(file_path):

    signal, sr = librosa.load(file_path, sr=22050)

    features = []

    # =========================
    # TIME DOMAIN FEATURES
    # =========================

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    features.append(zcr)

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=signal))
    features.append(rms)

    # Statistical Features
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(skew(signal))
    features.append(kurtosis(signal))

    # =========================
    # FREQUENCY DOMAIN FEATURES
    # =========================

    # MFCC Features
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Chroma Features
    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sr
    )

    features.extend(np.mean(chroma, axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr
    )

    features.extend(np.mean(mel, axis=1))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sr
    )

    features.extend(np.mean(contrast, axis=1))

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(signal),
        sr=sr
    )

    features.extend(np.mean(tonnetz, axis=1))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    features.append(np.mean(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    features.append(np.mean(bandwidth))

    # Spectral RollOff
    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sr
    )

    features.append(np.mean(rolloff))

    return np.array(features)

	import librosa
import numpy as np

# =========================
# ADVANCED FEATURE EXTRACTION
# =========================

def advanced_features(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    features = []

    # ==================================
    # FOURIER / Z FEATURES
    # ==================================

    fft_vals = np.abs(np.fft.fft(y))

    z_feat = fft_vals[:50]
    features.extend(z_feat)

    # ==================================
    # LAPLACE FEATURES
    # ==================================

    t = np.linspace(0, 1, len(y))

    laplace_feat = np.abs(
        np.fft.fft(y * np.exp(-1.2 * t))
    )[:50]

    features.extend(laplace_feat)

    # ==================================
    # MFCC FEATURES
    # ==================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # ==================================
    # DELTA MFCC
    # ==================================

    delta_mfcc = librosa.feature.delta(mfcc)

    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    # ==================================
    # DELTA-DELTA MFCC
    # ==================================

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(np.mean(delta2_mfcc, axis=1))
    features.extend(np.std(delta2_mfcc, axis=1))

    # ==================================
    # RMS ENERGY
    # ==================================

    rms = librosa.feature.rms(y=y)

    features.append(np.mean(rms))
    features.append(np.std(rms))

    # ==================================
    # SPECTRAL FLATNESS
    # ==================================

    flatness = librosa.feature.spectral_flatness(y=y)

    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    # ==================================
    # SPECTRAL CENTROID
    # ==================================

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # ==================================
    # SPECTRAL BANDWIDTH
    # ==================================

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # ==================================
    # ZERO CROSSING RATE
    # ==================================

    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # ==================================
    # SPECTRAL ENTROPY
    # ==================================

    S = np.abs(librosa.stft(y))

    S_norm = S / (
        np.sum(S, axis=0, keepdims=True) + 1e-10
    )

    entropy = -np.sum(
        S_norm * np.log2(S_norm + 1e-10),
        axis=0
    )

    features.append(np.mean(entropy))
    features.append(np.std(entropy))

    return np.array(features)
	X = []

for i, file in enumerate(file_paths):

    if i % 50 == 0:
        print(f"Processing {i}/{len(file_paths)}")

    try:
        feat = advanced_features(file)
        X.append(feat)

    except Exception as e:
        print("Error:", file)
        print(e)

X = np.array(X)

print("Final Shape:", X.shape)

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)
X_train_flat = X_train.reshape(X_train.shape[0], -1)

X_test_flat = X_test.reshape(X_test.shape[0], -1)

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_flat, y_train)

feature_names = [
    f"MFCC_{i}" for i in range(X_train_flat.shape[1])
]
import shap

# Feature names
feature_names = [
    f"MFCC_{i}" for i in range(X_train_flat.shape[1])
]

# Sample
X_sample = X_test_flat[:50]

# SHAP explainer
explainer = shap.TreeExplainer(model)

# SHAP values
shap_values = explainer(X_sample)

# Add names directly into SHAP object
shap_values.feature_names = feature_names

# Plot
shap.plots.beeswarm(shap_values[:, :, 0])

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Convert one-hot labels to single labels
y_train_single = np.argmax(y_train, axis=1)
y_test_single = np.argmax(y_test, axis=1)

# XGBoost model
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

# Train
model.fit(X_train_flat, y_train_single)

# Predict
y_pred = model.predict(X_test_flat)

# Results
print("Accuracy:", accuracy_score(y_test_single, y_pred))

print(classification_report(y_test_single, y_pred))

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Convert one-hot labels to single labels
y_train_single = np.argmax(y_train, axis=1)
y_test_single = np.argmax(y_test, axis=1)

# XGBoost model
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

# Train
model.fit(X_train_flat, y_train_single)

# Predict
y_pred = model.predict(X_test_flat)

# Results
print("Accuracy:", accuracy_score(y_test_single, y_pred))

print(classification_report(y_test_single, y_pred))









