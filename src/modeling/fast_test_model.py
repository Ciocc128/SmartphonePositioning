import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib
import os
import copy
from tqdm import tqdm
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Caricamento dei dati
def load_data(file_path):
    """Carica i dati da un file CSV."""
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocessing
def preprocess_data(data):
    """Effettua il preprocessing dei dati."""
    # Separare caratteristiche, target e soggetti
    X = data.drop(columns=['Position', 'Subject'])
    y = data['Position']
    subjects = data['Subject']

    return X, y, subjects

# Step 3: Creazione dei modelli
def create_models(input_dim, num_classes):
    """Crea una lista di modelli da testare."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'XGBoost': XGBClassifier(
            tree_method='hist',  # Ottimizzato per CPU
            n_jobs=8,  # Usa 8 core per mantenere il sistema fluido
            max_depth=8,  # Profondità massima dell'albero
            colsample_bytree=0.8,  # Percentuale di feature per split
            subsample=0.8,  # Percentuale di dati usati per ogni albero
            n_estimators=150,  # Numero di alberi
            grow_policy='lossguide',  # Politica di crescita ottimizzata per grandi dataset
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42),
        'ANN': create_ann(input_dim, num_classes)
    }
    return models

def create_ann(input_dim, num_classes):
    """Crea un modello di rete neurale artificiale con TensorFlow e GPU."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def test_models(models, X_sample, y_sample, subjects_sample):
    print("Eseguendo test rapido su tutti i modelli...")
    for name, model in models.items():
        try:
            print(f"\nTestando il modello: {name}")
            if name == 'ANN':
                # Testa l'ANN separatamente con una divisione semplice
                train_idx = X_sample.index[:800]  # 80% training
                val_idx = X_sample.index[800:]   # 20% validation
                X_train, X_val = X_sample.loc[train_idx], X_sample.loc[val_idx]
                y_train, y_val = y_sample.loc[train_idx], y_sample.loc[val_idx]

                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=1,  # Solo 1 epoca per il test
                    batch_size=32,
                    verbose=0  # Riduci l'output
                )
            else:
                # Usa GroupKFold per gli altri modelli
                group_kfold = GroupKFold(n_splits=2)  # Dividi in 2 fold per test veloce
                for train_idx, val_idx in group_kfold.split(X_sample, y_sample, groups=subjects_sample):
                    X_train, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
                    y_train, y_val = y_sample.iloc[train_idx], y_sample.iloc[val_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    print(f"{name} - Accuracy: {model.score(X_val, y_val):.4f}")
                    break  # Esegui solo un fold per il test

            print(f"Modello {name} testato con successo!")
        except Exception as e:
            print(f"Errore durante il test del modello {name}: {e}")

if __name__ == "__main__":
    # Specifica il percorso del file CSV
    data_file = "/Volumes/Mac/DatasetSP/preprocessed/feature_selected/dataset_with_pcs_train.csv"

    print("Caricamento del dataset...")
    data = load_data(data_file)
    print("Dataset caricato con successo!")

    print("Preprocessing dei dati...")
    X, y, subjects = preprocess_data(data)
    print("Preprocessing completato!")

    print("Creazione dei modelli...")
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    models = create_models(input_dim, num_classes)
    print("Modelli creati!")

    # Crea un sottoinsieme per il test rapido
    X_sample = X.sample(n=1000, random_state=42)
    y_sample = y.loc[X_sample.index]
    subjects_sample = subjects.loc[X_sample.index]

    # Esegui il test rapido
    test_models(models, X_sample, y_sample, subjects_sample)
