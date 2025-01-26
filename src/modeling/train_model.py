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

# Step 4: Training e validazione basata sui soggetti
def train_and_evaluate_models(models, X, y, subjects, save_dir, log_dir):
    """Allena e valuta diversi modelli con cross-validation basata sui soggetti, salva i pesi e le metriche."""
    # Assicurati che le directory di salvataggio esistano
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    group_kfold = GroupKFold(n_splits=5)
    total_models = len(models)
    with tqdm(total=total_models, desc="Training modelli") as pbar:
        for idx, (name, model) in enumerate(models.items(), start=1):
            print(f"\n[{idx}/{total_models}] Inizio allenamento modello: {name}")
            start_time = time.time()
            cv_scores = []

            if name == 'ANN':
                # Specifico per ANN: dividere in training e validation set per soggetti
                for train_idx, val_idx in group_kfold.split(X, y, groups=subjects):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=256,
                        callbacks=[early_stopping],
                        verbose=1
                    )

                    # Salva la storia dell'allenamento
                    log_path = os.path.join(log_dir, f"{name.replace(' ', '_').lower()}_training_history.log")
                    with open(log_path, "w") as log_file:
                        for epoch, (train_loss, val_loss, val_acc) in enumerate(zip(
                            history.history['loss'],
                            history.history['val_loss'],
                            history.history.get('val_accuracy', [])
                        )):
                            log_file.write(
                                f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n"
                            )

                    # Valutare il modello sul validation set
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
                    cv_scores.append(score)

                    # Calcola e stampa le metriche finali
                    y_pred = model.predict(X_val)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    report = classification_report(y_val, y_pred_classes)
                    matrix = confusion_matrix(y_val, y_pred_classes)
                    print(f"Classification Report:\n{report}")
                    print(f"Confusion Matrix:\n{matrix}")

                    # Salva nel log file
                    final_log_path = os.path.join(log_dir, f"{name.replace(' ', '_').lower()}_final_metrics.log")
                    with open(final_log_path, "w") as log_file:
                        log_file.write(f"Classification Report:\n{report}\n")
                        log_file.write(f"Confusion Matrix:\n{matrix}\n")

                # Salva i pesi dell'ANN
                ann_model_filename = os.path.join(save_dir, f"baseline_{name.lower()}.h5")
                model.save(ann_model_filename)
                print(f"[{name}] Modello salvato in {ann_model_filename}")

            else:
                # Cross-validation per gli altri modelli
                for train_idx, test_idx in group_kfold.split(X, y, groups=subjects):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = model.score(X_test, y_test)
                    cv_scores.append(score)

                    # Calcola e stampa le metriche
                    report = classification_report(y_test, y_pred)
                    matrix = confusion_matrix(y_test, y_pred)
                    print(f"Classification Report:\n{report}")
                    print(f"Confusion Matrix:\n{matrix}")

                    # Salva nel log file
                    log_path = os.path.join(log_dir, f"{name.replace(' ', '_').lower()}_metrics.log")
                    with open(log_path, "a") as log_file:
                        log_file.write(f"Classification Report:\n{report}\n")
                        log_file.write(f"Confusion Matrix:\n{matrix}\n")

            print(f"[{name}] Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

            # Salva il modello allenato sull'ultimo fold
            if name != 'ANN':
                model_filename = os.path.join(save_dir, f"baseline_{name.replace(' ', '_').lower()}.joblib")
                joblib.dump(model, model_filename)
                print(f"[{name}] Modello salvato in {model_filename}")

            elapsed_time = time.time() - start_time
            print(f"[{name}] Modello completato in {elapsed_time:.2f} secondi.")

            # Avanza la barra di progresso
            pbar.update(1)

# Main
if __name__ == "__main__":
    # Specifica il percorso del file CSV
    data_file = "/Volumes/Mac/DatasetSP/preprocessed/feature_selected/dataset_with_pcs_train.csv"
    save_dir = "models/baseline"  # Percorso per salvare i modelli
    log_dir = "logs/baseline"  # Percorso per salvare i log

    # Caricamento e preprocessing dei dati
    print("Caricamento del dataset...")
    data = load_data(data_file)
    print("Dataset caricato con successo!")

    print("Preprocessing dei dati...")
    X, y, subjects = preprocess_data(data)
    print("Preprocessing completato!")

    # Creazione dei modelli
    print("Creazione dei modelli...")
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    models = create_models(input_dim, num_classes)
    print("Modelli creati!")

    # Training e validazione
    print("Inizio del training e della validazione...")
    train_and_evaluate_models(models, X, y, subjects, save_dir, log_dir)
    print("Training e validazione completati!")
