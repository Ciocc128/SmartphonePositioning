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
import pickle
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

# Step 4: Gestione dei fold di GroupKFold
def save_folds(X, y, subjects, n_splits=5, save_path="modeling/fold_indices.pkl"):
    """Genera e salva gli indici di GroupKFold per garantire suddivisioni consistenti tra esperimenti."""
    group_kfold = GroupKFold(n_splits=n_splits)
    folds = list(group_kfold.split(X, y, groups=subjects))
    with open(save_path, "wb") as f:
        pickle.dump(folds, f)
    print(f"Folds salvati in {save_path}")

def load_folds(load_path="modeling/fold_indices.pkl"):
    """Carica gli indici salvati di GroupKFold."""
    try:
        with open(load_path, "rb") as f:
            folds = pickle.load(f)
        print(f"Folds caricati da {load_path}")
        return folds
    except FileNotFoundError:
        print(f"Errore: Il file {load_path} non esiste.")
        return None

def manage_folds(X, y, subjects, n_splits=5, save_path="modeling/fold_indices.pkl"):
    """Gestisce il caricamento o la generazione dei fold."""
    folds = load_folds(load_path=save_path)
    if folds is None:
        save_folds(X, y, subjects, n_splits=n_splits, save_path=save_path)
        folds = load_folds(load_path=save_path)
    return folds

# Step 5: Training e validazione basata sui soggetti
def train_and_evaluate_single_model(model_name, models, X, y, subjects, logs_dir, models_dir, pipeline_name, experiment_name, folds, use_early_stopping=True):
    """Allena e valuta un singolo modello con cross-validation basata sui soggetti, salva i pesi e le metriche."""
    # Creazione delle directory pipeline e esperimento
    pipeline_logs_dir = os.path.join(logs_dir, pipeline_name, experiment_name)
    pipeline_models_dir = os.path.join(models_dir, pipeline_name, experiment_name)
    os.makedirs(pipeline_logs_dir, exist_ok=True)
    os.makedirs(pipeline_models_dir, exist_ok=True)

    # Salva iperparametri nel file dell'esperimento
    hyperparams_path = os.path.join(pipeline_logs_dir, "hyperparameters.txt")
    with open(hyperparams_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        if model_name == 'ANN':
            f.write("ANN Hyperparameters:\n")
            f.write(" - Layers: [128, 64, num_classes]\n")
            f.write(" - Activation: relu\n")
            f.write(" - Optimizer: adam\n")
            f.write(" - Loss: sparse_categorical_crossentropy\n")
            f.write(" - Batch size: 256\n")
            f.write(" - Early stopping: {use_early_stopping}\n")
        else:
            f.write(f"Hyperparameters: {models[model_name].get_params()}\n")

    print(f"\nInizio allenamento modello: {model_name}")
    start_time = time.time()
    cv_scores = []
    all_reports = []

    if model_name == 'ANN':
        # Specifico per ANN: dividere in training e validation set per soggetti
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Cross-validation folds")):
            # Ricrea il modello per ogni fold
            model = create_ann(X.shape[1], len(np.unique(y)))

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            callbacks = []
            if use_early_stopping:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=256,
                callbacks=callbacks,
                verbose=1
            )

            # Salva la storia dell'allenamento
            log_path = os.path.join(pipeline_logs_dir, f"{model_name.replace(' ', '_').lower()}_training_history_fold_{fold_idx}.log")
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

            # Soggetti per il fold corrente
            train_subjects = subjects.iloc[train_idx].unique()
            val_subjects = subjects.iloc[val_idx].unique()

            # Aggiungi al report complessivo
            all_reports.append(f"Fold {fold_idx + 1} Train Subjects:\n{train_subjects}\n")
            all_reports.append(f"Fold {fold_idx + 1} Validation Subjects:\n{val_subjects}\n")
            all_reports.append(f"Fold {fold_idx + 1} Classification Report:\n{report}\n")
            all_reports.append(f"Fold {fold_idx + 1} Confusion Matrix:\n{matrix}\n")

        # Salva il report complessivo in un unico file
        combined_report_path = os.path.join(pipeline_logs_dir, "final_metrics.log")
        with open(combined_report_path, "w") as report_file:
            report_file.write("\n".join(all_reports))

        # Stampa e salva la Cross-Validation Accuracy
        print(f"[{model_name}] Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        with open(combined_report_path, "a") as report_file:
            report_file.write(f"\n[{model_name}] Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")

        # Salva i pesi dell'ANN
        ann_model_filename = os.path.join(pipeline_models_dir, f"baseline2_{model_name.lower()}.h5")
        model.save(ann_model_filename)
        print(f"[{model_name}] Modello salvato in {ann_model_filename}")

    else:
        # Cross-validation per gli altri modelli
        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(folds, desc="Cross-validation folds")):
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

            # Soggetti per il fold corrente
            train_subjects = subjects.iloc[train_idx].unique()
            val_subjects = subjects.iloc[val_idx].unique()

            # Aggiungi al report complessivo
            all_reports.append(f"Fold {fold_idx + 1} Train Subjects:\n{train_subjects}\n")
            all_reports.append(f"Fold {fold_idx + 1} Validation Subjects:\n{val_subjects}\n")
            all_reports.append(f"Fold {fold_idx + 1} Classification Report:\n{report}\n")
            all_reports.append(f"Fold {fold_idx + 1} Confusion Matrix:\n{matrix}\n")

        # Salva il report complessivo in un unico file
        combined_report_path = os.path.join(pipeline_logs_dir, "final_metrics.log")
        with open(combined_report_path, "w") as report_file:
            report_file.write("\n".join(all_reports))

        # Stampa e salva la Cross-Validation Accuracy
        print(f"[{model_name}] Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        with open(combined_report_path, "a") as report_file:
            report_file.write(f"\n[{model_name}] Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")

        # Salva il modello allenato sull'ultimo fold
        model_filename = os.path.join(pipeline_models_dir, f"baseline2_{model_name.replace(' ', '_').lower()}.joblib")
        joblib.dump(model, model_filename)
        print(f"[{model_name}] Modello salvato in {model_filename}")

    elapsed_time = time.time() - start_time
    print(f"[{model_name}] Modello completato in {elapsed_time:.2f} secondi.")

# Main
if __name__ == "__main__":
    # Specifica il percorso del file CSV
    data_file = "/Volumes/Mac/DatasetSP/feature engineering/feature_selected/dataset_with_pcs_train.csv"
    modeling_dir = "/Users/giorgio/Desktop/SmartphonePositioning/src/modeling"  # Cartella madre per il modeling
    logs_dir = os.path.join(modeling_dir, "logs")  # Cartella madre per i log
    models_dir = os.path.join(modeling_dir, "models" ) # Cartella madre per i modelli

    pipeline_name = "baseline2"  # Nome della pipeline
    experiment_name = "ann_no_ES"  # Nome dell'esperimento

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

    # Gestione dei fold
    fold_path = os.path.join(modeling_dir, "fold_indices.pkl")
    folds = manage_folds(X, y, subjects, save_path=fold_path)

    # Selezione del modello da allenare
    selected_model = "ANN"  # Modifica questo valore per selezionare un modello specifico o "ALL" per allenarli tutti

    use_early_stopping = False  # Modifica questo valore per abilitare/disabilitare l'early stopping nell'ANN

    if selected_model == "ALL":
        for model_name in models.keys():
            print(f"Inizio del training e della validazione per il modello: {model_name}...")
            train_and_evaluate_single_model(model_name, models, X, y, subjects, logs_dir, models_dir, pipeline_name, experiment_name, folds, use_early_stopping)
            print(f"Training e validazione completati per il modello: {model_name}!")
    else:
        print(f"Inizio del training e della validazione per il modello selezionato: {selected_model}...")
        train_and_evaluate_single_model(selected_model, models, X, y, subjects, logs_dir, models_dir, pipeline_name, experiment_name, folds, use_early_stopping)
        print(f"Training e validazione completati per il modello: {selected_model}!")
