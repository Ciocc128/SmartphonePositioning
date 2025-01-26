import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tsfresh import select_features
from tsfresh import extract_relevant_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
import os

if __name__ == "__main__":

    # ================================
    # 1. Caricamento del dataset
    # ================================
    print("Caricamento del dataset noredundant_features...")
    dataset_path = "/Volumes/Mac/DatasetSP/preprocessed/data_cleaning/noredundant_features.csv"
    features_df = pd.read_csv(dataset_path)
    print(f"Dataset caricato con {features_df.shape[0]} righe e {features_df.shape[1]} colonne.")

    # ================================
    # 2. Split dei soggetti in training e test
    # ================================

    # Conta il numero di Bout per ogni posizione per soggetto
    subject_position_counts = features_df.groupby(["Subject", "Position"])['Bout'].nunique().unstack(fill_value=0)
    print(subject_position_counts.head())

    # Bilanciamento: seleziona 3 soggetti per il test set
    train_subjects = []
    remaining_subjects = set(subject_position_counts.index)

    while len(train_subjects) < len(subject_position_counts) - 3:
        # Calcola il bilanciamento attuale delle posizioni nel training set
        current_balance = subject_position_counts.loc[train_subjects].sum()

        # Seleziona il prossimo soggetto che migliora il bilanciamento
        best_subject = None
        best_balance = None

        for subject in remaining_subjects:
            candidate_balance = current_balance + subject_position_counts.loc[subject]

            # Miglior bilanciamento: minimizza lo sbilanciamento massimo tra le posizioni
            if best_balance is None or candidate_balance.max() - candidate_balance.min() < best_balance.max() - best_balance.min():
                best_subject = subject
                best_balance = candidate_balance

        # Aggiorna i soggetti selezionati
        train_subjects.append(best_subject)
        remaining_subjects.remove(best_subject)

    # Soggetti rimanenti nel test set
    test_subjects = list(remaining_subjects)

    print(f"Soggetti nel training set: {train_subjects}")
    print(f"Soggetti nel test set: {test_subjects}")

    # ================================
    # 3. Divisione del dataset
    # ================================

    # Filtra i dati per training e test
    train_df = features_df[features_df['Subject'].isin(train_subjects)]
    test_df = features_df[features_df['Subject'].isin(test_subjects)]

    print(f"Training set: {train_df.shape[0]} righe")
    print(f"Test set: {test_df.shape[0]} righe")

    # Separa le feature e il target
    X_train = train_df.drop(columns=["Subject", "Test", "Trial", "Bout", "Position", "Unnamed: 0"])
    y_train = train_df["Position"]
    X_test = test_df.drop(columns=["Subject", "Test", "Trial", "Bout", "Position", "Unnamed: 0"])
    y_test = test_df["Position"]

    # Supponiamo di avere X (feature) e y (target)
    # X: DataFrame con le feature
    # y: Serie con il target

    # ==========================
    # Funzione per calcolare distribuzione con percentuali
    # ==========================
    def calculate_distribution_with_percentage(counts):
        total = counts.sum()
        percentages = (counts / total * 100).round(2)
        return pd.DataFrame({"Count": counts, "Percentage": percentages})

    # Calcola distribuzioni per training e test
    train_distribution = calculate_distribution_with_percentage(subject_position_counts.loc[train_subjects].sum())
    test_distribution = calculate_distribution_with_percentage(subject_position_counts.loc[test_subjects].sum())

    # ============================
    # 1. Selezione con tsfresh
    # ============================
    print("Selezione con tsfresh...")
    tsfresh_selected_features = select_features(X_train, y_train, multiclass=True, fdr_level=0.05).columns.tolist()
    print(f"Feature selezionate con tsfresh: {len(tsfresh_selected_features)}")

    # ============================
    # 2. Salvataggio delle feature selezionate
    # ============================

    # Salva le feature selezionate in una lista
    output_path = "/Volumes/Mac/DatasetSP/preprocessed/feature_selected/tsfresh_selected_features.csv"
    pd.Series(tsfresh_selected_features).to_csv(output_path, index=False)
    print(f"Feature selezionate salvate in '{output_path}'")

    # Salva il dataset di training e di test 
    train_df.to_csv("/Volumes/Mac/DatasetSP/preprocessed/train_df.csv", index=False)
    test_df.to_csv("/Volumes/Mac/DatasetSP/preprocessed/test_df.csv", index=False)

    # Salva in un log file i soggetti selezionati per il training e il test set e le position counts
    with open("/Volumes/Mac/DatasetSP/preprocessed/train_test_split_log.txt", "w") as f:
        f.write(f"Soggetti nel training set: {train_subjects}\n")
        f.write(f"Soggetti nel test set: {test_subjects}\n")
        f.write("\nDistribuzione dei Bout per posizione (training set):\n")
        f.write(train_distribution.to_string())
        f.write("\n\nDistribuzione dei Bout per posizione (test set):\n")
        f.write(test_distribution.to_string())

    print("Log file salvato in 'train_test_split_log.txt'")


    

    """# ============================
    # 2. Importanza delle feature con Random Forest
    # ============================
    print("Selezione con Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Ordina le feature in base all'importanza
    rf_feature_importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Mantieni le prime 50 feature
    rf_selected_features = rf_feature_importances.head(50)["Feature"].tolist()
    print(f"Feature selezionate con Random Forest: {len(rf_selected_features)}")

    # ============================
    # 3. Selezione basata su Mutua Informazione
    # ============================
    print("Selezione con Mutua Informazione...")
    mutual_info = mutual_info_classif(X_train, y_train, random_state=42)
    mutual_info_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Mutual_Info": mutual_info
    }).sort_values(by="Mutual_Info", ascending=False)

    # Mantieni le prime 50 feature (o imposta un'altra soglia)
    mutual_info_selected_features = mutual_info_df.head(50)["Feature"].tolist()
    print(f"Feature selezionate con Mutua Informazione: {len(mutual_info_selected_features)}")

    # ============================
    # 4. Combina i risultati
    # ============================
    # Unione dei risultati
    combined_features = set(tsfresh_selected_features) | set(rf_selected_features) | set(mutual_info_selected_features)
    print(f"Numero totale di feature selezionate dopo l'unione: {len(combined_features)}")

    # Creazione del nuovo DataFrame con le feature selezionate
    X_combined = X_train[list(combined_features)]

    # ============================
    # 5. Riduzione con metodo finale (opzionale)
    # ============================
    # Usa Lasso per una riduzione finale
    print("Riduzione finale con Lasso...")
    from sklearn.linear_model import LogisticRegression
    lasso = LogisticRegression(C=0.01, penalty="l1", solver="liblinear", random_state=42)
    lasso.fit(X_combined, y_train)

    # Seleziona le feature importanti
    model = SelectFromModel(lasso, prefit=True)
    X_final = model.transform(X_combined)
    final_selected_features = X_combined.columns[model.get_support()].tolist()
    print(f"Feature finali selezionate dopo Lasso: {len(final_selected_features)}")
    print(f"Feature selezionate finali: {final_selected_features}")

    # ============================
    # 6. Salva il risultato finale
    # ============================
    # Applica la stessa trasformazione ai dati di test
    X_test_final = X_test[final_selected_features]

    # Salva i risultati
    X_final_df = pd.DataFrame(X_final, columns=final_selected_features)
    X_final_df.to_csv("selected_features.csv", index=False)
    print("Feature selezionate salvate in 'selected_features.csv'")"""

