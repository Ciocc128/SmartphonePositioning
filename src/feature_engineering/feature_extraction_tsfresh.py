# Estrazione del dataset con 2 righe per segmento temporale 1 per accelerometro e 1 per giroscopio
"""
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tqdm import tqdm
import os

if __name__ == "__main__":

    # Caricamento del dataset
    print("Caricamento del dataset...")
    output_path = "/Volumes/Mac/DatasetSP/preprocessed"
    partial_output_path = os.path.join(output_path, "partial_batches")
    os.makedirs(partial_output_path, exist_ok=True)

    df = pd.read_csv("/Volumes/Mac/DatasetSP/preprocessed/combined_dataset.csv")
    print(f"Dataset caricato con {len(df)} campioni.")

    # 1. Riordina il dataset
    print("Riordino del dataset...")
    df = df.sort_values(by=["Subject", "Test", "Trial", "Bout", "Position", "Sensor"]).reset_index(drop=True)
    print("Dataset riordinato.")

    # 2. Segmenta il dataset bout per bout
    grouped = df.groupby(["Subject", "Test", "Trial", "Bout", "Position"])

    # Funzione per segmentare un gruppo in finestre
    def segment_sensors(group, window_size=100):
        """
"""
        Segmenta un gruppo (bout) in finestre separate per accelerometro e giroscopio,
        allineando temporalmente i dati.
        """
"""
        # Separiamo accelerometro e giroscopio
        acc_data = group[group["Sensor"] == 1]
        gyr_data = group[group["Sensor"] == 2]
        
        num_acc_samples = len(acc_data)
        num_gyr_samples = len(gyr_data)
        
        # Verifica che i dati siano consistenti
        if num_acc_samples != num_gyr_samples:
            raise ValueError("Numero di campioni non corrispondenti tra accelerometro e giroscopio!")

        # Segmentazione per accelerometro
        acc_segments = [
            acc_data.iloc[i:i+window_size]
            for i in range(0, num_acc_samples - window_size + 1, window_size)
        ]
        
        # Segmentazione per giroscopio
        gyr_segments = [
            gyr_data.iloc[i:i+window_size]
            for i in range(0, num_gyr_samples - window_size + 1, window_size)
        ]
        
        return acc_segments, gyr_segments

    # Lista per salvare tutte le finestre e i metadati
    all_segments = []
    metadata = []

    # Itera su ogni bout
    print("Segmentazione dei gruppi...")
    for (subject, test, trial, bout, position), group in tqdm(grouped, desc="Processando gruppi"):
        # Segmenta accelerometro e giroscopio
        acc_segments, gyr_segments = segment_sensors(group)
        
        # Combina le finestre
        for acc_segment, gyr_segment in zip(acc_segments, gyr_segments):
            all_segments.append({"Acc": acc_segment, "Gyr": gyr_segment})
            metadata.append({
                "Subject": subject,
                "Test": test,
                "Trial": trial,
                "Bout": bout,
                "Position": position
            })

    print(f"Segmentazione completata. Numero totale di segmenti: {len(all_segments)}")

    # Calcola la norma per ciascun campione all'interno delle finestre
    print("Calcolo della norma per ciascun segmento...")
    for segment_pair in tqdm(all_segments, desc="Calcolando la norma"):
        segment_pair["Acc"]["Norm"] = np.sqrt(segment_pair["Acc"]["X"]**2 + segment_pair["Acc"]["Y"]**2 + segment_pair["Acc"]["Z"]**2)
        segment_pair["Gyr"]["Norm"] = np.sqrt(segment_pair["Gyr"]["X"]**2 + segment_pair["Gyr"]["Y"]**2 + segment_pair["Gyr"]["Z"]**2)

    # Estrai le feature con TSFresh in batch
    print("Estrazione delle feature con TSFresh...")
    batch_size = 1000
    combined_features = []
    num_batches = (len(all_segments) // batch_size) + 1

    for batch_idx in tqdm(range(num_batches), desc="Elaborazione batch"):  # Aggiunta di tqdm per i batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_segments))

        # Prepara i dati per il batch
        batch_segments = all_segments[start_idx:end_idx]
        batch_metadata = metadata[start_idx:end_idx]

        ts_data = []
        for idx, segment_pair in enumerate(batch_segments):
            acc_data = segment_pair["Acc"].copy()
            acc_data["id"] = f"{start_idx + idx}_acc"
            acc_data = acc_data.rename(columns={"Norm": "value"})

            gyr_data = segment_pair["Gyr"].copy()
            gyr_data["id"] = f"{start_idx + idx}_gyr"
            gyr_data = gyr_data.rename(columns={"Norm": "value"})

            ts_data.append(acc_data["id value".split()])
            ts_data.append(gyr_data["id value".split()])

        ts_data = pd.concat(ts_data, ignore_index=True)

        # Estrai le feature per il batch
        features = extract_features(ts_data, column_id="id", column_value="value")

        # Salva i risultati parziali su disco
        batch_output_file = os.path.join(partial_output_path, f"batch_{batch_idx + 1}.csv")
        features.to_csv(batch_output_file, index=True)

    print("Feature extraction completata.")

    # Unione dei risultati parziali
    print("Unione dei file parziali...")
    partial_files = [os.path.join(partial_output_path, f"batch_{i + 1}.csv") for i in range(num_batches)]
    final_features = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)

    # Crea il DataFrame finale
    print("Creazione del DataFrame finale...")
    meta_df = pd.DataFrame(metadata)
    final_df = pd.merge(meta_df, final_features, left_index=True, right_index=True)

    # Esporta il dataset finale
    print("Esportazione del dataset finale...")
    output_file_path = os.path.join(output_path, "combined_features_1.csv")
    final_df.to_csv(output_file_path, index=False)
    print("Esportazione completata. Dataset salvato come 'combined_features_1.csv'.")



# Estrazione del dataset con 1 riga per segmento temporale e per colonne sia features di accelerometro che di giroscopio
"""
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tqdm import tqdm
import os

if __name__ == "__main__":

    # Caricamento del dataset
    print("Caricamento del dataset...")
    output_path = "/Volumes/Mac/DatasetSP/preprocessed"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv("/Volumes/Mac/DatasetSP/preprocessed/combined_dataset.csv")
    print(f"Dataset caricato con {len(df)} campioni.")

    # 1. Riordina il dataset
    print("Riordino del dataset...")
    df = df.sort_values(by=["Subject", "Test", "Trial", "Bout", "Position", "Sensor"]).reset_index(drop=True)
    print("Dataset riordinato.")

    # Funzione per segmentare un gruppo in finestre
    def segment_bout(group, window_size=100):
        """
        Segmenta un gruppo (bout) in finestre temporali per accelerometro e giroscopio,
        mantenendo la coerenza temporale.
        """
        acc_data = group[group["Sensor"] == 1].copy()
        gyr_data = group[group["Sensor"] == 2].copy()
        
        # Segmentazione per accelerometro
        acc_segments = [
            acc_data.iloc[i:i+window_size]
            for i in range(0, len(acc_data) - window_size + 1, window_size)
        ]
        
        # Segmentazione per giroscopio
        gyr_segments = [
            gyr_data.iloc[i:i+window_size]
            for i in range(0, len(gyr_data) - window_size + 1, window_size)
        ]
        
        return acc_segments, gyr_segments

    # Raggruppa i dati per bout
    grouped = df.groupby(["Subject", "Test", "Trial", "Bout", "Position"])

    # Creazione del dataset finale
    all_segments = []
    metadata = []

    print("Segmentazione e batching dei dati...")
    for (subject, test, trial, bout, position), group in tqdm(grouped, desc="Processando gruppi"):
        acc_segments, gyr_segments = segment_bout(group)
        
        for i, (acc_segment, gyr_segment) in enumerate(zip(acc_segments, gyr_segments)):
            acc_segment["Norm"] = np.sqrt(acc_segment["X"]**2 + acc_segment["Y"]**2 + acc_segment["Z"]**2)
            gyr_segment["Norm"] = np.sqrt(gyr_segment["X"]**2 + gyr_segment["Y"]**2 + gyr_segment["Z"]**2)
            
            # Prepara il batch con id coerenti
            all_segments.append({
                "Acc": acc_segment,
                "Gyr": gyr_segment,
                "Meta": {
                    "Subject": subject,
                    "Test": test,
                    "Trial": trial,
                    "Bout": bout,
                    "Position": position,
                    "Segment_Index": i
                }
            })

    print(f"Segmentazione completata. Numero totale di segmenti: {len(all_segments)}")

    # Estrazione delle feature in batch
    print("Estrazione delle feature con TSFresh...")
    batch_size = 1000  # Numero di segmenti per batch
    combined_features = []

    for batch_start in tqdm(range(0, len(all_segments), batch_size), desc="Processando batch"):
        batch_end = min(batch_start + batch_size, len(all_segments))
        batch_segments = all_segments[batch_start:batch_end]

        ts_data = []
        meta_data = []
        for segment in batch_segments:
            acc_data = segment["Acc"]
            gyr_data = segment["Gyr"]
            meta = segment["Meta"]

            acc_data["id"] = f"{meta['Subject']}_{meta['Test']}_{meta['Trial']}_{meta['Bout']}_acc_{meta['Segment_Index']}"
            gyr_data["id"] = f"{meta['Subject']}_{meta['Test']}_{meta['Trial']}_{meta['Bout']}_gyr_{meta['Segment_Index']}"
            
            acc_data["value"] = acc_data["Norm"]
            gyr_data["value"] = gyr_data["Norm"]

            ts_data.append(acc_data[["id", "value"]])
            ts_data.append(gyr_data[["id", "value"]])
            meta_data.append(meta)

        ts_data = pd.concat(ts_data, ignore_index=True)

        # Estrai le feature con TSFresh
        features = extract_features(ts_data, column_id="id", column_value="value")

        # Dividi e combina le feature
        acc_features = features.filter(like="_acc").add_prefix("Acc_")
        gyr_features = features.filter(like="_gyr").add_prefix("Gyr_")

        combined_batch = pd.concat([acc_features, gyr_features], axis=1).reset_index(drop=True)
        combined_features.append((pd.DataFrame(meta_data), combined_batch))

    print("Combinazione completata.")

    # Combina i batch finali
    print("Creazione del DataFrame finale...")
    metadata_df = pd.concat([meta for meta, _ in combined_features], ignore_index=True)
    feature_df = pd.concat([features for _, features in combined_features], ignore_index=True)
    final_df = pd.concat([metadata_df, feature_df], axis=1)

    # Esporta il dataset
    print("Esportazione del dataset finale...")
    output_file_path = os.path.join(output_path, "combined_features.csv")
    final_df.to_csv(output_file_path, index=False)
    print("Esportazione completata. Dataset salvato come 'combined_features.csv'.")


