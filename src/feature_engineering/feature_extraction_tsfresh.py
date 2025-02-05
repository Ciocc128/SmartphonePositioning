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
    output_path = "/Volumes/Mac/DatasetSP/pipeline1"
    partial_output_path = os.path.join(output_path, "partial_batches")
    os.makedirs(partial_output_path, exist_ok=True)

    df = pd.read_csv("/Volumes/Mac/DatasetSP/pipeline1/filtered_dataset.csv")
    print(f"Dataset caricato con {len(df)} campioni.")

    # 1. Riordina il dataset
    print("Riordino del dataset...")
    df = df.sort_values(by=["Subject", "Test", "Trial", "Bout", "Position", "Sensor"]).reset_index(drop=True)
    print("Dataset riordinato.")

    # 2. Segmenta il dataset bout per bout
    grouped = df.groupby(["Subject", "Test", "Trial", "Bout", "Position"])

    # Contatori per campioni esclusi
    total_excluded_samples = 0
    small_bout_excluded_samples = 0
    leftover_excluded_samples = 0

    # Funzione per segmentare un gruppo in finestre
    def segment_sensors(group, window_size=100):
        """
        Segmenta un gruppo (bout) in finestre separate per accelerometro e giroscopio,
        allineando temporalmente i dati.
        """
        global total_excluded_samples, small_bout_excluded_samples, leftover_excluded_samples

        # Separiamo accelerometro e giroscopio
        acc_data = group[group["Sensor"] == 1]
        gyr_data = group[group["Sensor"] == 2]
        
        num_acc_samples = len(acc_data)
        num_gyr_samples = len(gyr_data)

        # Gestione bout troppo piccoli
        if num_acc_samples < window_size or num_gyr_samples < window_size:
            small_bout_excluded_samples += num_acc_samples + num_gyr_samples
            total_excluded_samples += num_acc_samples + num_gyr_samples
            return [], []
        
        # Calcolo dei campioni esclusi dalla finestratura
        acc_leftover = num_acc_samples % window_size
        gyr_leftover = num_gyr_samples % window_size
        leftover_excluded_samples += acc_leftover + gyr_leftover
        total_excluded_samples += acc_leftover + gyr_leftover

        # Segmentazione per accelerometro
        acc_segments = [
            acc_data.iloc[i:i+window_size]
            for i in range(0, num_acc_samples - acc_leftover, window_size)
        ]
        
        # Segmentazione per giroscopio
        gyr_segments = [
            gyr_data.iloc[i:i+window_size]
            for i in range(0, num_gyr_samples - gyr_leftover, window_size)
        ]
        
        return acc_segments, gyr_segments

    # Lista per salvare tutte le finestre e i metadati
    all_segments = []
    metadata = []

    # Contatore di segmenti temporali
    segment_counter = 0

    # Itera su ogni bout
    print("Segmentazione dei gruppi...")
    for (subject, test, trial, bout, position), group in tqdm(grouped, desc="Processando gruppi"):
        # Segmenta accelerometro e giroscopio
        acc_segments, gyr_segments = segment_sensors(group)
        
        # Combina le finestre
        for acc_segment, gyr_segment in zip(acc_segments, gyr_segments):
            # Calcola la norma triassiale
            acc_segment = acc_segment.copy()
            acc_segment["Norm"] = np.sqrt(acc_segment["X"]**2 + acc_segment["Y"]**2 + acc_segment["Z"]**2)

            gyr_segment = gyr_segment.copy()
            gyr_segment["Norm"] = np.sqrt(gyr_segment["X"]**2 + gyr_segment["Y"]**2 + gyr_segment["Z"]**2)

            all_segments.append({"Acc": acc_segment, "Gyr": gyr_segment})
            metadata.append({
                "Subject": subject,
                "Test": test,
                "Trial": trial,
                "Bout": bout,
                "Position": position
            })

            # Incrementa il contatore di segmenti
            segment_counter += 1

    print(f"Segmentazione completata. Numero totale di segmenti: {len(all_segments)}")
    print(f"Numero totale di segmenti calcolati dal contatore: {segment_counter}")
    print(f"Campioni esclusi per bout troppo piccoli: {small_bout_excluded_samples}")
    print(f"Campioni esclusi come avanzati dalla finestratura: {leftover_excluded_samples}")
    print(f"Totale campioni esclusi: {total_excluded_samples}")

    # Creazione di batch a livello di finestra temporale
    print("Creazione di batch a livello di finestra temporale...")
    batch_size = 1000  # Numero di finestre temporali per batch
    batches = [
        all_segments[i:i + batch_size]
        for i in range(0, len(all_segments), batch_size)
    ]

    # Estrai le feature con TSFresh e combina accelerometro e giroscopio
    print("Estrazione delle feature con TSFresh in batch...")
    combined_features = []

    for batch_idx, batch_segments in tqdm(enumerate(batches), desc="Processando batch", total=len(batches)):
        ts_data = []
        for idx, segment_pair in enumerate(batch_segments):
            acc_data = segment_pair["Acc"].copy()
            acc_data["id"] = f"{batch_idx}_{idx}_acc"
            acc_data["value"] = acc_data["Norm"]

            gyr_data = segment_pair["Gyr"].copy()
            gyr_data["id"] = f"{batch_idx}_{idx}_gyr"
            gyr_data["value"] = gyr_data["Norm"]

            ts_data.append(acc_data[["id", "value"]])
            ts_data.append(gyr_data[["id", "value"]])

        ts_data = pd.concat(ts_data, ignore_index=True)

        # Estrai le feature per il batch
        features = extract_features(ts_data, column_id="id", column_value="value")

        # Filtra le righe per accelerometro e giroscopio
        acc_features = features[features.index.str.contains("_acc")]
        gyr_features = features[features.index.str.contains("_gyr")]

        # Combina le feature di accelerometro e giroscopio in un'unica riga per segmento
        combined_batch = []
        for acc_id, acc_row in acc_features.iterrows():
            gyr_id = acc_id.replace("_acc", "_gyr")
            if gyr_id in gyr_features.index:
                combined_row = pd.concat([acc_row.add_prefix("Acc_"), gyr_features.loc[gyr_id].add_prefix("Gyr_")])
                combined_batch.append(combined_row)

        # Aggiungi il batch combinato solo se contiene dati
        if combined_batch:
            combined_features.append(pd.DataFrame(combined_batch))

    print("Combinazione completata.")

    # Verifica del numero di righe atteso
    combined_total_rows = sum(len(batch) for batch in combined_features)
    print(f"Numero totale di righe combinate: {combined_total_rows}. Dovrebbe essere circa la metà delle righe originali.")
    print(f"Numero totale di segmenti calcolati dal contatore: {segment_counter}")
    print(f"Campioni esclusi per bout troppo piccoli: {small_bout_excluded_samples}")
    print(f"Campioni esclusi come avanzati dalla finestratura: {leftover_excluded_samples}")
    print(f"Totale campioni esclusi: {total_excluded_samples}")

    # Crea il DataFrame finale
    print("Creazione del DataFrame finale...")
    meta_df = pd.DataFrame(metadata)
    final_features = pd.concat(combined_features, ignore_index=True)
    final_df = pd.concat([meta_df, final_features], axis=1)

    # Esporta il dataset finale
    print("Esportazione del dataset finale...")
    output_file_path = os.path.join(output_path, "combined_features_1.csv")
    final_df.to_csv(output_file_path, index=False)
    print("Esportazione completata. Dataset salvato come 'combined_features_1.csv'.")
