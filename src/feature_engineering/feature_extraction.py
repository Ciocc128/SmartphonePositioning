import pandas as pd
import numpy as np
from tsfel import *
import os
from tqdm import tqdm

# Caricamento del dataset
print("Caricamento del dataset...")
output_path = "/Volumes/Mac/DatasetSP/preprocessed"
df = pd.read_csv("/Volumes/Mac/DatasetSP/preprocessed/combined_dataset.csv")
print(f"Dataset caricato con {len(df)} campioni.")

# 1. Riordina il dataset
print("Riordino del dataset...")
df = df.sort_values(by=["Subject", "Test", "Trial", "Bout", "Position", "Sensor"]).reset_index(drop=True)
print("Dataset riordinato.")

# Configurazione TSFEL
print("Configurazione delle feature TSFEL...")
cfg = tsfel.get_features_by_domain()

# 2. Segmenta il dataset bout per bout
grouped = df.groupby(["Subject", "Test", "Trial", "Bout", "Position"])

# Funzione per segmentare un gruppo in finestre
def segment_sensors(group, window_size=100):
    """
    Segmenta un gruppo (bout) in finestre separate per accelerometro e giroscopio,
    allineando temporalmente i dati.
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

# Estrai le feature con TSFEL
print("Estrazione delle feature con TSFEL...")
combined_features = []
for segment_pair, meta in tqdm(zip(all_segments, metadata), desc="Estrazione delle feature", total=len(all_segments)):
    absEnergy = tsfel.abs_energy(segment_pair["Acc"]["Norm"])
    print(absEnergy)

    # Estrai le feature per accelerometro
    acc_features = tsfel.time_series_features_extractor(cfg, segment_pair["Acc"]["Norm"], fs=100, verbose=0)
    # Estrai le feature per giroscopio
    gyr_features = tsfel.time_series_features_extractor(cfg, segment_pair["Gyr"]["Norm"], fs=100, verbose=0)

    # Combina le feature di accelerometro e giroscopio
    combined = pd.concat([acc_features.reset_index(drop=True), gyr_features.reset_index(drop=True)], axis=1)

    # Aggiungi metadati
    combined["Subject"] = meta["Subject"]
    combined["Test"] = meta["Test"]
    combined["Trial"] = meta["Trial"]
    combined["Bout"] = meta["Bout"]
    combined["Position"] = meta["Position"]

    combined_features.append(combined)

print("Feature extraction completata.")
# Crea il DataFrame finale
print("Creazione del DataFrame finale...")
final_df = pd.concat(combined_features, ignore_index=True)

# Esporta il dataset finale
print("Esportazione del dataset finale...")
output_file_path = os.path.join(output_path, "combined_features.csv")
final_df.to_csv(output_file_path, index=False)
print("Esportazione completata. Dataset salvato come 'combined_features.csv'.")
