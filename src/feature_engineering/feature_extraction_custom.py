import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.stats import entropy

def extract_custom_features(ts_data, column_id="id", fs=100):
    feature_dict = {}
    grouped = ts_data.groupby(column_id)
    for idx, group in grouped:
        x_values = group["X"].values
        y_values = group["Y"].values
        z_values = group["Z"].values
        norm_values = np.sqrt(x_values**2 + y_values**2 + z_values**2)
        
        def extract_signal_features(values):
            var_value = np.var(values)
            rms_value = np.sqrt(np.mean(values ** 2))
            sma_value = np.mean(np.abs(values))
            num_peaks_value = len(find_peaks(values)[0])
            values_fft = np.abs(fft(values - np.mean(values)))
            freqs = np.fft.fftfreq(len(values), d=1/fs)
            dom_power_value = np.max(values_fft)
            dom_freq_value = freqs[np.argmax(values_fft)]
            spec_entropy_value = entropy(values_fft / np.sum(values_fft))
            return [var_value, rms_value, sma_value, num_peaks_value, dom_freq_value, dom_power_value, spec_entropy_value]
        
        features_x = extract_signal_features(x_values)
        features_y = extract_signal_features(y_values)
        features_z = extract_signal_features(z_values)
        features_norm = extract_signal_features(norm_values)
        feature_dict[idx] = features_x + features_y + features_z + features_norm
    
    feature_columns = [f"{axis}_{feat}" for axis in ["X", "Y", "Z", "Norm"] for feat in ["Var", "RMS", "SMA", "NumPeaks", "DomFreq", "DomPower", "SpecEntropy"]]
    feature_df = pd.DataFrame.from_dict(feature_dict, orient="index", columns=feature_columns)
    feature_df.index.name = column_id
    return feature_df

if __name__ == "__main__":

    # Caricamento del dataset
    print("Caricamento del dataset...")
    output_path = "/Volumes/Mac/DatasetSP/pipeline2"
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

    # Funzione per segmentare un gruppo in finestre con overlap del 50%
    def segment_sensors(group, window_size=300, overlap=0.5):
        """
        Segmenta un gruppo (bout) in finestre separate per accelerometro e giroscopio,
        allineando temporalmente i dati con un overlap specificato.
        """
        global total_excluded_samples, small_bout_excluded_samples, leftover_excluded_samples

        step_size = int(window_size * (1 - overlap))  # Calcola il passo per la sovrapposizione

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
        
        # Segmentazione per accelerometro con overlap
        acc_segments = [
            acc_data.iloc[i:i+window_size]
            for i in range(0, num_acc_samples - window_size + 1, step_size)
        ]
        
        # Segmentazione per giroscopio con overlap
        gyr_segments = [
            gyr_data.iloc[i:i+window_size]
            for i in range(0, num_gyr_samples - window_size + 1, step_size)
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
            gyr_data = segment_pair["Gyr"].copy()
            gyr_data["id"] = f"{batch_idx}_{idx}_gyr"
            ts_data.append(acc_data[["id", "X", "Y", "Z"]])
            ts_data.append(gyr_data[["id", "X", "Y", "Z"]])

        ts_data = pd.concat(ts_data, ignore_index=True)

        # Estrai le feature per il batch
        features = extract_custom_features(ts_data, column_id="id")
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

    # Crea il DataFrame finale
    print("Creazione del DataFrame finale...")
    meta_df = pd.DataFrame(metadata)
    final_features = pd.concat(combined_features, ignore_index=True)
    final_df = pd.concat([meta_df, final_features], axis=1)

    # Esporta il dataset finale
    print("Esportazione del dataset finale...")
    output_file_path = os.path.join(output_path, "combined_features_2.2.csv")
    final_df.to_csv(output_file_path, index=False)
    print("Esportazione completata. Dataset salvato come 'combined_features_2.2.csv'.")
