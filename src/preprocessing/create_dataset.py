import os
import pandas as pd
from tqdm import tqdm
# Percorso della directory principale
base_path = "/Volumes/Mac/DatasetSP/hot data"
output_path = "/Volumes/Mac/DatasetSP/preprocessed"

# Mappa delle posizioni a etichette numeriche
position_map = {
    "LB": 0,  # Lower Back
    "BP": 1,  # Back Pocket
    "FP": 2,  # Front Pocket
    "H": 3,   # Hand
    "SB": 4,   # Shoulder Bag
    "CP": 5,  # Coat Pocket
}

# Lista per raccogliere tutti i dati
data_list = []

# Loop sulle cartelle dei soggetti
for subject_folder in tqdm(os.listdir(base_path), desc="Processing subjects"):
    subject_path = os.path.join(base_path, subject_folder)
    
    if os.path.isdir(subject_path):
        # Loop sui file all'interno della cartella del soggetto
        for file_name in tqdm(os.listdir(subject_path)):
            if file_name.endswith(".csv"):  # Considera solo i file CSV
                # Estrai informazioni dal nome del file
                parts = file_name.split("-")
                subject_id = parts[0]  # Es: "001"
                test = int(parts[1].split("Test")[-1])  # Es: "Test4" -> 4
                # Gestisci il caso in cui manca il trial (Test10)
                if test == 10:
                    trial = 4  # Real World
                    bout = parts[2]  # Es: "1"
                    position_sensor = parts[3]  # Es: "BP_Acc.csv"
                else:
                    trial = int(parts[2].split("Trial")[-1])  # Es: "Trial1" -> 1
                    bout = parts[3]  # Es: "1"
                    position_sensor = parts[4]  # Es: "BP_Acc.csv"
                # Split per posizione e tipo di sensore
                position, sensor = position_sensor.split("_")
                sensor = sensor.replace(".csv", "")  # Rimuove l'estensione
                
                # Leggi i dati dal CSV (senza intestazione, aggiungi manualmente X, Y, Z)
                file_path = os.path.join(subject_path, file_name)
                df = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'])
                
                # Aggiungi metadati al DataFrame
                df["Subject"] = subject_id
                df["Test"] = test
                df["Trial"] = trial
                df["Bout"] = bout
                df["Position"] = position_map[position]  # Converti posizione in numero
                if sensor == "Acc":
                    df["Sensor"] = 1
                elif sensor == "Gyr":
                    df["Sensor"] = 2
                
                # Aggiungi il DataFrame alla lista
                data_list.append(df)

print(f"Numero di file processati: {len(data_list)}")

# Unisci tutti i dati in un unico DataFrame
full_dataset = pd.concat(data_list, ignore_index=True)

# Salva il dataset combinato in un file
output_file_path = os.path.join(output_path, "combined_dataset.csv")
full_dataset.to_csv(output_file_path, index=False)
print(f"Dataset combinato salvato in {output_file_path}!")
