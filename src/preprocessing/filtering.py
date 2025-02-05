# Group by walking bouts
perwb_data = data.groupby(["Subject", "Test", "Trial", "Bout", "Position", "Sensor"])
# Filtering the signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parametri del filtro Butterworth
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Parametri di configurazione del filtro
fs = 100  # Frequenza di campionamento stimata (Hz)
cutoff = 20  # Frequenza di taglio (Hz)
order = 4  # Ordine del filtro

# Filtra un bout specifico
def plot_filtered_bout(data, subject, test, trial, bout, sensor):
    # Filtra i dati per il bout e il sensore specificato
    bout_data = data[
        (data["Subject"] == subject) &
        (data["Test"] == test) &
        (data["Trial"] == trial) &
        (data["Bout"] == bout) &
        (data["Sensor"] == sensor)
    ]

    # Estrai le colonne di interesse
    time = np.arange(len(bout_data)) / fs  # Assegna un asse temporale
    x = bout_data["X"].values
    y = bout_data["Y"].values
    z = bout_data["Z"].values

    # Applica il filtro
    x_filtered = butter_lowpass_filter(x, cutoff, fs, order)
    y_filtered = butter_lowpass_filter(y, cutoff, fs, order)
    z_filtered = butter_lowpass_filter(z, cutoff, fs, order)

    # Seleziona i primi 50 campioni
    num_samples = 50
    time_zoomed = np.arange(num_samples) / fs  # Zoom sull'asse temporale per 50 campioni
    x_zoomed = x[:num_samples]
    x_filtered_zoomed = x_filtered[:num_samples]
    y_zoomed = y[:num_samples]
    y_filtered_zoomed = y_filtered[:num_samples]
    z_zoomed = z[:num_samples]
    z_filtered_zoomed = z_filtered[:num_samples]

    # Plot dei risultati
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    # Grafici per X
    axes[0, 0].plot(time_zoomed, x_zoomed, label="Originale", alpha=0.7)
    axes[0, 0].set_title("X - Segnale Originale (Zoomato)")
    axes[0, 1].plot(time_zoomed, x_filtered_zoomed, label="Filtrato", color='orange')
    axes[0, 1].set_title("X - Segnale Filtrato (Zoomato)")

    # Grafici per Y
    axes[1, 0].plot(time_zoomed, y_zoomed, label="Originale", alpha=0.7)
    axes[1, 0].set_title("Y - Segnale Originale (Zoomato)")
    axes[1, 1].plot(time_zoomed, y_filtered_zoomed, label="Filtrato", color='orange')
    axes[1, 1].set_title("Y - Segnale Filtrato (Zoomato)")

    # Grafici per Z
    axes[2, 0].plot(time_zoomed, z_zoomed, label="Originale", alpha=0.7)
    axes[2, 0].set_title("Z - Segnale Originale (Zoomato)")
    axes[2, 1].plot(time_zoomed, z_filtered_zoomed, label="Filtrato", color='orange')
    axes[2, 1].set_title("Z - Segnale Filtrato (Zoomato)")

    # Layout e legenda
    for ax in axes.flat:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.xlabel("Tempo (secondi)")
    plt.show()


# Carica un esempio di dati (sostituire con il tuo dataset)
# data = pd.read_csv("dataset.csv")  # Inserire il file corretto

# Esegui il confronto per un bout specifico
# Esempio: soggetto 15, test 10, trial 4, bout 19, sensore accelerometro (1)
# plot_filtered_bout(data, subject=15, test=10, trial=4, bout=19, sensor=1)

# Esempio: soggetto 15, test 10, trial 4, bout 19, sensore giroscopio (2)
# plot_filtered_bout(data, subject=15, test=10, trial=4, bout=19, sensor=2)

plot_filtered_bout(data, subject=9, test=10, trial=4, bout=19, sensor=1)

plot_filtered_bout(data, subject=9, test=10, trial=4, bout=19, sensor=2)
filtered_data = []

# Raggruppa per Subject, Test, Trial, Bout, Position e Sensor
for (subject, test, trial, bout, position, sensor), group in perwb_data:
    # Applica il filtro alle colonne X, Y, Z
    group_filtered = group.copy()
    group_filtered["X"] = butter_lowpass_filter(group["X"], cutoff, fs, order)
    group_filtered["Y"] = butter_lowpass_filter(group["Y"], cutoff, fs, order)
    group_filtered["Z"] = butter_lowpass_filter(group["Z"], cutoff, fs, order)
    
    # Aggiungi il gruppo filtrato alla lista
    filtered_data.append(group_filtered)

# Combina tutti i gruppi filtrati in un unico dataset
filtered_dataset = pd.concat(filtered_data)

# Ordina per mantenere l'ordine originale (facoltativo)
filtered_dataset = filtered_dataset.sort_index()

# Mostra i primi dati filtrati
print(filtered_dataset.head())