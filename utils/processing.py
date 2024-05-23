import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def process_ppg_signal(file_path):
    ppg_data = pd.read_csv(file_path)
    ppg_signal = ppg_data[' HR'].values

    ppg_signal_norm = (ppg_signal - np.min(ppg_signal)) / (np.max(ppg_signal) - np.min(ppg_signal))
    wavelet = 'db4'
    level = 5

    coeffs = pywt.wavedec(ppg_signal_norm, wavelet, level=level)
    threshold = 0.6745 * np.median(np.abs(coeffs[-level]))
    
    denoised_coeffs = [coeffs[0]] + [soft_threshold(coeff, threshold) for coeff in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    nyquist_freq = 125 / 2
    b_filt, a_filt = butter(2, [5 / nyquist_freq, 35 / nyquist_freq], btype='bandpass', analog=False)
    fil_sig = filtfilt(b_filt, a_filt, denoised_signal)

    peaks, _ = find_peaks(fil_sig, height=0)
    return ppg_signal, fil_sig, denoised_signal, peaks

def soft_threshold(coefficients, threshold):
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0)

def create_segments(fil_sig, segment_length=60, step_size=60):
    num_segments = int(np.floor((len(fil_sig) - segment_length) / step_size)) + 1
    segments = np.zeros((num_segments, segment_length))
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_length
        segments[i, :] = fil_sig[start_idx:end_idx]
    return segments

def cluster_segments(segments, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=1)
    cluster_labels = kmeans.fit_predict(segments)
    return cluster_labels

def calculate_hrv(peaks):
    rr_intervals = np.diff(peaks)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return sdnn, rmssd

def plot_pie_chart():
    labels = 'Light Sleep (N1)', 'Moderate Sleep (N2)', 'Deep Sleep (N3)', 'REM Sleep'
    sizes = [5, 50, 20, 25]  # Example distribution in percentage
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sleep Stages Distribution')
    plt.savefig('static/pie_chart.png')
    plt.close()


