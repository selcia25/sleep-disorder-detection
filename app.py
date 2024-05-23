from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.cluster import KMeans
from utils.processing import process_ppg_signal, plot_pie_chart
from keras.models import load_model
import warnings 
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Function to plot a pie chart
def plot_pie_chart(normal_count, abnormal_count):
    labels = 'Normal', 'Abnormal'
    sizes = [normal_count, abnormal_count]
    colors = ['#1acebf', '#ffccf3']
    explode = (0.1, 0)  # explode the 1st slice (Normal)

    plt.figure(figsize=(8,8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    pie_chart_path = os.path.join('static', 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process PPG signal and detect peaks
            ppg_signal, fil_sig, denoised_signal, peaks = process_ppg_signal(file_path)

            # Define segment length and step size
            segment_length = 60  # 60 seconds
            step_size = 60  # 60 seconds

            # Calculate number of segments
            num_segments = int(np.floor((len(fil_sig) - segment_length) / step_size)) + 1

            # Initialize array to store segments
            segments = np.zeros((num_segments, segment_length))

            # Extract segments using sliding window
            for i in range(num_segments):
                start_idx = i * step_size
                end_idx = start_idx + segment_length
                segments[i, :] = fil_sig[start_idx:end_idx]

            # Perform clustering
            num_clusters = 3  # Define number of clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=1)
            cluster_labels = kmeans.fit_predict(segments)

            # Load the pre-trained model
            model = load_model('segment_classifier_model.h5')

            # Reshape segments for the model
            segments = segments.reshape((segments.shape[0], segments.shape[1], 1))

            # Classify sleep stages using the pre-trained model
            sleep_stages = model.predict(segments)
            sleep_stages = (sleep_stages > 0.5).astype(int)  # Assuming binary classification

            # Count normal and abnormal sleep stages
            normal_count = np.sum(sleep_stages == 0)
            abnormal_count = np.sum(sleep_stages == 1)

            
            
            # Plot the signals
            plt.figure(figsize=(8,4))

            # Plot original PPG signal
            plt.subplot(3, 1, 1)
            plt.plot(ppg_signal, label='PPG Signal', color='black')
            plt.legend()
            plt.title('Original PPG Signal')

            # Plot denoised signal
            plt.subplot(3, 1, 2)
            plt.plot(denoised_signal, label='Denoised Signal', color='blue')
            plt.legend()
            plt.title('Denoised PPG Signal')

            # Plot filtered signal
            plt.subplot(3, 1, 3)
            plt.plot(fil_sig, label='Filtered Signal', color='green')
            plt.plot(peaks, fil_sig[peaks], "x", color='red', label='Detected Peaks')
            plt.legend()
            plt.title('Filtered PPG Signal with Detected Peaks')

            plt.tight_layout()
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)
            plt.close()
            # Prepare data for rendering
            segment_info = []
            for i in range(num_segments):
                segment_info.append({
                    'segment_no': i + 1,
                    'cluster': cluster_labels[i],
                    'sleep_stage': sleep_stages[i][0]
                })
            # Plot the pie chart
            plt.figure(figsize=(4,2))
            pie_chart_path = plot_pie_chart(normal_count, abnormal_count)
            # Provide suggestions based on sleep stage classification
            if abnormal_count / (normal_count + abnormal_count) > 0.5:
                suggestions = "To improve your sleep, consider the following: Maintain a regular sleep schedule, avoid caffeine and heavy meals before bedtime, create a restful environment, and limit exposure to screens before sleep."
            else:
                suggestions = "Your sleep stages are mostly normal. To maintain this, continue practicing good sleep hygiene, such as keeping a consistent sleep schedule, creating a comfortable sleep environment, and managing stress effectively."
            return render_template('visualize.html', plot_path=plot_path, segment_info=segment_info, pie_chart_path=pie_chart_path, suggestions=suggestions)

    return render_template('upload.html')

@app.route('/visualize/<filename>')
def visualize(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ppg_signal, fil_sig, denoised_signal, peaks = process_ppg_signal(file_path)

    plt.figure(figsize=(12, 6))
    plt.plot(fil_sig, label='Filtered Signal', color='green')
    plt.plot(peaks, fil_sig[peaks], "x", color='red', label='Detected Peaks')
    plt.legend()
    plt.title('Filtered PPG Signal with Detected Peaks')
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('visualize.html', plot_path=plot_path)

@app.route('/learnmore')
def learnmore():
    return render_template('learn_more.html',)

@app.route('/suggestions')
def suggestions():
    
    return render_template('suggestions.html')

if __name__ == '__main__':
    app.run(debug=True)
