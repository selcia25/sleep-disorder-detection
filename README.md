# Sleep Disorder Detection from Heart Rate Signals

This project aims to develop an automated method for detecting sleep disorders from heart rate signals collected using a pulse oximeter. Sleep disorders can significantly impact an individual's health and quality of life. Early detection of these disorders is crucial for timely interventions and improved outcomes.

## Problem Statement
- Sleep disorders are prevalent and can impact health and quality of life.
- Current methods for diagnosing sleep disorders are manual, time-consuming, and subjective.
- There is a need for an automated method to detect sleep disorders from heart rate signals to improve efficiency and accuracy.

## Solution Approach
- Preprocess signals for denoising and feature extraction.
- Cluster heart rate signals using K-means clustering.
- Segment signals into shorter segments.
- Classify segments using a Convolutional Neural Network (CNN).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/selcia25/sleep-disorder-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Usage
- Upload heart rate signal data in CSV format.
- View preprocessed data and segmented signals.
- Receive classification results indicating the presence of sleep disorders.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
