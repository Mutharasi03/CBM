from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle
import os
import base64
from scipy.fftpack import fft
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime
import uuid
import seaborn as sns

app = Flask(__name__)

os.makedirs('uploads', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def load_models():
    models = {}
    models['fault_model'] = pickle.load(open('models/fault_model.pkl', 'rb'))
    models['severity_model'] = pickle.load(open('models/severity_model.pkl', 'rb'))
    models['area_model'] = pickle.load(open('models/area_model.pkl', 'rb'))
    models['scaler'] = pickle.load(open('models/scaler.pkl', 'rb'))
    models['pca'] = pickle.load(open('models/pca.pkl', 'rb'))
    models['label_encoders'] = pickle.load(open('models/label_encoders.pkl', 'rb'))
     # Load anomaly detection models
    models['anomaly_scaler'] = pickle.load(open('models/anomaly_scaler.pkl', 'rb'))
    models['svm_anomaly_model'] = pickle.load(open('models/svm_model.pkl', 'rb'))
    
    return models

def extract_time_features(signal):
    return {
        'Mean': np.mean(signal),
        'RMS': np.sqrt(np.mean(signal**2)),
        'Std': np.std(signal),
        'Kurtosis': kurtosis(signal),
        'Skewness': skew(signal),
        'Crest Factor': np.max(signal) / np.sqrt(np.mean(signal**2)),
        'Peak-to-Peak': np.ptp(signal)
    }

def extract_frequency_features(signal, sampling_rate=50000):
    n = len(signal)
    freq_spectrum = np.abs(fft(signal))[:n // 2]
    freq_bins = np.linspace(0, sampling_rate / 2, len(freq_spectrum))

    freqs, psd = welch(signal, fs=sampling_rate)
    
    # Handle potential division by zero or log of zero
    psd_normalized = psd / np.sum(psd) if np.sum(psd) > 0 else np.ones_like(psd) / len(psd)
    psd_normalized = np.where(psd_normalized <= 0, 1e-10, psd_normalized)
    
    return {
        'FFT Energy': np.sum(freq_spectrum**2),
        'Spectral Entropy': -np.sum(psd_normalized * np.log2(psd_normalized)),
        'Dominant Frequency': freq_bins[np.argmax(freq_spectrum)] if len(freq_spectrum) > 0 else 0,
        'Bandwidth': np.sqrt(np.sum((freq_bins - np.mean(freq_bins))**2 * freq_spectrum) / np.sum(freq_spectrum)) if np.sum(freq_spectrum) > 0 else 0
    }

def process_vibration_data(csv_file, window_size=50000, overlap=5000, sampling_rate=50000):
    # Load the CSV without headers
    df = pd.read_csv(csv_file, header=None)

    # Define column names
    df.columns = [
        "Tachometer", "Underhang_Axial", "Underhang_Radial", "Underhang_Tangential",
        "Overhang_Axial", "Overhang_Radial", "Overhang_Tangential", "Microphone"
    ]

    # Sensor columns for segmentation (excluding Tachometer & Microphone)
    sensor_columns = ['Underhang_Axial', 'Underhang_Radial', 'Underhang_Tangential',
                      'Overhang_Axial', 'Overhang_Radial', 'Overhang_Tangential']

    # Perform segmentation using sliding window
    features = []
    num_samples = len(df)

    for start in range(0, num_samples - window_size, overlap):
        end = start + window_size
        segment = df.iloc[start:end]

        # Extract features for each window
        window_features = {'Start_Index': start, 'End_Index': end}

        for col in sensor_columns:
            signal = segment[col].values
            time_features = extract_time_features(signal)
            freq_features = extract_frequency_features(signal, sampling_rate)

            # Prefix feature names with sensor name
            for key, value in {**time_features, **freq_features}.items():
                window_features[f"{col}_{key}"] = value

        features.append(window_features)

    # Convert to DataFrame
    segmented_df = pd.DataFrame(features)
    
    return segmented_df, df

def make_prediction(features_df, models):
    # Extract feature columns, dropping start/end index
    X = features_df.drop(columns=['Start_Index', 'End_Index'], errors='ignore')
    
    # Scale the features
    X_scaled = models['scaler'].transform(X)
    
    # Apply PCA
    X_pca = models['pca'].transform(X_scaled)
    
    # Make predictions
    fault_pred = models['fault_model'].predict(X_pca)
    
    # Get majority vote from segments
    fault_class = np.bincount(fault_pred).argmax()
    fault_label = models['label_encoders']['class'].inverse_transform([fault_class])[0]
    
    results = {
        'fault_class': fault_label,
        'is_faulty': fault_label != 'Normal'
    }
    
    # If fault detected, predict severity and area
    if results['is_faulty']:
        severity_pred = models['severity_model'].predict(X_pca)
        severity_class = np.bincount(severity_pred).argmax()
        severity_label = models['label_encoders']['severity'].inverse_transform([severity_class])[0]
        
        area_pred = models['area_model'].predict(X_pca)
        area_class = np.bincount(area_pred).argmax()
        area_label = models['label_encoders']['area'].inverse_transform([area_class])[0]
        
        results['severity'] = severity_label
        results['area'] = area_label
    
    return results

def generate_visualizations(raw_data, filename_prefix):
    # Data for visualization
    sensor_columns = ['Underhang_Axial', 'Underhang_Radial', 'Underhang_Tangential',
                      'Overhang_Axial', 'Overhang_Radial', 'Overhang_Tangential']
    
    time_domain_images = []
    freq_domain_images = []
    
    for col in sensor_columns:
        # Time domain plot
        plt.figure(figsize=(10, 4))
        plt.plot(raw_data[col].values[:5000])  # Plot first 5000 samples for clarity
        plt.title(f'Time Domain - {col}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Save to BytesIO
        time_img_io = io.BytesIO()
        plt.savefig(time_img_io, format='png', bbox_inches='tight')
        time_img_io.seek(0)
        time_domain_images.append({
            'name': col,
            'img': base64.b64encode(time_img_io.getvalue()).decode('utf-8')
        })
        plt.close()
        
        # Frequency domain plot
        signal = raw_data[col].values[:50000]  # Use 50k samples for FFT
        n = len(signal)
        freq_spectrum = np.abs(fft(signal))[:n // 2]
        freq_bins = np.linspace(0, 50000 / 2, len(freq_spectrum))
        
        plt.figure(figsize=(10, 4))
        plt.plot(freq_bins, freq_spectrum)
        plt.title(f'Frequency Domain - {col}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 5000)  # Limit to 5kHz for better visibility
        
        # Save to BytesIO
        freq_img_io = io.BytesIO()
        plt.savefig(freq_img_io, format='png', bbox_inches='tight')
        freq_img_io.seek(0)
        freq_domain_images.append({
            'name': col,
            'img': base64.b64encode(freq_img_io.getvalue()).decode('utf-8')
        })
        plt.close()
    
    return time_domain_images, freq_domain_images

def create_pdf_report(results, time_images, freq_images, report_id):
    report_path = f"reports/report_{report_id}.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,
        spaceAfter=12
    )
    story.append(Paragraph("Vibration Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        spaceAfter=12
    )
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {current_date}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Results section
    story.append(Paragraph("Analysis Results", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Results table
    result_data = [["Parameter", "Value"]]
    result_data.append(["Fault Classification", results['fault_class']])
    
    if results['is_faulty']:
        result_data.append(["Fault Severity", results['severity']])
        result_data.append(["Fault Area", results['area']])
    
    result_table = Table(result_data, colWidths=[2.5*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Summary text
    summary_text = "The machine is "
    if results['is_faulty']:
        summary_text += f"showing signs of a {results['severity']} fault in the {results['fault_class']} at the {results['area']} location. "
        
        # Add recommendations based on fault type
        if 'Cage' in results['fault_class']:
            summary_text += "Recommendation: Inspect the bearing cage for wear, deformation, or damage. Consider replacement if damage is confirmed."
        elif 'Ball' in results['fault_class']:
            summary_text += "Recommendation: Replace the bearing. Ball faults typically indicate the bearing has reached the end of its service life."
        elif 'Outer Race' in results['fault_class']:
            summary_text += "Recommendation: Replace the bearing. Outer race faults indicate significant wear that cannot be repaired."
    else:
        summary_text += "operating normally with no detected faults."
    
    story.append(Paragraph("Summary:", styles['Heading3']))
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Add visualizations
    story.append(Paragraph("Time Domain Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("The following time domain plots show the raw vibration signals:", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    # Add time domain images in pairs
    for i in range(0, len(time_images), 2):
        # Create a 1x2 table for each pair of images
        img_data = []
        img_row = []
        
        # First image in pair
        img_buf = io.BytesIO(base64.b64decode(time_images[i]['img']))
        img = Image(img_buf, width=3*inch, height=1.5*inch)
        img_cell = [img, Paragraph(time_images[i]['name'], styles['Normal'])]
        img_row.append(img_cell)
        
        # Second image in pair (if available)
        if i+1 < len(time_images):
            img_buf = io.BytesIO(base64.b64decode(time_images[i+1]['img']))
            img = Image(img_buf, width=3*inch, height=1.5*inch)
            img_cell = [img, Paragraph(time_images[i+1]['name'], styles['Normal'])]
            img_row.append(img_cell)
        else:
            img_row.append([])
        
        img_data.append(img_row)
        
        # Create and style the table
        img_table = Table(img_data, colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Add frequency domain visualizations
    story.append(Paragraph("Frequency Domain Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("The following frequency domain plots show the frequency components of the vibration signals:", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    # Add frequency domain images in pairs
    for i in range(0, len(freq_images), 2):
        # Create a 1x2 table for each pair of images
        img_data = []
        img_row = []
        
        # First image in pair
        img_buf = io.BytesIO(base64.b64decode(freq_images[i]['img']))
        img = Image(img_buf, width=3*inch, height=1.5*inch)
        img_cell = [img, Paragraph(freq_images[i]['name'], styles['Normal'])]
        img_row.append(img_cell)
        
        # Second image in pair (if available)
        if i+1 < len(freq_images):
            img_buf = io.BytesIO(base64.b64decode(freq_images[i+1]['img']))
            img = Image(img_buf, width=3*inch, height=1.5*inch)
            img_cell = [img, Paragraph(freq_images[i+1]['name'], styles['Normal'])]
            img_row.append(img_cell)
        else:
            img_row.append([])
        
        img_data.append(img_row)
        
        # Create and style the table
        img_table = Table(img_data, colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Build the PDF
    doc.build(story)
    return report_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Process the file and redirect to results page
        return redirect(url_for('analyze', file_path=file_path, analysis_id=analysis_id))

@app.route('/analyze')
def analyze():
    file_path = request.args.get('file_path')
    analysis_id = request.args.get('analysis_id')
    
    try:
        # Load models
        models = load_models()
        
        # Process vibration data
        features_df, raw_data = process_vibration_data(file_path)
        
        # Make predictions
        results = make_prediction(features_df, models)
        
        # Generate visualizations
        time_images, freq_images = generate_visualizations(raw_data, analysis_id)
        
        # Create the report PDF
        report_path = create_pdf_report(results, time_images, freq_images, analysis_id)
        
        return render_template('results.html', 
                              results=results,
                              time_images=time_images,
                              freq_images=freq_images,
                              analysis_id=analysis_id)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/download_report/<analysis_id>')
def download_report(analysis_id):
    report_path = f"reports/report_{analysis_id}.pdf"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name="vibration_analysis_report.pdf")
    else:
        return "Report not found", 404

if __name__ == '__main__':
    app.run(debug=True)