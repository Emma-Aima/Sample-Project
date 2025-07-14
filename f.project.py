# Cleaned and Corrected Version of Your Code

# --------- First Cell: Simulate Health Data ---------
import streamlit as st
import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
pip install scikit-learn
scikit-learn==1.4.2

# Generate timestamps at 1-minute intervals
timestamps = pd.date_range(start='2023-10-01', periods=100, freq='T')

# Simulate health metrics
heart_rate = np.random.randint(60, 100, size=100)
blood_oxygen = np.random.randint(90, 100, size=100)

# Combine into DataFrame
health_data = pd.DataFrame({
    'timestamp': timestamps,
    'heart_rate': heart_rate,
    'blood_oxygen': blood_oxygen
})

# Now this will work
print(health_data.head())


# --------- Second Cell: Basic Random Forest Model ---------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    'heart_rate': np.random.randint(50, 120, data_size),
    'blood_oxygen': np.random.randint(85, 100, data_size),
    'activity_level': np.random.choice(['low', 'moderate', 'high'], size=data_size),
    'label': np.random.choice([0, 1], size=data_size, p=[0.8, 0.2])
})

df['activity_level_encoded'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
X = df[['heart_rate', 'blood_oxygen', 'activity_level_encoded']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --------- Third Cell: Advanced Model + Cross Validation ---------
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-10-01', periods=data_size, freq='T'),
    'heart_rate': np.random.randint(50, 120, data_size),
    'blood_oxygen': np.random.randint(85, 100, data_size),
    'activity_level': np.random.choice(['low', 'moderate', 'high'], size=data_size)
})

def classify_anomaly(row):
    return 'Anomaly' if row['heart_rate'] > 110 or row['blood_oxygen'] < 90 else 'Normal'

df['anomaly'] = df.apply(classify_anomaly, axis=1)
df['activity_encoded'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
X = df[['heart_rate', 'blood_oxygen', 'activity_encoded']]
y = df['anomaly'].apply(lambda x: 1 if x == 'Anomaly' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔎 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5)
print("\n✅ Cross-Validation Accuracy Scores:", cv_scores)
print("🔁 Mean CV Score:", round(cv_scores.mean(), 3))

# --------- Fourth Cell: Flask App with Plotly Chart ---------
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

np.random.seed(42)
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-10-01', periods=50, freq='H'),
    'heart_rate': np.random.randint(60, 130, size=50),
    'blood_oxygen': np.random.randint(85, 100, size=50)
})

latest_hr = df['heart_rate'].iloc[-1]
latest_spo2 = df['blood_oxygen'].iloc[-1]
status = 'Anomaly' if latest_hr > 110 or latest_spo2 < 90 else 'Normal'

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate'], mode='lines+markers', name='Heart Rate'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blood_oxygen'], mode='lines+markers', name='Blood Oxygen'))
fig.update_layout(title='Health Metric Trends', xaxis_title='Time', yaxis_title='Value')
chart_html = pio.to_html(fig, full_html=False)

@app.route('/')
def home():
    return render_template('index.html', data={
        'heart_rate': latest_hr,
        'blood_oxygen': latest_spo2,
        'status': status,
        'chart': chart_html
    })

# --------- Firebase + Token Auth Routes ---------
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred = credentials.Certificate('firebase-key.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    @app.before_request
    def check_token():
        token = request.headers.get('Authorization')
        if token != 'Bearer YOUR_SECRET_KEY':
            return jsonify({'error': 'Unauthorized'}), 401

    @app.route('/upload', methods=['POST'])
    def upload():
        data = request.json
        db.collection('health_metrics').add({
            'heart_rate': data.get('heart_rate'),
            'blood_oxygen': data.get('blood_oxygen'),
            'timestamp': data.get('timestamp')
        })
        return jsonify({'status': 'success'}), 200

    @app.route('/metrics', methods=['GET'])
    def metrics():
        records = db.collection('health_metrics').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        return jsonify([record.to_dict() for record in records])

except Exception as e:
    print("Firebase not configured or missing credentials.", str(e))

# Run server for development
if __name__ == '__main__':
    app.run(debug=True, port=5000)

# --------- Dockerfile (Separate File) ---------
# This Dockerfile content should be placed in a file named 'Dockerfile', not in Python code.
'''
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
'''

# --------- GCloud & Docker CLI Commands (Run in Terminal) ---------
# These are NOT Python and should be run in your terminal
'''
docker build -t weanwise-app .
docker run -p 5000:5000 weanwise-app
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
docker tag weanwise-app gcr.io/YOUR_PROJECT_ID/weanwise-app
docker push gcr.io/YOUR_PROJECT_ID/weanwise-app

gcloud run deploy weanwise-app \
  --image gcr.io/YOUR_PROJECT_ID/weanwise-app \
  --platform managed \
  --region YOUR_REGION \
  --allow-unauthenticated
'''

# --------- Project Folder Structure (Comment for Reference) ---------
'''
# weanwise_app/
# ├── app.py
# ├── requirements.txt
# ├── templates/
# │   └── index.html
# ├── static/
# │   └── chart.js (optional)
# └── firebase-key.json (if using Firestore)
'''
