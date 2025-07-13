# First Cell
import pandas as pd
import numpy as np

# Generate timestamps at 1-minute intervals
timestamps = pd.date_range(start='2023-10-01', periods=100, freq='T')

# Simulate health metrics
heart_rate = np.random.randint(60, 100, size=100)            # Typical heart rate range for infants
blood_oxygen = np.random.randint(90, 100, size=100)          # Oxygen saturation levels (SpO₂)
activity_level = np.random.choice(['low', 'moderate', 'high'], size=100)  # Activity classifications

# Assemble data into a DataFrame
health_data = pd.DataFrame({
    'timestamp': timestamps,
    'heart_rate': heart_rate,
    'blood_oxygen': blood_oxygen,
    'activity_level': activity_level
})

# Preview the first five rows
print(health_data.head())


# Second Cell
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulate labeled health data
np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    'heart_rate': np.random.randint(50, 120, data_size),
    'blood_oxygen': np.random.randint(85, 100, data_size),
    'activity_level': np.random.choice(['low', 'moderate', 'high'], size=data_size),
    # Label: 0 - Normal, 1 - Abnormal
    'label': np.random.choice([0, 1], size=data_size, p=[0.8, 0.2])
})

# Convert categorical activity_level to numerical
df['activity_level_encoded'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})

# Features and target
X = df[['heart_rate', 'blood_oxygen', 'activity_level_encoded']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Third Cell
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Step 1: Simulate health data
# ----------------------------
np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-10-01', periods=data_size, freq='T'),
    'heart_rate': np.random.randint(50, 120, data_size),
    'blood_oxygen': np.random.randint(85, 100, data_size),
    'activity_level': np.random.choice(['low', 'moderate', 'high'], size=data_size)
})

# -----------------------------------------------
# Step 2: Generate anomaly labels for the dataset
# -----------------------------------------------
def classify_anomaly(row):
    if row['heart_rate'] > 110 or row['blood_oxygen'] < 90:
        return 'Anomaly'
    else:
        return 'Normal'

df['anomaly'] = df.apply(classify_anomaly, axis=1)

# -----------------------------------------------------
# Step 3: Feature engineering and label transformation
# -----------------------------------------------------
df['activity_encoded'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
X = df[['heart_rate', 'blood_oxygen', 'activity_encoded']]
y = df['anomaly'].apply(lambda x: 1 if x == 'Anomaly' else 0)

# --------------------------------------
# Step 4: Train-test split and model fit
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------
# Step 5: Model predictions and evaluation
# ------------------------------------------
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔎 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------------
# Step 6: Cross-validation for model robustness
# ----------------------------------------------
cv_scores = cross_val_score(model, X, y, cv=5)
print("✅ Cross-Validation Accuracy Scores:", cv_scores)
print("🔁 Mean CV Score:", round(cv_scores.mean(), 3))


# Fourth Cell
from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# ----------------------------
# Simulate health data
# ----------------------------
np.random.seed(42)
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-10-01', periods=50, freq='H'),
    'heart_rate': np.random.randint(60, 130, size=50),
    'blood_oxygen': np.random.randint(85, 100, size=50)
})

# ----------------------------
# Detect health status
# ----------------------------
latest_hr = df['heart_rate'].iloc[-1]
latest_spo2 = df['blood_oxygen'].iloc[-1]
status = 'Anomaly' if latest_hr > 110 or latest_spo2 < 90 else 'Normal'

# ----------------------------
# Create interactive chart
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['heart_rate'],
    mode='lines+markers', name='Heart Rate'
))
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['blood_oxygen'],
    mode='lines+markers', name='Blood Oxygen'
))
fig.update_layout(
    title='Health Metric Trends',
    xaxis_title='Time',
    yaxis_title='Value'
)

chart_html = pio.to_html(fig, full_html=False)

# ----------------------------
# Flask route
# ----------------------------
@app.route('/')
def home():
    latest_data = {
        'heart_rate': latest_hr,
        'blood_oxygen': latest_spo2,
        'status': status,
        'chart': chart_html
    }
    return render_template('index.html', data=latest_data)

# ----------------------------
# Run server in prototype mode
# ----------------------------
from werkzeug.serving import run_simple
run_simple("localhost", 5000, app, use_debugger=True)


# Fifth Cell 
# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask’s default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]


# Sixth Cell
docker build -t weanwise-app .
docker run -p 5000:5000 weanwise-app
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

docker tag weanwise-app gcr.io/YOUR_PROJECT_ID/weanwise-app
docker push gcr.io/YOUR_PROJECT_ID/weanwise-app


# Seventh Cell
gcloud run deploy weanwise-app \
  --image gcr.io/YOUR_PROJECT_ID/weanwise-app \
  --platform managed \
  --region YOUR_REGION \
  --allow-unauthenticated
pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('path/to/firebase-key.json')  # your downloaded private key
firebase_admin.initialize_app(cred)
db = firestore.client()
from flask import request, jsonify

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json  # Expect JSON from wearable device
    heart_rate = data.get('heart_rate')
    blood_oxygen = data.get('blood_oxygen')
    timestamp = data.get('timestamp')

    # Store data in Firestore
    db.collection('health_metrics').add({
        'heart_rate': heart_rate,
        'blood_oxygen': blood_oxygen,
        'timestamp': timestamp
    })

    return jsonify({'status': 'success'}), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    records = db.collection('health_metrics').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
    result = []
    for record in records:
        result.append(record.to_dict())
    return jsonify(result)
@app.before_request
def check_token():
    token = request.headers.get('Authorization')
    if token != 'Bearer YOUR_SECRET_KEY':
        return jsonify({'error': 'Unauthorized'}), 401
weanwise_app/ ├── app.py ├── requirements.txt ├── templates/ │ └── index.html ├── static/ │ └── chart.js (optional) └── firebase-key.json (if using Firestore)

FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
docker build -t weanwise-health .
docker run -p 5000:5000 weanwise-health
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

