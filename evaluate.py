import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load preprocessed data and labels
X = pd.read_csv('preprocessed_data.csv')
y = pd.read_csv('labels.csv')

# Load model
model = joblib.load('model.joblib')

# Evaluate model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Print accuracy
print('Accuracy:', accuracy)
