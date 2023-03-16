import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load preprocessed data
X = pd.read_csv('preprocessed_data.csv')
y = pd.read_csv('labels.csv')

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.joblib')
