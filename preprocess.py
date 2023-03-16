import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data.csv')

# Preprocess data
X = data.drop('target', axis=1)
y = data['target']

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save preprocessed data
X.to_csv('preprocessed_data.csv', index=False)
y.to_csv('labels.csv', index=False)


cat train.py
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
