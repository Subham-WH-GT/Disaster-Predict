import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load data
df = pd.read_csv('q.csv')
data = pd.DataFrame(df)

# Convert 'Origin Time' to datetime format and then to numerical timestamp
data['Origin Time'] = pd.to_datetime(data['Origin Time'].str.replace(' IST', ''), format='%Y-%m-%d %H:%M:%S')
data['Origin Time'] = data['Origin Time'].astype(int) / 10**9  # Convert to seconds since epoch

# Remove the '[ML]' suffix in 'Magnitude' and convert it to float
data['Magnitude'] = data['Magnitude'].str.extract(r'([0-9]+\.[0-9]+)').astype(float)

# Handle missing values
data = data.dropna()

# Define the scaler
scaler = MinMaxScaler()

# Normalize 'Lat', 'Long', 'Depth', and 'Origin Time'
data[['Lat', 'Long', 'Depth', 'Origin Time']] = scaler.fit_transform(data[['Lat', 'Long', 'Depth', 'Origin Time']])

# Convert 'Magnitude' into categories for classification
data['Magnitude_Category'] = pd.qcut(data['Magnitude'], q=3, labels=['Low', 'Medium', 'High'])

# Drop rows with NaNs in the target variable
data = data.dropna(subset=['Magnitude_Category'])

# Label encoding for the 'Magnitude_Category'
le = LabelEncoder()
data['Magnitude_Category'] = le.fit_transform(data['Magnitude_Category'])

# Define features and target for classification and regression
X = data[['Lat', 'Long', 'Depth', 'Origin Time']]
y_classification = data['Magnitude_Category']
y_regression = data['Magnitude']

# Train-test split
X_train, X_test, y_train_classification, y_test_classification = train_test_split(X, y_classification, test_size=0.2, random_state=0)
_, _, y_train_regression, y_test_regression = train_test_split(X, y_regression, test_size=0.2, random_state=0)

# Train the classifier with class weights to handle imbalance
clf = RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=10, random_state=0)
clf.fit(X_train, y_train_classification)

# Train the regression model to predict the actual magnitude
reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
reg.fit(X_train, y_train_regression)

# Save the models, scaler, and label encoder
joblib.dump(clf, 'random_forest_classifier.pkl')
joblib.dump(reg, 'random_forest_regressor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')



#Flood
df = pd.read_csv('q.csv')
df['Origin Time'] = pd.to_datetime(df['Origin Time'].str.replace(' IST', ''), format='%Y-%m-%d %H:%M:%S')
df['Origin Time'] = df['Origin Time'].astype('int64') // 10**9
df['Magnitude'] = df['Magnitude'].str.extract(r'([0-9]+\.[0-9]+)').astype(float)
df = df.dropna()

# Normalize earthquake features
earthquake_scaler = MinMaxScaler()
df[['Lat', 'Long', 'Depth', 'Origin Time']] = earthquake_scaler.fit_transform(df[['Lat', 'Long', 'Depth', 'Origin Time']])

# Train earthquake models
# (Place any new earthquake model code here)

# Load and prepare flood data
flood_data = pd.read_csv('run.csv')
flood_cols = ['Latitude', 'Longitude', 'Rainfall', 'Temperature', 'Humidity', 'River Discharge', 'Water Level', 'Elevation', 'Historical Floods']
flood_data = flood_data[flood_cols]

# Split and scale flood data
X_flood = flood_data.drop(['Historical Floods'], axis=1)
y_flood = flood_data['Historical Floods']
X_flood_train, X_flood_test, y_flood_train, y_flood_test = train_test_split(X_flood, y_flood, test_size=0.25, random_state=42)
flood_scaler = StandardScaler()
X_flood_train_scaled = flood_scaler.fit_transform(X_flood_train)
X_flood_test_scaled = flood_scaler.transform(X_flood_test)

# Train flood model
flood_clf = RandomForestClassifier(n_estimators=100, random_state=42)
flood_clf.fit(X_flood_train_scaled, y_flood_train)

# Save the flood model and scaler
joblib.dump(flood_clf, 'flood_classifier.pkl')
joblib.dump(flood_scaler, 'flood_scaler.pkl')