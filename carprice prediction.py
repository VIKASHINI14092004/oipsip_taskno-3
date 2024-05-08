import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Data Loading and Initial Exploration
df = pd.read_csv('car data.csv', encoding='ascii')
print(df.head())
print(df.describe())
print(df.info())

# Step 2: Data Preprocessing
encoder = LabelEncoder()
df['Fuel_Type'] = encoder.fit_transform(df['Fuel_Type'])
df['Selling_Type'] = encoder.fit_transform(df['Selling_Type'])
df['Transmission'] = encoder.fit_transform(df['Transmission'])

scaler = StandardScaler()
numerical_features = ['Year', 'Present_Price', 'Kms_Driven']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())

# Step 3: Splitting Data into Training and Test Sets
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)

# Step 4: Model Training and Evaluation
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error:', mse)
