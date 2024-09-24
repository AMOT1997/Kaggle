import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import re

# Step 1: Load the datasets
train_df = pd.read_csv('C:/Users/Jeff/Documents/GitHub/Kaggle/Used Car Regression/train.csv')  # Replace with the correct path to your train.csv
test_df = pd.read_csv('C:/Users/Jeff/Documents/GitHub/Kaggle/Used Car Regression/test.csv')    # Replace with the correct path to your test.csv

# Extract horsepower from engine descriptions
def extract_horsepower(engine_str):
    match = re.search(r'(\d+(\.\d+)?)HP', engine_str)
    return float(match.group(1)) if match else None

train_df['horsepower'] = train_df['engine'].apply(extract_horsepower)
test_df['horsepower'] = test_df['engine'].apply(extract_horsepower)

# Drop the original 'engine' column
train_df.drop('engine', axis=1, inplace=True)
test_df.drop('engine', axis=1, inplace=True)

# OneHot encode categorical columns
categorical_columns = ['brand', 'fuel_type', 'transmission', 'accident', 'clean_title']
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')

# Combine, encode, then split train and test data to ensure consistent feature space
combined = pd.concat([train_df.drop('price', axis=1), test_df], ignore_index=True)
combined_encoded = pd.DataFrame(encoder.fit_transform(combined[categorical_columns]), columns=encoder.get_feature_names_out())
combined.drop(categorical_columns, axis=1, inplace=True)
combined = pd.concat([combined, combined_encoded], axis=1)

# Separate the modified combined dataset back into train and test sets
X_train = combined.iloc[:len(train_df)]
X_test = combined.iloc[len(train_df):]

# Corresponding target values
y_train = train_df['price']

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Output (you might want to add these predictions to test_df and save to CSV as shown before)
test_df['predicted_price'] = predictions
test_df.to_csv('predicted_prices.csv', index=False)
print("Predictions saved to predicted_prices.csv")