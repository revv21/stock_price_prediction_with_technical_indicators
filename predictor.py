import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import ta

# Load data
file_path = 'nq024.csv'  # Replace with your actual file path
data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
data = data[::-1]  # Ensure chronological order

# Extract 'Last' price as 'Close' for simplicity
data['Close'] = data['Last']
df_close = data['Close']

# Add technical indicators
data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
data['SMA_30'] = ta.trend.sma_indicator(data['Close'], window=30)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
bb = ta.volatility.BollingerBands(data['Close'], window=20)
data['BB_upper'] = bb.bollinger_hband()
data['BB_middle'] = bb.bollinger_mavg()
data['BB_lower'] = bb.bollinger_lband()

# Add MACD and Signal Line
macd = ta.trend.macd(data['Close'])
macd_signal = ta.trend.macd_signal(data['Close'])
data['MACD'] = macd
data['MACD_Signal'] = macd_signal

# Add ATR
data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

# Add lagged returns
data['Return'] = data['Close'].pct_change()
for lag in range(1, 6):
    data[f'Lag_{lag}'] = data['Return'].shift(lag)

# Add volume indicators (calculate SMA manually)
data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
data['Volume_SMA_30'] = data['Volume'].rolling(window=30).mean()

# Transform data
df_close_log = np.log(df_close)
df_close_tf = np.sqrt(df_close_log)
df_close_shift = df_close_tf - df_close_tf.shift()
df_close_shift.dropna(inplace=True)

# Combine technical indicators with transformed data
data['Close_shift'] = df_close_shift
features = ['SMA_10', 'SMA_30', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 
            'MACD', 'MACD_Signal', 'ATR', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 
            'Volume_SMA_10', 'Volume_SMA_30']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Normalize features
data[features] = (data[features] - data[features].mean()) / data[features].std()

# Create binary target variable (predicting the direction)
data['Target'] = (data['Close_shift'] > df_close_shift.mean()).astype(int)

# Split dataset into train and test sets
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning using GridSearchCV for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_smote, y_train_smote)
best_params_rf = grid_search_rf.best_params_
print("Best parameters for RandomForest: ", best_params_rf)

# Train the Random Forest model with the best parameters and class weights
rf_model = RandomForestClassifier(**best_params_rf, class_weight='balanced', random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Train Gradient Boosting Model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_smote, y_train_smote)

# Ensemble model with Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('gb', gb_model)
], voting='soft')
ensemble_model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Evaluate the model
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

# Plot feature importance for RandomForest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (RandomForest)")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Predicting next 4 or 8 hours
def predict_next_hours(model, data, features, imputer, n_hours=8):
    last_data = data[features].tail(1)
    last_data = pd.DataFrame(imputer.transform(last_data), columns=features)  # Ensure no NaN values in the last_data
    predictions = []
    last_close = data['Close'].iloc[-1]
    last_timestamp = data.index[-1]

    for _ in range(n_hours):
        # Get the prediction probabilities
        prediction_prob = model.predict_proba(last_data)[0]
        direction = np.argmax(prediction_prob)
        
        # Simulate the price movement with uncertainty
        if direction == 1:
            # Predicted to go up
            price_change = np.random.normal(loc=0.002, scale=0.001)  # mean positive change
        else:
            # Predicted to go down
            price_change = np.random.normal(loc=-0.002, scale=0.001)  # mean negative change

        predicted_price = last_close * (1 + price_change)
        low_bound = predicted_price * (1 - 0.002)
        high_bound = predicted_price * (1 + 0.002)

        last_timestamp += pd.Timedelta(hours=1)
        predictions.append((last_timestamp, direction, predicted_price, low_bound, high_bound))

        # Update last_data for next prediction
        last_data = last_data.shift(-1, axis=1)
        last_data.iloc[0, -1] = predicted_price  # Use the predicted price for the next iteration
        last_data = pd.DataFrame(imputer.transform(last_data), columns=features)  # Ensure no NaN values after update

        last_close = predicted_price

    return predictions

# Ensure the imputer is fitted
imputer.fit(X)

# Predict next 8 hours
predicted_data = predict_next_hours(ensemble_model, data, features, imputer, n_hours=8)

# Print last closing price and predicted prices with timestamps and directions
print("Last closing price:", data['Close'].iloc[-1])
print("Predicted prices for the next 8 hours:")
for timestamp, direction, price, low, high in predicted_data:
    direction_str = 'Up' if direction == 1 else 'Down'
    print(f"Time: {timestamp}, Direction: {direction_str}, Predicted Price: {price:.2f} ({low:.2f} - {high:.2f})")
