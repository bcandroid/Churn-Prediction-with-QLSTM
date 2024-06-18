

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from collections import deque
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import ADASYN
from tensorflow import keras

import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

import warnings
warnings.filterwarnings("ignore")

# Load the dataset from the GitHub URL
url = 'https://raw.githubusercontent.com/SulmanK/Customer-Churn-in-World-of-Warcraft/master/data/churn.csv'
df = pd.read_csv(url)
X = df[['guild', 'max_level', 'Average_Hour', 'Average_Playing_density']]
y = df['Playing_after_6_months']



adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)

print("Number of Class 1 Samples (After):", sum(y_resampled == 1))
print("Number of Class 0 Samples (After):", sum(y_resampled == 0))
print("Shape of X_resampled:", X_resampled.shape)
print("Shape of y_resampled:", y_resampled.shape)

# Split the data into training, validation, and test sets
training_data, validation_and_test_data, y_train, y_validation_and_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=10, stratify=y_resampled, shuffle=True)
validation_data, test_data, y_val, y_test = train_test_split(validation_and_test_data, y_validation_and_test, test_size=0.6, random_state=10, stratify=y_validation_and_test, shuffle=True)

col_names = ['guild', 'max_level', 'Average_Hour', 'Average_Playing_density']
features = training_data[col_names]
features_test = test_data[col_names]
features_val = validation_data[col_names]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(features)
X_test_scaled = scaler.transform(features_test)
X_val_scaled = scaler.transform(features_val)


X_train = np.expand_dims(X_train_scaled, axis=-1)
X_test = np.expand_dims(X_test_scaled, axis=-1)
X_val = np.expand_dims(X_val_scaled, axis=-1)
print(X_train.shape)  # Accessing the shape attribute correctly

# Define the LSTM model
model = models.Sequential()
model.add(layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(units=32))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the target model with the same structure
model_target = models.Sequential()
model_target.add(layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_target.add(layers.Dropout(0.2))
model_target.add(layers.LSTM(units=32))
model_target.add(layers.Dropout(0.2))
model_target.add(layers.Dense(units=1, activation='sigmoid'))
model_target.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_target.set_weights(model.get_weights())

# Q-learning parameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)
alpha = 0.5
target_update_frequency = 10

# Initialize Q-table
n_states = 10 ** X_test.shape[1]
n_actions = len(X_test)
Q = np.zeros((n_states, n_actions))

epochs = 100
cumulative_reward = 0

# Training loop
for epoch in range(epochs):
    # Train the model
    history = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    # Predictions
    predictions = np.zeros(len(X_test))
    for i in range(len(X_test)):
        X = np.array(X_test[i]).reshape((1, X_test.shape[1], 1))
        predictions[i] = model.predict(X, verbose=0)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target model
    if epoch % target_update_frequency == 0:
        model_target.set_weights(model.get_weights())

    # Q-learning action selection and updates
    state = np.random.randint(0, n_states)
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])

    # Calculate new state
    new_state = (state * 10 + action) % n_states

    # Calculate reward
    reward = np.abs(X_test[action] - predictions[action]).sum()  # Reduce reward to a single value
    memory.append((state, action, reward, new_state))
    cumulative_reward += reward

    # Update Q-table
    batch = random.sample(memory, min(len(memory), batch_size))
    for state, action, reward, new_state in batch:
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

    # Print epoch information
    print(f"Epoch {epoch + 1}, Cumulative Reward: {cumulative_reward}")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Predict on test set
y_pred = model.predict(X_test, verbose=0)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Calculate mean squared error
mse = np.mean((y_pred - y_test.values.reshape(-1, 1)) ** 2)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
print('Mean Squared Error:', mse)
!pip install shap
import shap
# Initialize SHAP
shap.initjs()

def predict_fn(x):
    return model.predict(x.reshape(x.shape[0], x.shape[1], 1))  # Reshape to match model input

# Create SHAP explainer and calculate SHAP values
explainer = shap.Explainer(predict_fn, X_test_scaled)
shap_values = explainer(X_test_scaled)


# SHAP değerlerini görselleştirme
# Bar plot
shap.plots.bar(shap_values)

# Beeswarm plot
shap.plots.beeswarm(shap_values)

# Violin plot
shap.summary_plot(shap_values, plot_type='violin')

# Waterfall plot (ilk örnek için)
shap.plots.waterfall(shap_values[0])

# Force plot (ilk örnek için)
shap.plots.force(shap_values[0])
