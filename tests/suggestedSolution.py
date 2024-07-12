import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

# Load and preprocess data
data = pd.read_csv("/Users/MOPOLLIKA/python_StockDL/test1.csv")
features = data.drop(['Target'], axis=1)
target = data['Target']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model
train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)
print(f'Training Loss: {train_loss}, Validation Loss: {val_loss}')
print(np.stack(X_train[len(X_train) - 10:len(X_train)]))
print(model(np.stack(X_train[len(X_train) - 10:len(X_train)])))
