import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset Titanic
data = pd.read_csv('titanic.csv')

# Seleccionar las características (parámetros) y el target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

data['Sex'] = data['Sex'].replace(['female', 'male'], [0, 1])
data['Embarked'] = data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2])

# Imputar valores faltantes
mean_age = data['Age'].mean()
data['Age'].fillna(mean_age, inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

X = data[features]
y = data['Survived']

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir el modelo
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))  # Capa de entrada con 7 neuronas
model.add(Dense(2, activation='relu'))  # Capa intermedia con 2 neuronas (puede ajustarse)
model.add(Dropout(0.2))  # Dropout para prevenir sobreajuste
model.add(Dense(1, activation='sigmoid'))  # Capa de salida
model.summary()

# Compilar el modelo
opt = Adam(learning_rate=1e-2)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Configurar EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluar el modelo
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

print("Classification Report:\n", classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Curvas de aprendizaje
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# Guardar el modelo
model.save('RNA_Titanic.h5')
print("Modelo guardado como 'RNA_Titanic.h5'.")
