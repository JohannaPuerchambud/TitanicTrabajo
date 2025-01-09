import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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

# Construir el modelo mejorado
model = Sequential()

# Primera capa oculta
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Segunda capa oculta
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Tercera capa oculta
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())

# Capa de salida
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compilar el modelo
opt = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Configurar EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1,
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

# Cargar el modelo guardado y recompilar
loaded_model = load_model('RNA_Titanic.h5')
loaded_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Evaluar el modelo cargado
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=1)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(X_test)
y_pred_loaded = (y_pred_loaded > 0.5)

print("Classification Report (Loaded Model):\n", classification_report(y_test, y_pred_loaded))
