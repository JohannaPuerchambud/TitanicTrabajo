import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo entrenado
model = load_model('RNA_Titanic.h5')

# Escalador (debe ajustarse a los datos con los que fue entrenado el modelo)
scaler = MinMaxScaler(feature_range=(0, 1))

# Título de la aplicación
st.title("Predicción de Supervivencia en el Titanic")

# Descripción
st.write("""
Ingrese los datos del pasajero para predecir si sobrevivirá o no.
""")

# Formulario de entrada de datos
pclass = st.selectbox("Clase del Pasajero (Pclass)", [1, 2, 3])
sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
age = st.slider("Edad", 0, 100, 25)
sibsp = st.number_input("Número de Hermanos/Esposos a bordo (SibSp)", min_value=0, value=0)
parch = st.number_input("Número de Padres/Hijos a bordo (Parch)", min_value=0, value=0)
fare = st.number_input("Tarifa del Boleto (Fare)", min_value=0.0, value=15.0)
embarked = st.selectbox("Puerto de Embarque (Embarked)", ["C", "Q", "S"])

# Preprocesar entradas
sex = 1 if sex == "Hombre" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Normalizar los datos (ajustar si usaste otra estrategia en el entrenamiento)
input_data = scaler.fit_transform(input_data)

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(input_data)
    prediction = "Sobrevive" if prediction > 0.5 else "No Sobrevive"
    st.success(f"La predicción es: **{prediction}**")
