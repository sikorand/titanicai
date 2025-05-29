
import streamlit as st
import numpy as np

# Параметри моделі (попередньо треновані вручну)
weights = np.array([[-0.745], [2.159], [-0.416], [0.136]])
bias = -0.195
mean = [0.829, 0.383, 29.651, 32.204]
std = [0.822, 0.486, 14.457, 49.693]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def preprocess_input(pclass, sex, age, fare):
    arr = np.array([pclass, sex, age, fare])
    arr_scaled = (arr - mean) / std
    return arr_scaled.reshape(1, -1)

def predict_survival(X):
    linear = np.dot(X, weights) + bias
    prob = sigmoid(linear)
    return int(prob >= 0.5), float(prob)

# Streamlit інтерфейс
st.title("🚢 Titanic Survival Prediction")
st.markdown("Введи параметри пасажира:")

pclass = st.selectbox("Pclass (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare", 0.0, 500.0, 50.0)

sex_val = 0 if sex == "male" else 1
X_input = preprocess_input(pclass, sex_val, age, fare)

if st.button("🧠 Прогнозувати"):
    survived, probability = predict_survival(X_input)
    st.success(f"Ймовірність виживання: {probability:.2%}")
    st.write("🟢 **Виживе**" if survived else "🔴 **Не виживе**")
