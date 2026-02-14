import streamlit as st
import pickle
from PIL import Image

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

# Title
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements to predict the Iris species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Prediction button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    labels = ["Setosa", "Versicolor", "Virginica"]
    flower = labels[prediction[0]]

    st.success(f"🌼 Predicted Flower: **{flower}**")

    # Image mapping
    image_paths = {
        "Setosa": "images/setosa.jpg",
        "Versicolor": "images/versicolor.jpg",
        "Virginica": "images/virginica.jpg"
    }

    image = Image.open(image_paths[flower])
    st.image(image, caption=flower, use_column_width=True)
