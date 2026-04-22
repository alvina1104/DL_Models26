import streamlit as st
import requests
from PIL import Image

def check_flowers():

    st.set_page_config(page_title="Flower Classifier", layout="centered")

    st.title("Flower Classification")
    st.write("Upload a flower image and the model will predict its class")

    api_url = "http://127.0.0.1:8000/flowers_predict/"


    uploaded_file = st.file_uploader(
        "Upload flower image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):

                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                try:
                    response = requests.post(api_url, files=files)

                    if response.status_code == 200:
                        result = response.json()

                        st.success(f"Prediction: **{result['label']}**")
                        st.write(f"Class ID: {result['class_id']}")

                    else:
                        st.error(f"Error: {response.text}")

                except Exception as e:
                    st.error(f"Server error: {e}")