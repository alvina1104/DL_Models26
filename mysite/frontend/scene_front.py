import streamlit as st
import requests

def check_scene():

    api_url = "http://127.0.0.1:8000/predict/"

    st.title("Scene Image Classifier")
    st.write("Upload an image and the model will predict the scene type")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Predict 🔍"):

            with st.spinner("Analyzing image..."):

                files = {"file": uploaded_file.getvalue()}
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success(f"The thinks it's: {result['label'].upper()}")
                    st.write(f"Class ID: `{result['class_id']}`")

                else:
                    st.error(f"Error: {response.text}")