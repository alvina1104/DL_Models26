import streamlit as st
import requests
from PIL import Image

def check_fashion():


    st.title("Fashion Image Classifier")
    st.header("Upload an image to identify clothing")

    api_url = "http://127.0.0.1:8000/fashion_predict/"


    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)


        st.image(image, caption="Uploaded image", width="stretch")


        if st.button("Predict"):
            files = {"file": uploaded_file.getvalue()}

            try:
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    result = response.json()


                    st.success(f"The model's prediction: {result['label']}")

                else:
                    st.error("An error occurred")

            except Exception as e:
                st.error(f"Server is unavailable: {e}")