import streamlit as st
import requests
from PIL import Image


def check_mnist():


    st.title("Приложение для распознавания изображений")
    api_url = "http://127.0.0.1:8000/predict/"


    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)


        st.image(image, caption="Загруженное изображение", width="stretch")


        if st.button("Предсказать"):
            files = {"file": uploaded_file.getvalue()}

            try:
                response = requests.post(
                    api_url,files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Результат: {result['Answer']}")
                else:
                    st.error("Произошла ошибка 😢")

            except Exception as e:
                st.error(f"Сервер недоступен: {e}")