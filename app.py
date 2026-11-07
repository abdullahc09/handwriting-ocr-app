import streamlit as st
from PIL import Image
from ocr_engine import load_ocr_model, recognise_text

@st.cache_resource
def get_model():
    return load_ocr_model()

st.title("Handwriting to Text")
st.write("Upload one or more handwriting images to convert them into text.")

uploaded_files = st.file_uploader(
    "Drag images here",
    type = ["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    convertbutton_clicked = st.button("Convert to text")
    st.success(f"{len(uploaded_files)} image(s) ready.")
    for uploaded_file in uploaded_files:
        with Image.open(uploaded_file) as image:
            st.image(image)
    if convertbutton_clicked:
        for uploaded_file in uploaded_files:
            processor, model, device = get_model()
            with Image.open(uploaded_file) as image:
                model_output = recognise_text(image, processor, model, device)
                with st.container(border=True):
                    st.subheader(uploaded_file.name)
                    st.caption("Recognised text")
                    st.code(model_output, language=None)

else:
    st.info("Choose one or more images to begin.")