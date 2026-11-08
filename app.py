import streamlit as st
from PIL import Image
from ocr_engine import load_ocr_model, recognise_text
from page_splitter import split_page
import numpy as np

@st.cache_resource
def get_model():
    return load_ocr_model()

st.title("Handwriting to Text")
st.write("Upload one or more handwriting images to convert them into text.")
upload_type = st.radio("What are you uploading?", ["Line images", "Full-page images"], horizontal=True)
uploaded_files = st.file_uploader("Drag images here", type = ["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    convertbutton_clicked = st.button("Convert to text")
    st.success(f"{len(uploaded_files)} image(s) ready.")
    for uploaded_file in uploaded_files:
        with Image.open(uploaded_file) as image:
            st.image(image)
    if convertbutton_clicked:
        for uploaded_file in uploaded_files:
            processor, model, device = get_model()
            if upload_type == "Line images":
                with Image.open(uploaded_file) as image:
                    model_output = recognise_text(image, processor, model, device)
                    with st.container(border=True):
                        st.subheader(uploaded_file.name)
                        st.caption("Recognised text")
                        st.code(model_output, language=None)
            else:
                with Image.open(uploaded_file) as page_image:
                    bandw_image, row_ink_counts, line_ranges = split_page(page_image)
                    recognised_lines = []
                    page_width, page_height = page_image.size
                    for start_row, end_row in line_ranges:
                        padding = 40
                        crop_start = max(0, start_row - padding)
                        crop_end = min(page_height, end_row + 1 + padding)
                        line_image = page_image.crop((0, crop_start, page_width, crop_end))
                        st.image(line_image, caption=f"Detected line: rows {start_row}-{end_row}")
                        model_output_linetext = recognise_text(line_image, processor, model, device)
                        recognised_lines.append(model_output_linetext)
                        with st.container(border=True):
                            st.subheader(uploaded_file.name)
                            st.caption("Recognised page text")
                            st.code(model_output_linetext)
                            paragraph = " ".join(recognised_lines)
                    st.code(paragraph, language=None)
                
else:
    st.info("Choose one or more images to begin.")