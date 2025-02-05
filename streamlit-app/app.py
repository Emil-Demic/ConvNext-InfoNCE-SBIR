import torch
import numpy as np
import streamlit as st

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from utils import find_similar_embeddings, init_model_transforms
from pillow_heif import register_heif_opener

register_heif_opener()

model, transforms = init_model_transforms()

gallery_embeddings_FSCOCO = np.load('unseen_emb.npy')
gallery_files_FSCOCO = np.load('unseen_paths.npy')

gallery_embeddings_photos = np.load('embeddings.npy')
gallery_files_photos = np.load('files.npy')

embedding_options = {
    "FSCOCO unseen": gallery_embeddings_FSCOCO,
    "My Photos": gallery_embeddings_photos,
}

files_options = {
    "FSCOCO unseen": gallery_files_FSCOCO,
    "My Photos": gallery_files_photos,
}

st.sidebar.header("Select dataset")
selected_option = st.sidebar.selectbox("Select dataset", ["My Photos", "FSCOCO unseen"])

gallery_embeddings = embedding_options[selected_option]
gallery_files = files_options[selected_option]

st.title("ConvNext-InfoNCE SBIR")

st.sidebar.header("Examples of images from selected dataset:")
ex_cols = [st.sidebar.columns(2), st.sidebar.columns(2), st.sidebar.columns(2)]

for i in range(6):
    ex_img = Image.open(gallery_files[i])
    ex_img = ImageOps.exif_transpose(ex_img)
    ex_cols[i // 2][i % 2].image(ex_img)

with st.form("Sketch"):
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="#EEEEEE",
            height=512,
            width=512,
            drawing_mode="freedraw",
            key="canvas",
        )
        generate_embedding = st.form_submit_button("Search")

if generate_embedding:
    if canvas_result.image_data is not None:
        with st.spinner(text="Searching"):
            img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
            img_tensor = transforms(img).unsqueeze(0)

            with torch.no_grad():
                query_embedding = model(img_tensor).numpy()

            top_k_indices = find_similar_embeddings(query_embedding, gallery_embeddings, top_k=10)

            st.write("Top 10 most similar images:")
            cols = st.columns(5)
            cols2 = st.columns(5)

            for i, idx in enumerate(top_k_indices):
                similar_img = Image.open(gallery_files[idx])
                similar_img = ImageOps.exif_transpose(similar_img)
                if i < 5:
                    cols[i % 5].image([similar_img], use_column_width=True, caption=str(i + 1))
                else:
                    cols2[i % 5].image(similar_img, use_column_width=True, caption=str(i + 1))
    else:
        st.write("Please draw something first!")
