import streamlit as st
import torch
from generate_function import create_image_album, load_diffuser_model, album, clear_cuda_memory
import os
from PIL import Image
import zipfile
import io

def create_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, file)
                     
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

def main():
    cache = torch.cuda.memory_cached() / 1024 ** 3
    print(cache)
    if cache >= 4.3 :
        clear_cuda_memory()
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1024 ** 3:.2f} GB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    st.subheader("Please enter in your script:")
    st.write("Scene 1: [Scene description]")
    st.write("Scene 2: [Scene description]")
    input = st.text_area("")
    if st.button("Create Storyboard"):
        pipe = load_diffuser_model(device)    
        create_image_album(input_string=input, pipeline=pipe)

        
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for i, img in enumerate(album):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                img_bytes = img_buffer.getvalue()
                zipf.writestr(f"scene_{i + 1}.jpg", img_bytes)

    zip_buffer.seek(0)
    st.download_button(
        label="Download",
        data=zip_buffer,
        file_name="generated_scenes.zip",
        key="download_all_button",
    )


            
    # Using object notation

    # Using "with" notation
    with st.sidebar:
        st.title('FAQ')
        st.subheader('Github : https://github.com/LPK99/AI-Storyboard-Stable-Diffusion-')
        st.write('Developed by Duc Luu')


if __name__ == "__main__":
    main()
