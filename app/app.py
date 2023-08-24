import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from generate_function import create_image_album
import os
from PIL import Image
import zipfile


def resize_image(image_path, max_width):
    img = Image.open(image_path)
    width_percent = (max_width / float(img.size[0]))
    new_height = int((float(img.size[1]) * float(width_percent)))
    resized_img = img.resize((max_width, new_height), Image.LANCZOS)
    return resized_img


def delete_images_in_folder(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not image_files:
        st.write("No images found in the folder.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        os.remove(image_path)

def display_scenes(image_folder):
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not image_files:
        return
    
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        st.image(img, use_column_width=True, caption=image_file)

@st.cache_resource
def loadmodel(device):
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

def create_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, file)
                     
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

def main():
    st.set_page_config(
        page_title="AI Storyboard",
        page_icon="media_file/ai.png",
        initial_sidebar_state="expanded"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = loadmodel(device)
    st.image('media_file/ai.png', width=100)    
    st.title("AI Storyboard")
    image_folder = "media_file/test_image"
    st.subheader("Please enter in your scriptq:")
    st.write("Scene 1: [Scene description]")
    st.write("Scene 2: [Scene description]")
    input = st.text_area("")
    if st.button("Create Storyboard"):
        delete_images_in_folder(image_folder=image_folder)
        create_image_album(input, pipeline=pipe, image_folder=image_folder)
        create_zip(image_folder, zip_path='media_file/images.zip')
        with open('media_file/images.zip', 'rb') as f:
            st.download_button('Download', f, file_name='images.zip')
    display_scenes(image_folder)
            
    # Using object notation

    # Using "with" notation
    with st.sidebar:
        st.title('FAQ')
        st.subheader('Github : https://github.com/LPK99/AI-Storyboard-Stable-Diffusion-')
        st.write('Developed by Duc Luu')


if __name__ == "__main__":
    main()
