import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from generate_function import create_image_album, album
import os
from PIL import Image
import zipfile
import io
torch.set_default_dtype(torch.float32)

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

@st.cache
def loadmodel(device):
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
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
        initial_sidebar_state="expanded"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = loadmodel(device)  
    st.title("AI Storyboard")
    st.subheader("Please enter in your scriptq:")
    st.write("Scene 1: [Scene description]")
    st.write("Scene 2: [Scene description]")
    input = st.text_area("")
    if st.button("Create Storyboard"):
        create_image_album(input_string=input, pipeline=pipe)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for i, img in enumerate(album):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="JPEG")
                img_bytes = img_buffer.getvalue()
                zipf.writestr(f"random_image_{i + 1}.jpg", img_bytes)

    zip_buffer.seek(0)
    st.download_button(
        label="Download your Storyboard",
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
