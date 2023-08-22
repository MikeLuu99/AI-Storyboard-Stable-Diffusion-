import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from torch import inference_mode
from generate_function import create_image_album
import os
from PIL import Image
import zipfile
from pathlib import Path

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
        st.write("No images found in the folder.")
        return
    
    st.write("Images in folder:")
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        st.image(img, use_column_width=True, caption=image_file)

@st.cache_resource
def loadmodel():
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def string_to_dict(input_string):
    scenes = {}
    scene_lines = input_string.split('\n')

    for line in scene_lines:
        line = line.strip()
        if line:
            scene_parts = line.split(':', 1)
            if len(scene_parts) == 2:
                scene_num = scene_parts[0].strip()
                description = scene_parts[1].strip()
                scenes[scene_num] = description
    return scenes

def zip_images_in_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(foldername, filename)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

def main():
    pipe = loadmodel()    
    st.title("AI Storyboard")
    image_folder = r"C:\Users\trung\Documents\AI_Storyboard\test_image"
    st.write("Please enter in the format:")
    st.write("Scene 1: [Scene description]")
    input = st.text_area("Enter your script here")
    if st.button("Submit"):
        delete_images_in_folder(image_folder=image_folder)
        create_image_album(input, pipeline=pipe)
        display_scenes(image_folder=image_folder)
        if os.path.exists(image_folder) and os.path.isdir(image_folder):
            download_path = Path.home() / "Downloads"
            zip_output_path = os.path.join(download_path, "downloaded_images.zip")
            st.download_button(
                label="Download",
                data=open(zip_output_path, 'rb').read(),
                file_name="downloaded_images.zip"
            )
        else:
            st.error("Invalid folder path. Please provide a valid path.")
     # Get a list of all image files in the folder
 # Get a list of all image files in the folder
    #create_image_album(input)

if __name__ == "__main__":
    main()
