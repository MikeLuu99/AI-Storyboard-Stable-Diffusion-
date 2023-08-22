import os
import zipfile
import streamlit as st

def create_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

def main():
    st.title("Image Zipper")
    st.write("Select a folder containing images and create a zip file.")

    folder_path = st.sidebar.selectbox("Select a folder:", os.listdir("."))
    zip_path = st.text_input("Enter the zip file name:", "images.zip")
    zip_button = st.button("Create Zip")

    if zip_button:
        create_zip(folder_path, zip_path)
        st.success(f"Zip file '{zip_path}' created!")

if __name__ == "__main__":
    main()
