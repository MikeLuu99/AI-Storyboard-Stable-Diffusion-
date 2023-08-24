import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile

# Function to generate a random image
def generate_random_image():
    width, height = 300, 300
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image)

# Main Streamlit app
def main():
    st.title("Random Image Generator with Batch Download")

    num_images = st.number_input("Number of Images to Generate", min_value=1, max_value=10, value=3)

    image_list = []

    for i in range(num_images):
        random_image = generate_random_image()
        image_list.append(random_image)

        st.subheader(f"Image {i + 1}")
        st.image(random_image, use_column_width=True, caption=f"Random Image {i + 1}")

    # Create a button to download all images as a ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for i, img in enumerate(image_list):
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

if __name__ == "__main__":
    main()
