import os
import random
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

import streamlit as st
import os
import torch
import pickle
from torchvision import transforms, models
from PIL import Image
import chromadb
from chromadb.config import Settings
import matplotlib.pyplot as plt

# Function to convert image to base64
def image_to_base64(image):
    """Convert image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def Home_page():
    # Get the path to the image folder
    image_folder = "inventory2"
    image_width = 200  # Set desired width for images

    # Create a layout with 3 columns
    col1, col2, col3 = st.columns(3)

    # Loop through the images and display them in the columns as product cards
    for i, filename in enumerate(os.listdir(image_folder)):
        if i >= 30:  # Stop after processing 30 images
            break
        
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image extensions
            image_path = os.path.join(image_folder, filename)

            # Generate random price
            price = round(random.uniform(10, 500), 2)  # Random price between $10 and $500

            # Open image and display in the right column based on index
            try:
                image = Image.open(image_path)
                card_html = f"""
                <style>
                    .product-card {{
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        background-color: #f9f9f9;
                        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
                        margin-bottom: 20px;  /* Added space between rows */
                    }}
                    .product-card:hover {{
                        transform: translateY(-10px);
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                    }}
                    .product-card img {{
                        width: 100%;
                        border-radius: 8px;
                    }}
                    .price {{
                        font-size: 20px;
                        font-weight: bold;
                        color: #e74c3c;
                        margin-top: 10px;
                    }}
                    .add-to-cart {{
                        background-color: #27ae60;
                        color: white;
                        border: none;
                        padding: 10px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin-top: 10px;
                    }}
                    .add-to-cart:hover {{
                        background-color: #2ecc71;
                    }}
                </style>
                <div class="product-card">
                    <img src="data:image/jpeg;base64,{image_to_base64(image)}" alt="{filename}">
                    <p><strong>{filename}</strong></p>
                    <p class="price">${price}</p>
                    <button class="add-to-cart" onclick="alert('Added to cart')">Add to Cart</button>
                </div>
                """
                # Distribute products evenly across the columns
                if i % 3 == 0:
                    with col1:
                        st.markdown(card_html, unsafe_allow_html=True)
                elif i % 3 == 1:
                    with col2:
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    with col3:
                        st.markdown(card_html, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Could not display image {filename}: {e}")

# # Run Home_page when executed
# if __name__ == "__main__":
#     Home_page()
