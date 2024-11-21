import os
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle as pkl
from ultralytics import YOLO
from tensorflow.keras.applications import ResNet50
from io import BytesIO
import random
import base64

# Load required files
def load_gallery():
    try:
        gallery_embeddings = pkl.load(open('gallery_features.pkl', 'rb'))
        gallery_file_names = pkl.load(open('galleryfilenames.pkl', 'rb'))
        return gallery_embeddings, gallery_file_names
    except Exception as e:
        st.error(f"Error loading gallery data: {e}")
        return None, None

# Initialize models
def initialize_models():
    try:
        yolo = YOLO("best_yolo_model.pt")
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
        feature_model = Sequential([base_model, GlobalAveragePooling2D()])
        
        for layer in base_model.layers:
            layer.trainable = False  # Freeze layers for inference
        return yolo, feature_model
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

# CLIP Model Setup
@st.cache_resource
def initialize_clip():
    """Load CLIP model and preprocessor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)
    return model, preprocess, device

clip_model, preprocess, device = initialize_clip()

# Encode image
def encode_image(image_path):
    """Generate embeddings for an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        embedding = clip_model.encode_image(image_input).detach().cpu().numpy()
        return embedding
    except Exception as e:
        st.warning(f"Error encoding image {image_path}: {e}")
        return None

# Encode text
def encode_text(description):
    """Generate embeddings for a text description."""
    try:
        text_input = clip.tokenize([description]).to(device)
        embedding = clip_model.encode_text(text_input).detach().cpu().numpy()
        return embedding
    except Exception as e:
        st.warning(f"Error encoding text '{description}': {e}")
        return None

# Load images and compute embeddings
@st.cache_resource
def load_images_from_folder(folder_path):
    """Generate and store image embeddings."""
    embeddings = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(("jpg", "jpeg", "png")):
            image_path = os.path.join(folder_path, filename)
            embedding = encode_image(image_path)
            if embedding is not None:
                embeddings[filename] = embedding
    return embeddings


# Search inventory
def search_inventory(description, image_embeddings, top_k=5):
    """Search for top-k similar items."""
    text_embedding = encode_text(description)
    if text_embedding is None:
        return []
    
    text_embedding = text_embedding.reshape(1, -1)
    scores = {
        filename: cosine_similarity(embedding.reshape(1, -1), text_embedding)[0][0]
        for filename, embedding in image_embeddings.items()
    }
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results

# Display products in a card layout with hover effects
def display_similar_products(results, image_folder):
    """Display search results with product cards."""
    st.markdown("### Top Matches")
    if not results:
        st.write("No matches found.")
        return

    cols = st.columns(min(3, len(results)))  # Adjust columns dynamically based on results
    for idx, (filename, score) in enumerate(results):
        image_path = os.path.join(image_folder, filename)
        
        # Generate a random price for the product
        price = round(random.uniform(10, 500), 2)  # Generate a random price between $10 and $500

        try:
            image = Image.open(image_path)
            with cols[idx % len(cols)]:  # Distribute products evenly across columns
                # Create a styled card with hover effects
                card_html = f"""
                <style>
                    .product-card {{
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        background-color: #f9f9f9;
                        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
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
                st.markdown(card_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not display image {filename}: {e}")

def image_to_base64(image):
    """Convert image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def search_box():


    inventory_folder = "inventory2"  # Path to inventory images

    # Load inventory data
    image_embeddings = load_images_from_folder(inventory_folder)
    if not image_embeddings:
        st.error("No images found in the inventory.")
        return

    # User Input
    user_input = st.text_input(label="", placeholder="Describe the product you're looking for...")
    if user_input:
        with st.spinner("Searching inventory..."):
            results = search_inventory(user_input, image_embeddings, top_k=5)
        
        # Display results
        display_similar_products(results, inventory_folder)

# if __name__ == "__main__":
#     search_box()
