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
from chromadb.config import Settings


# # Initialize Chroma client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# inventory_collection = chroma_client.get_or_create_collection("fashion_inventory")
# Initialize Chroma client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",  # Path to the existing database
    settings=Settings()  # Default settings
)

# Load the existing collection
inventory_collection = chroma_client.get_or_create_collection("fashion_inventory")


# Define the EfficientNet model for embedding generation
model = models.efficientnet_b0(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get image embedding
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().numpy()

def save_user_history(user_email, image_path, embedding):
    user_file = f"user_histories/{user_email}.pkl"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(user_file), exist_ok=True)

    entry = {"image_path": image_path, "embedding": embedding.tolist()}

    if os.path.exists(user_file):
        with open(user_file, "rb") as file:
            history = pickle.load(file)
    else:
        history = []

    history.append(entry)

    with open(user_file, "wb") as file:
        pickle.dump(history, file)

# Retrieve user history
def get_user_last_n_searches(user_email, n=5):
    user_file = f"user_histories/{user_email}.pkl"
    if not os.path.exists(user_file):
        return []
    with open(user_file, "rb") as file:
        history = pickle.load(file)
    return history[-n:]

def recommend_for_last_n_history_images(user_email, n=5, top_n=4):
    last_searches = get_user_last_n_searches(user_email, n=n)
    if not last_searches:
        return []

    recommendations = []
    for entry in last_searches:
        results = inventory_collection.query(
            query_embeddings=[entry["embedding"]],
            n_results=top_n
        )
        
        # Debug: print the query result
        # st.write(f"Query result for image {entry['image_path']}: {results}")

        # Ensure results contain documents and distances
        recommended_images = [img for sublist in results.get("documents", []) for img in sublist]
        distances = [dist for sublist in results.get("distances", []) for dist in sublist]
        
        recommendations.append({
            "query_image": entry["image_path"],
            "recommended_images": recommended_images,
            "distances": distances
        })
    
    return recommendations

# Function to display user history with recommendations in tabs
def display_user_history_with_recommendations(user_email, n=5, top_n=4):
    """
    Fetch and display the last `n` searches of a user along with their recommendations.
    Each history entry and its recommendations are displayed in separate tabs.
    """
    # Retrieve last n searches
    last_searches = get_user_last_n_searches(user_email, n=n)
    if not last_searches:
        st.warning("No search history found.")
        return

    # Fetch recommendations for the last n searches
    recommendations = recommend_for_last_n_history_images(user_email, n=n, top_n=top_n)
    if not recommendations:
        st.warning("No recommendations found.")
        return

    # Enhance UI with CSS
    st.markdown("""
        <style>
            .product-card {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                background-color: #f9f9f9;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
                margin-bottom: 20px;
            }
            .product-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }
            .product-card img {
                width: 100%;
                border-radius: 8px;
            }
            .price {
                font-size: 20px;
                font-weight: bold;
                color: #e74c3c;
                margin-top: 10px;
            }
            .add-to-cart {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 10px;
            }
            .add-to-cart:hover {
                background-color: #2ecc71;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create tabs for each history entry
    history_tabs = st.tabs([f"History {idx + 1}" for idx in range(len(recommendations))])

    for idx, (tab, rec) in enumerate(zip(history_tabs, recommendations)):
        with tab:
            recommended_image_paths = rec["recommended_images"]
            distances = rec["distances"]

            if not recommended_image_paths:
                st.warning(f"No recommendations found for History {idx + 1}.")
                continue

            # Create three columns for displaying recommendations
            cols = st.columns(3)  # Three columns
            for i, (image_path, distance) in enumerate(zip(recommended_image_paths, distances)):
                if not os.path.exists(image_path):
                    st.error(f"Image not found: {image_path}")
                    continue

                # Assign product card to one of the three columns
                target_col = cols[i % 3]
                with target_col:
                    try:
                        # Display product card
                        card_html = f"""
                        <div class="product-card">
                            <img src="data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode()}" alt="Recommendation {i + 1}">
                            <p><strong>Recommendation {i + 1}</strong></p>
                            <p class="price">Distance: {distance:.2f}</p>
                            <button class="add-to-cart" onclick="alert('Added to cart')">Add to Cart</button>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display image {image_path}: {e}")
