import time
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import streamlit as st
from recommendations import get_embedding, recommend_for_last_n_history_images 
import os
import base64
import torch
from torchvision import transforms, models
import chromadb
from chromadb.config import Settings
from chromadb.config import Settings
from recommendations import save_user_history
from recommendations import recommend_for_last_n_history_images
from recommendations import display_user_history_with_recommendations

# # Initialize YOLO Model
yolo_model = YOLO("best_yolo_model.pt")


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


# Function to process and display user-uploaded image with YOLO segmentation and recommendations
def process_image_with_yolo(upload_path, email):
    # Run YOLO model to detect objects in the uploaded image
    results = yolo_model(upload_path)
    image_cv = cv2.imread(upload_path)
    st.success("Image is ready for processing!")
    st.image(upload_path, width=250)

    # Ensure cropped images directory exists
    cropped_folder = "cropped_images"
    os.makedirs(cropped_folder, exist_ok=True)

    # Process YOLO results to crop images
    cropped_images = []
    segmented_items = []
    for r in results:
        boxes = r.boxes.xyxy  # YOLO bounding boxes in [x1, y1, x2, y2] format
        classes = r.boxes.cls  # Class indices for detected objects
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_image = image_cv[y1:y2, x1:x2]
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_images.append(cropped_image_rgb)
            cropped_image_path = os.path.join(cropped_folder, f"cropped_image_{i+1}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

            # Get class label for this detection
            class_idx = int(classes[i])
            class_name = yolo_model.names[class_idx] if yolo_model.names else f"Item {i+1}"

            # Extract features for the cropped image
            user_embedding = get_embedding(cropped_image_path)
            
            # Define timestamp
            timestamp = int(time.time())  # Get the current time in seconds

            # Instead of `cropped_image.name`, use a unique filename based on the index
            unique_filename = f"upload/{timestamp}_cropped_image_{i+1}.jpg"
            
                        # Generate embedding
            # embedding = get_embedding(unique_filename, model)
            
            # Save search history
            save_user_history(email, unique_filename, user_embedding)
            st.success("Search history saved!")

            # Generate and display recommendations
            recommendations = recommend_for_last_n_history_images(email, n=5, top_n=4)
            if recommendations:
                st.subheader("Recommendations based on your search history:")
                # display_user_history_with_recommendations(email, n=5, top_n=4)
            else:
                st.warning("No recommendations available yet.")

            # Query Chroma for similar items
            similar_items = inventory_collection.query(
                query_embeddings=[user_embedding.tolist()],
                n_results=4
            )
            # st.write("Debug similar_items:", similar_items)
            similar_products = []
            if similar_items and "documents" in similar_items:
                for i, doc in enumerate(similar_items["documents"][0]):
                    doc_path = os.path.normpath(doc)
                    if os.path.exists(doc_path):
                        similar_products.append({
                            "name": os.path.basename(doc_path),
                            "price": f"${np.random.randint(10, 100)}.99",
                            "image": doc_path,
                            "category": similar_items["metadatas"][0][i].get("label", "Unknown") if "metadatas" in similar_items else "Unknown"
                        })
                    else:
                        st.warning(f"Image not found: {doc_path}")
            else:
                st.error("No similar items found or structure mismatch.")

                        
            # else:
            #     # Handle case where the structure is not as expected or is empty
            #     similar_products = []

            # Append segmented items with relevant details
            segmented_items.append(
                {
                    "name": class_name,
                    "image": cropped_image_path,
                    "similar_products": similar_products
                }
            )

    # Display segmented items and similar products in tabs
    if segmented_items:
        tabs = st.tabs([item["name"] for item in segmented_items])

        # Add custom styling for the products
        st.markdown("""
            <style>
                .product-card {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 16px;
                    text-align: center;
                    background-color: #f9f9f9;
                    margin-bottom: 16px;
                }
                .product-image {
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                }
                .product-name {
                    font-weight: bold;
                    margin-top: 10px;
                }
                .product-price {
                    color: #2dbe60;
                    font-size: 18px;
                    margin-top: 5px;
                }
                .add-to-cart-btn {
                    margin-top: 10px;
                    padding: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .add-to-cart-btn:hover {
                    background-color: #45a049;
                }
            </style>
        """, unsafe_allow_html=True)

        # Display each segmented item with similar product recommendations in tabs
        for i, item in enumerate(segmented_items):
            with tabs[i]:
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.image(item["image"], caption=f"Segmented {item['name']}", width=100)
                with col2:
                    st.subheader(f"Similar Products for {item['name']}")

                # Display similar products
                col1, col2 = st.columns(2)
                for idx, product in enumerate(item["similar_products"]):
                    target_col = col1 if idx % 2 == 0 else col2
                    with target_col:
                        product_name = product["name"]
                        product_image_path = product["image"]

                        # Convert product image to Base64 for inline display
                        with open(product_image_path, "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode()

                        st.markdown(f"""
                            <div class="product-card">
                                <img src="data:image/jpeg;base64,{encoded_image}" class="product-image" alt="{product_name}">
                                <div class="product-name">{product_name}</div>
                                <div class="product-price">{product['price']}</div>
                                <button class="add-to-cart-btn" onclick="window.location.href='#'">Add to Cart</button>
                            </div>
                        """, unsafe_allow_html=True)

# Streamlit Search Page
def Search_page():
    st.subheader("Search by Image")

    # Email input for user identification
    email = st.text_input("Enter your email:", key="user_email")
    if not email:
        st.warning("Please enter your email to proceed.")
        return

    # Radio button for choosing between upload or camera
    image_option = st.radio(
        "Choose how to get the image:", 
        ("Upload Image", "Use Camera"),
        key="image_option_radio"
    )

    # Handle image upload
    if image_option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Generate a unique filename using the current timestamp
            timestamp = int(time.time())  # Get the current time in seconds
            unique_filename = f"upload/{timestamp}_{uploaded_image.name}"

            # Save uploaded file with a unique filename
            os.makedirs("upload", exist_ok=True)
            with open(unique_filename, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Process the image with YOLO and get recommendations
            process_image_with_yolo(unique_filename, email)
            

    # Handle image capture using camera
    elif image_option == "Use Camera":
        captured_image = st.camera_input("Capture an Image")
        if captured_image is not None:
            # Generate a unique filename using the current timestamp
            timestamp = int(time.time())  # Get the current time in seconds
            unique_filename = f"upload/captured_image_{timestamp}.jpg"

            # Save captured image with a unique filename
            os.makedirs("upload", exist_ok=True)
            with open(unique_filename, "wb") as f:
                f.write(captured_image.getbuffer())

            # Process the image with YOLO and get recommendations
            process_image_with_yolo(unique_filename, email)
