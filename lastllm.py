import streamlit as st
import pickle
import google.generativeai as genai

# Configure the generative model with your API key
genai.configure(api_key='AIzaSyApNtt5rp6eFyTGikT8276nQh6ra7y9rYM')

# Load precomputed embeddings
def load_embeddings(file_path):
    """Load precomputed embeddings from a file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Function to analyze fashion store data
def analyze_fashion_store_data():
    # Paths to precomputed embedding files
    SEARCHED_EMBEDDINGS_FILE = "searched_embeddings.pkl"
    INVENTORY_EMBEDDINGS_FILE = "inventory_embeddings.pkl"

    # Load embeddings
    st.write("Loading embeddings...")
    try:
        searched_images_vectors = load_embeddings(SEARCHED_EMBEDDINGS_FILE)
        inventory_images_vectors = load_embeddings(INVENTORY_EMBEDDINGS_FILE)
        st.write("Embeddings loaded successfully!")
    except Exception as e:
        st.error(f"An error occurred while loading embeddings: {e}")
        return

    # Streamlit app layout
    st.title("Fashion Store Insights with LLM")
    st.write("Analyze static datasets of searched and inventory images to get insights.")

    # Analyze button
    if st.button("Get Insights"):
        try:
            # Summarize filenames for prompt
            searched_images_summary = ", ".join(searched_images_vectors.keys())
            inventory_images_summary = ", ".join(inventory_images_vectors.keys())

            # Define prompt for LLM
            prompt = f"""
            You are an AI assistant analyzing image data for an online fashion store to generate insights and recommendations to increase revenue and customer satisfaction.

            The following data represents:
            - Searched Images: items frequently searched by users but not always present in the inventory.
            - Inventory Images: items currently available in the store inventory.

            Analyze the patterns and provide insights such as:
            1. Common patterns in searched images vs. inventory (e.g., color, style, or trend gaps).
            2. Recommendations on missing items in the inventory that could improve sales.
            3. Potential customer trends that the store should consider in upcoming seasons.

            Below is the data in summarized form:
            Searched Images: {searched_images_summary}
            Inventory Images: {inventory_images_summary}

            Provide a professional report with clear insights and actionable recommendations.
            """

            # Call the generative AI model with the prompt
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(
                contents=[{"parts": [{"text": prompt}]}],
                generation_config={"max_output_tokens": 1000}
            )

            # Display the response
            st.header("AI-Generated Insights")
            st.write(response.text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
