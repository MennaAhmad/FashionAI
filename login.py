import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from home_page import Home_page
from search_by_description import search_box
from searchlast import Search_page
from recommendations import get_user_last_n_searches, display_user_history_with_recommendations
from lastllm import analyze_fashion_store_data

# Sidebar navigation using option_menu
with st.sidebar:
    choice = option_menu(
        "FashionIQ",
        ["Home", "Search by Description", "Search by Image", "Get Insights and Recommendations"],
        icons=["house", "search", "image", "bar-chart-line"],
        menu_icon="list",
        default_index=0,
    )

# Logic to handle navigation
if choice == "Home":
    # Input email at the top of the Home page only
    user_email = st.text_input("Enter your email to personalize your experience:", placeholder="e.g., user@example.com")
    
    # Display the user's email above the content on Home page
    if user_email:
        st.write(f"**Welcome, {user_email}!**")  # Display email prominently
        
        # Fetch and display user history if it exists
        user_history = get_user_last_n_searches(user_email)
        if user_history:
            st.write("Here are your recent searches and recommendations:")
            display_user_history_with_recommendations(user_email)
        else:
            st.write("You have no search history. Explore our inventory below:")
            Home_page()  # Default home page content when no history
    else:
        st.write("Welcome! Explore our inventory below:")
        Home_page()  # Default home page content when no email provided

elif choice == "Search by Description":
    search_box()  # Functionality to search by description
elif choice == "Search by Image":
    Search_page()  # Functionality to search by image
elif choice == "Get Insights and Recommendations":
    # st.write("Insights and Recommendations page coming soon!")
    analyze_fashion_store_data()

