# from openai import OpenAI
import streamlit as st
import base64
from io import BytesIO

st.title("HiDream-E1 Chat")

# Initialize OpenAI client
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize current image
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display image if present
        if "image" in message and message["image"] is not None:
            st.image(message["image"])

# Add image upload functionality to sidebar
st.sidebar.title("Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_file is not None:
    # Store the image in session state
    st.session_state.current_image = uploaded_file.getvalue()
    st.sidebar.success("Image ready to send!")
else:
    st.session_state.current_image = None

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history with image if present
    user_message = {
        "role": "user", 
        "content": prompt,
    }
    
    # Add image to message if one is uploaded
    if st.session_state.current_image is not None:
        user_message["image"] = st.session_state.current_image
    
    # Add to history
    st.session_state.messages.append(user_message)
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = "This is a test response"
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": response
        }
        
        # If user sent an image, include it in the response
        if st.session_state.current_image is not None:
            assistant_message["image"] = st.session_state.current_image
            st.markdown(f"{response} I've received your image!")
            st.image(st.session_state.current_image)
            
            # Clear the current image after sending
            st.session_state.current_image = None
        else:
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append(assistant_message)
        
        # Rerun to update the sidebar (clear the upload field)
        st.rerun()
