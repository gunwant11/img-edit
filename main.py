# from openai import OpenAI
import streamlit as st
import torch
from PIL import Image
from model_setup import setup_models
from image_inference import process_image_edit, process_image_generation, is_image_generation_prompt
from openai import OpenAI

# Custom CSS to hide file uploader dropzone instructions
st.markdown("""
<style>
/* Hide file uploader dropzone instructions */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}

/* Make the file uploader more compact */
.st-emotion-cache-u8hs99, .st-emotion-cache-1erivf3, .st-bf {
    padding: 0 !important;
    margin: 0 !important;
}

/* Reduce the file uploader box size */
.st-emotion-cache-1vbkxwb, .st-emotion-cache-zq5wmm {
    min-height: 60px !important;
    max-height: 60px !important;
    padding: 5px !important;
}

/* Make columns tight and compact */
.st-emotion-cache-16txtl3, .st-emotion-cache-10y9jn9 {
    padding: 0 !important;
    gap: 5px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("HiDream AI Assistant")

# Initialize models
@st.cache_resource
def load_models():
    # Load HiDream models
    edit_pipe, gen_pipe, transformer, reload_keys = setup_models()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    return edit_pipe, gen_pipe, transformer, reload_keys, client

# Initialize session state
if "edit_pipe" not in st.session_state:
    st.session_state.edit_pipe, st.session_state.gen_pipe, st.session_state.transformer, st.session_state.reload_keys, st.session_state.chat_client = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

# Main chat container
chat_container = st.container()

# Display chat messages from history on app rerun
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message and message["image"] is not None:
                st.image(message["image"])

# Input area at the bottom
st.write("---")

# Create a layout with uploader and preview side by side
uploader_col, preview_col, spacer_col = st.columns([1, 1, 4])

# File uploader in first column
with uploader_col:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="uploader")
    if uploaded_file is not None:
        # Convert uploaded file to PIL Image and store in session state
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
    else:
        st.session_state.current_image = None

# Image preview in second column
with preview_col:
    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, width=80)

# Chat input below the image uploader
if prompt := st.chat_input("Chat with me or ask me to edit/create images..."):
    # Add user message to chat history
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
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
            if st.session_state.current_image is not None:
                st.image(st.session_state.current_image)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Check if it's an image generation/editing request
            if is_image_generation_prompt(prompt):
                with st.spinner("Generating new image..."):
                    generated_image = process_image_generation(
                        st.session_state.gen_pipe,
                        prompt=prompt
                    )
                    response = "Here's the image I generated based on your request."
                    st.markdown(response)
                    st.image(generated_image)
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "image": generated_image
                    }
            
            elif st.session_state.current_image is not None:
                with st.spinner("Editing your image..."):
                    edited_image = process_image_edit(
                        st.session_state.edit_pipe,
                        image=st.session_state.current_image,
                        prompt=prompt,
                        reload_keys=st.session_state.reload_keys
                    )
                    response = "Here's your edited image based on the instruction."
                    st.markdown(response)
                    st.image(edited_image)
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "image": edited_image
                    }
            
            else:
                # Regular chat response using OpenAI
                with st.spinner("Thinking..."):
                    # Convert message history to OpenAI format
                    messages = [
                        {"role": "system", "content": "You are a helpful AI assistant that can chat and help with image editing tasks."}
                    ]
                    
                    # Add previous messages from history
                    for msg in st.session_state.messages:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    # Get response from OpenAI
                    chat_response = st.session_state.chat_client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500,
                    )
                    
                    response = chat_response.choices[0].message.content
                    st.markdown(response)
                    assistant_message = {
                        "role": "assistant",
                        "content": response
                    }
            
            # Add assistant response to chat history
            st.session_state.messages.append(assistant_message)
    
    # Rerun to update the UI
    st.rerun()
