# HiDream AI Assistant

A powerful AI assistant that combines image editing, image generation, and chat capabilities using HiDream models and OpenAI's GPT-4.

## Features

- **Image Editing**: Transform existing images using natural language instructions
- **Image Generation**: Create new images from text descriptions
- **Chat Interface**: Natural conversation with GPT-4 for assistance and guidance
- **Real-time Processing**: See your results instantly
- **User-friendly Interface**: Clean and intuitive Streamlit interface

## Setup

1. Clone the repository with submodules:
```bash
git clone --recursive git@github.com:gunwant11/img-edit.git
cd img-edit
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -U git+https://github.com/huggingface/diffusers.git
```

3. Set up your API keys:
   - Create a `.streamlit/secrets.toml` file
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```

4. Model Access Requirements:
   - Accept the Llama model license on HuggingFace
   - Log in to HuggingFace: `huggingface-cli login`
   - You need access to:
     - meta-llama/Llama-3.1-8B-Instruct
     - HiDream-ai/HiDream-I1-Full
     - HiDream-ai/HiDream-I1-Dev
     - HiDream-ai/HiDream-E1-Full

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. The interface provides three main functionalities:

   ### Image Editing
   - Upload an image using the file uploader
   - Type an editing instruction (e.g., "Make this image look more vintage")
   - The edited image will be displayed in the chat

   ### Image Generation
   - Type a generation prompt (e.g., "Create an image of a sunset over mountains")
   - Keywords like "create", "generate", "make", "draw" trigger image generation
   - The generated image will appear in the chat

   ### Chat Interaction
   - Have natural conversations with the AI assistant
   - Ask questions or seek guidance
   - The assistant uses GPT-4 for intelligent responses

## Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- OpenAI API key
- HuggingFace account with model access
- Internet connection for API calls and model downloads

## Technical Details

The application uses:
- HiDream-E1 for image editing
- HiDream-I1-Dev for image generation
- GPT-4 Turbo for chat functionality
- Streamlit for the user interface

## Note

- First run will download required models (may take time)
- Keep your API keys secure and never commit them to the repository
- The application maintains chat history during the session
- Image editing and generation may take a few seconds depending on your GPU 