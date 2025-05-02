# HiDream-E1 Image Editor

A streamlined Streamlit interface for the HiDream-E1 image editing model, allowing users to transform images based on natural language instructions.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a CUDA-capable GPU with at least 16GB of VRAM.

3. You need to have access to the following models:
   - meta-llama/Llama-3.1-8B-Instruct
   - HiDream-ai/HiDream-I1-Full
   - HiDream-ai/HiDream-E1-Full

   You'll need to:
   - Accept the Llama model license on Hugging Face
   - Log in to Hugging Face using `huggingface-cli login`

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The application will start a web interface in your default browser. If it doesn't open automatically, you'll see a URL in the terminal that you can copy and paste into your browser.

3. In the interface:
   - Upload an image you want to edit using the file uploader
   - Enter your editing instruction in the text area (e.g., "Convert the image into a Ghibli style")
   - Click "Generate" to process the image

4. The processed image will appear in the output section on the right.

## Features

- Natural language image editing
- Automatic instruction refinement
- High-quality image generation
- User-friendly Streamlit interface
- Real-time progress updates
- Helpful tips and information in the sidebar

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Hugging Face account with access to required models
- Internet connection for model downloads

## Note

The first run will download the required models, which may take some time depending on your internet connection. The app will show a loading spinner while processing images. 