# RAG App  
## Overview  
The **RAG App** is a Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents and receive AI-generated answers based on their contents. This project has two implementations:  
1. **Streamlit Version** using **LLaMA** (via Ollama) or **Gemini**.  
2. **FastAPI Version** using **AWS Bedrock**.  
The PDF files are uploaded directly within the code.  
## Features  
- **PDF Upload** (Handled within the code)  
- **Text Extraction** from PDFs  
- **Vector-Based Retrieval**  
- **LLM-Powered Responses** using LLaMA , Gemini, or AWS Bedrock  
## Tech Stack  
- **LLMs**: LLaMA (via Ollama) / Gemini / AWS Bedrock  
- **UI Framework**: Streamlit  
- **API Framework**: FastAPI  
- **Vector Database**: FAISS / ChromaDB  
## Installation  
### Prerequisites  
- **For LLaMA 2 (Ollama)**: Install [Ollama](https://ollama.com)  
 ```sh
 curl -fsSL https://ollama.com/install.sh | sh
 ```
- **For Gemini**: Get an API key from [Google AI](https://ai.google.dev/) and set it as an environment variable:  
 ```sh
 export GEMINI_API_KEY="your-api-key"
 ```
- **For AWS Bedrock**: Configure AWS credentials using the AWS CLI:  
 ```sh
 aws configure
 ```
### For Streamlit Version  
1. Install dependencies:  
  ```sh
  pip install -r requirements.txt
  ```
2. Run the Streamlit app:  
  ```sh
  streamlit run Llama_App.py
or 
  streamlit run Gemini_App.py
  ```
### For FastAPI Version (AWS Bedrock)  
1. Install dependencies:  
  ```sh
  pip install -r requirements.txt
  ```
2. Run FastAPI server:  
  ```sh
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```

