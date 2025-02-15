import os
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import chromadb
import ollama

# System Prompt
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. If the user greets you (e.g., says "hello," "hi," or similar), respond with a friendly greeting.
Otherwise, answer the question or provide relevant information based on the provided context.

Context will be passed as "Context:"
User question will be passed as "Question:"
Chat history will be passed as "History:" to help you understand the ongoing conversation.

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
6. As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-3 sentences. "


Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Answer should be limited to 50 words and 2-3 sentences.  do not prompt to select answers or do not formualate a stand alone question. do not ask questions in the response. 
"""

# Load and Process Documents
def process_document(file_path: str) -> list[Document]:
    """Processes a PDF file by splitting it into chunks."""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", ".", "?", "!", " "]
    )
    return text_splitter.split_documents(docs)

# ChromaDB Vector Collection
def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text:latest"
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app", embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"}
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search."""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

# Load Documents at Startup
def load_documents_to_store(directory: str):
    """Loads all PDF documents from the specified directory."""
    for file_path in os.listdir(directory):
        if file_path.endswith(".pdf"):
            full_path = os.path.join(directory, file_path)
            file_name = os.path.basename(full_path).replace(" ", "_").replace("-", "_")
            all_splits = process_document(full_path)
            add_to_vector_collection(all_splits, file_name)

# Query Collection
def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection for relevant documents."""
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

# Call LLM
def call_llm(context: str, history: list[dict], prompt: str):
    """Calls the language model with context and history to generate a response."""
    formatted_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
    )
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"History: {formatted_history}\n\nContext: {context}\n\nQuestion: {prompt}"},
        ],
    )
    generated_text = ""
    for chunk in response:
        if chunk["done"] is False:
            generated_text += chunk["message"]["content"]
        else:
            break
    print(formatted_history)
    return generated_text

# Streamlit Chat Interface
st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query Collection
    results = query_collection(prompt)
    documents = results.get("documents", [])
    response = call_llm(context=documents, history=st.session_state.messages, prompt=prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
