from fastapi import FastAPI, Form
from dotenv import load_dotenv
import boto3

# LangChain Imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM Imports
from langchain_aws import BedrockLLM
from langchain_groq import ChatGroq
from langchain_community.llms import GPT4All
from langchain_ollama.llms import OllamaLLM

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store (FAISS)
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# ---------------------------
# SELECT YOUR LLM HERE
# ---------------------------

# Option 1: AWS Bedrock (Uncomment to use)
# llm = BedrockLLM(credentials_profile_name="default", model_id="anthropic.claude-v3")

# Option 2: Groq API (Uncomment to use)
# llm = ChatGroq(api_key="your-groq-api-key", model_name="llama-3.1-8b-instant", max_tokens=1024)

# ---------------------------
# PROMPT DEFINITIONS
# ---------------------------

# Prompt for contextualizing questions based on chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are an assistant specialized in rephrasing questions. "
         "Given a chat history and the latest user question, reformulate it so it stands alone. "
         "Always return the question in English. Do NOT answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Prompt for answering user questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are an assistant for question-answering tasks. "
         "Use the retrieved context to answer the question. "
         "If you don't know, just say so. Limit your response to three sentences.\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "User question: {input}"),
    ]
)

# ---------------------------
# RAG PIPELINE
# ---------------------------

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Create a question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine retrieval and question-answering into a RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize chat history
chat_history = []

# ---------------------------
# FASTAPI ENDPOINT
# ---------------------------

@app.post("/chat/")
async def chat(query: str = Form(...)):
    """Handles user queries and returns AI-generated responses."""
    
    # Retrieve answer using RAG
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    # Store chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))
    
    return {"query": query, "response": result["answer"]}

# ---------------------------
# RUN FASTAPI SERVER
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
