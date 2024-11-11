# src/rag_pipeline.py
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import chunk_text

def build_rag_pipeline(text_chunks, model_name):
    """Build the RAG pipeline using LangChain and Qwen LLM."""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Embedding Model on GPU
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False} 
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Create FAISS Vector Store
    vector_store = FAISS.from_texts(text_chunks, embedding_model)
    retriever = vector_store.as_retriever()
    
    # Load Qwen model and tokenizer on GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1,max_new_tokens = 500)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return rag_chain
