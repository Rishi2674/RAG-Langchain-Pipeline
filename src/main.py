# src/main.py
import os
from pdf_loader import load_pdf
from utils import chunk_text
from rag_pipeline import build_rag_pipeline

def main():
    # Load and process PDF
    pdf_path = "../data/document.pdf"
    text = load_pdf(pdf_path)
    text_chunks = chunk_text(text, chunk_size=300)
    
    # Build RAG pipeline
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    rag_chain = build_rag_pipeline(text_chunks=text_chunks,model_name=model_name)
    
    # Define queries
    queries = [
        "What is the significance of significant figures in measurement, and how do they indicate measurement precision?",
        "Explain the principle of homogeneity in dimensions and its role in verifying the correctness of equations.",
        "What are the SI base units, and why are they internationally standardized?",
        "How is the dimensional formula used to deduce relations among physical quantities? Provide an example.",
        "Describe the rules for rounding off numbers to appropriate significant figures in scientific notation."
    ]
    
    # Process each query
    for query in queries:
        result = rag_chain({"query": query})
        print(f"Query: {query}")
        print("Generated Answer:", result["result"])
        print("Source Documents:", result["source_documents"])
        print("-" * 50)

if __name__ == "__main__":
    main()
