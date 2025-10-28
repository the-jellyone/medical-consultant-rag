import os
import glob
import logging
import shutil

# Vectorstores
from langchain_chroma import Chroma
# Document loader
from langchain_community.document_loaders import PyMuPDFLoader
# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
# --- Configuration ---
BOOKS_DIR = "books"
VECTORSTORE_DIR = "vectorstores"
CHUNK_SIZE = 700  # As per your plan (500-800)
CHUNK_OVERLAP = 100 # As per your plan
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_pdf(pdf_path, persist_directory):
    """Loads, splits, and builds a vectorstore for a single PDF."""
    logging.info(f"Processing: {pdf_path}")
    
    # 1. Load the document
    # PyMuPDFLoader automatically handles extraction and metadata (source, page)
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        logging.warning(f"No text extracted from {pdf_path}. Skipping.")
        return

    # 2. Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        logging.warning(f"Failed to create chunks for {pdf_path}. Skipping.")
        return

    # 3. Build vectorstore
    if os.path.exists(persist_directory):
        logging.info(f"Removing old vectorstore at: {persist_directory}")
        shutil.rmtree(persist_directory)

    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logging.info(f"Creating new vectorstore at: {persist_directory} with {len(chunks)} chunks")
    
    # Use .from_documents, which takes the Document objects directly
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    db.persist()
    logging.info(f"Successfully built and persisted vectorstore: {persist_directory}")

def main():
    """Main function to process all PDFs and build vectorstores."""
    logging.info("Starting preprocessing...")
    
    if not os.path.exists(BOOKS_DIR):
        logging.error(f"Books directory not found: {BOOKS_DIR}")
        return
        
    if not os.path.exists(VECTORSTORE_DIR):
        os.makedirs(VECTORSTORE_DIR)
        
    pdf_files = glob.glob(os.path.join(BOOKS_DIR, "*.pdf"))
    
    if not pdf_files:
        # Corrected the typo here
        logging.warning(f"No PDF files found in {BOOKS_DIR}")
        return

    for pdf_path in pdf_files:
        # Determine domain name from filename
        domain_name = os.path.basename(pdf_path).replace('.pdf', '')
        persist_directory = os.path.join(VECTORSTORE_DIR, domain_name)
        
        try:
            process_pdf(pdf_path, persist_directory)
        except Exception as e:
            logging.error(f"Failed to process {pdf_path}: {e}")
            
    logging.info("Preprocessing finished.")

if __name__ == "__main__":
    main()

