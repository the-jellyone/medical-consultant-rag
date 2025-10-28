import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

# --- Configuration ---
# Load .env file BEFORE importing local modules (like llm.py)
load_dotenv()

# Langchain imports (Modern)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from app.llm import call_llm # This now loads *after* load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Globals ---
VECTORSTORE_DIR = "vectorstores"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORES: Dict[str, Chroma] = {} # Global dict to hold loaded stores

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Medical Consultant RAG API",
    description="API for querying medical textbooks using RAG."
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class ContextDoc(BaseModel):
    content: str
    source: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    contexts: List[ContextDoc]

# --- Domain Routing Logic ---
# Simple keyword-to-domain mapping.
DOMAIN_KEYWORDS = {
    "Cardiology": ["heart", "cardiac", "hypertension", "arrhythmia", "myocardial", "infarction"],
    "EmergencyMedicine": ["sepsis", "trauma", "acute", "triage", "overdose", "cpr"],
    "InfectiousDisease": ["virus", "bacteria", "fungus", "infection", "antibiotic", "pandemic", "hiv"],
    "InternalMedicine": ["diabetes", "thyroid", "anemia", "gastro", "renal", "diagnosis"] # Default/fallback
}

def get_domain(query: str) -> str:
    """Selects the most relevant vectorstore domain based on query keywords."""
    query_lower = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            if domain in VECTORSTORES:
                logging.info(f"Routing query to domain: {domain}")
                return domain
            else:
                logging.warning(f"Keyword match for '{domain}', but vectorstore not loaded.")
    
    if "InternalMedicine" in VECTORSTORES:
        logging.info("No specific keywords matched. Routing to fallback: InternalMedicine")
        return "InternalMedicine"
    
    if VECTORSTORES:
        fallback = list(VECTORSTORES.keys())[0]
        logging.info(f"No specific keywords. Routing to first available: {fallback}")
        return fallback

    return None # No vectorstores loaded

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    """
    Load all vectorstores from the local disk.
    This assumes the Render Build Command has already downloaded and unzipped them.
    """
    logging.info("Loading vectorstores from local disk...")
    if not os.path.exists(VECTORSTORE_DIR):
        logging.error(f"CRITICAL: Vectorstore directory not found: {VECTORSTORE_DIR}")
        logging.error("This is a fatal error. The app will not work.")
        logging.error("Ensure your Render Build Command successfully downloaded and unzipped the vectorstores.zip file.")
        return

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logging.error(f"Failed to load HuggingFaceEmbeddings: {e}")
        return
    
    # Iterate over subdirectories in VECTORSTORE_DIR
    for domain in os.listdir(VECTORSTORE_DIR):
        domain_path = os.path.join(VECTORSTORE_DIR, domain)
        if os.path.isdir(domain_path):
            try:
                logging.info(f"Loading vectorstore from: {domain_path}")
                # Use the new langchain_chroma package
                db = Chroma(
                    persist_directory=domain_path,
                    embedding_function=embeddings
                )
                VECTORSTORES[domain] = db
                logging.info(f"Successfully loaded domain: {domain}")
            except Exception as e:
                logging.error(f"Failed to load vectorstore for {domain}: {e}")
                
    if not VECTORSTORES:
        logging.warning("No vectorstores were loaded.")
    else:
        logging.info(f"Loaded {len(VECTORSTORES)} vectorstores: {list(VECTORSTORES.keys())}")


# --- API Endpoints ---
@app.get("/", summary="Health Check")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "loaded_vectorstores": list(VECTORSTORES.keys())}

@app.post("/query", response_model=QueryResponse, summary="Query the RAG Pipeline")
async def query(request: QueryRequest):
    """
    Perform a RAG query:
    1. Route to the correct domain (vectorstore).
    2. Retrieve top_k relevant documents.
    3. Build a prompt with contexts.
    4. Call the LLM to get an answer.
    5. Return the answer and source contexts.
    """
    logging.info(f"Received query: {request.query} (top_k={request.top_k})")
    
    if not VECTORSTORES:
        raise HTTPException(status_code=503, detail="No vectorstores are loaded. Service is unavailable.")

    # 1. Domain Routing
    domain = get_domain(request.query)
    if not domain:
        raise HTTPException(status_code=404, detail="Could not find a relevant domain for the query.")
    
    vectorstore = VECTORSTORES[domain]

    # 2. Retrieve documents
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
        # Use .invoke() which is the new standard
        retrieved_docs = retriever.invoke(request.query)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents from vectorstore.")

    if not retrieved_docs:
        logging.warning("No relevant documents found.")
        return QueryResponse(answer="I could not find any relevant information in the medical texts for your query.", contexts=[])

    # 3. Build Prompt
    context_str = ""
    response_contexts = []
    for i, doc in enumerate(retrieved_docs):
        page_num = doc.metadata.get('page', 'N/A')
        source_file = doc.metadata.get('source', 'N/A').split('/')[-1] # Get just the filename
        
        context_str += f"- Snippet {i+1} (from {source_file}, page {page_num}):\n"
        context_str += f"{doc.page_content}\n\n"
        
        response_contexts.append(ContextDoc(
            content=doc.page_content,
            source=source_file,
            page=page_num
        ))

    # --- UPDATED PROMPT for Ragas Optimization ---
    prompt_template = f"""
**Strict Instructions:**
1.  You are an expert medical assistant.
2.  Your task is to answer the 'QUESTION' **using ONLY the provided 'CONTEXTS'**.
3.  Do not use any external knowledge.
4.  Answer CONCISELY (max 2-3 sentences).
5.  If the 'CONTEXTS' do not contain the answer, you **MUST** state: "Based on the provided contexts, I cannot find the answer."
6.  Do not add any information that is not in the contexts.

**CONTEXTS:**
{context_str}
**QUESTION:** {request.query}

**Answer:**"""

    logging.info("Sending prompt to LLM...")

    # 4. Call LLM
    answer = call_llm(prompt_template)

    # 5. Return response
    return QueryResponse(answer=answer, contexts=response_contexts)

# --- Main execution ---
if __name__ == "__main__":
    # This block is for direct execution (e.g., `python app/main.py`)
    # Uvicorn will run this in production from your start command
    # Use port 8000 for local development consistency
    import uvicorn
    logging.info("Starting FastAPI server locally on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

