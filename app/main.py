import os
import io
import zipfile
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
import uvicorn

# --- LOAD ENV VARS FIRST ---
load_dotenv()

# --- CONFIGURATION ---
VECTORSTORE_DIR = "vectorstores"
VECTORSTORE_ZIP_URL = "https://drive.google.com/uc?export=download&id=1tehwg4o8WdvgtBFnbjGlkijCzN-QRm5Y"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORES: Dict[str, "Chroma"] = {}  # Will hold loaded vectorstores

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ensure Vectorstore exists ---
def ensure_vectorstore_ready():
    """Download and extract vectorstores.zip if not already present."""
    if os.path.exists(VECTORSTORE_DIR):
        print("✅ Vectorstores folder already exists. Skipping download.")
        return
    
    print("⬇️ Downloading vectorstores.zip from Google Drive...")
    response = requests.get(VECTORSTORE_ZIP_URL)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download vectorstores.zip, status code: {response.status_code}")
    
    print("📦 Extracting vectorstores.zip ...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(".")
    print("✅ Vectorstores extracted successfully.")

# Call immediately to ensure vectorstores are present before app startup
ensure_vectorstore_ready()

# --- LANGCHAIN IMPORTS ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Local imports ---
from app.llm import call_llm

# --- FastAPI App ---
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

# --- Domain Keywords ---
DOMAIN_KEYWORDS = {
    "Cardiology": ["heart", "cardiac", "hypertension", "arrhythmia", "myocardial", "infarction"],
    "EmergencyMedicine": ["sepsis", "trauma", "acute", "triage", "overdose", "cpr"],
    "InfectiousDisease": ["virus", "bacteria", "fungus", "infection", "antibiotic", "pandemic", "hiv"],
    "InternalMedicine": ["diabetes", "thyroid", "anemia", "gastro", "renal", "diagnosis"]
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

    return None

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load all vectorstores from disk into memory on startup."""
    logging.info("Loading vectorstores...")
    if not os.path.exists(VECTORSTORE_DIR):
        logging.error(f"Vectorstore directory not found: {VECTORSTORE_DIR}")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    for domain in os.listdir(VECTORSTORE_DIR):
        domain_path = os.path.join(VECTORSTORE_DIR, domain)
        if os.path.isdir(domain_path):
            try:
                logging.info(f"Loading vectorstore from: {domain_path}")
                db = Chroma(
                    persist_directory=domain_path,
                    embedding_function=embeddings
                )
                VECTORSTORES[domain] = db
                logging.info(f"Successfully loaded domain: {domain}")
            except Exception as e:
                logging.error(f"Failed to load vectorstore for {domain}: {e}")
                
    if not VECTORSTORES:
        logging.warning("No vectorstores were loaded. The /query endpoint will not work.")
    else:
        logging.info(f"Loaded {len(VECTORSTORES)} vectorstores: {list(VECTORSTORES.keys())}")

# --- Health Check Endpoint ---
@app.get("/", summary="Health Check")
async def root():
    return {"status": "ok", "loaded_vectorstores": list(VECTORSTORES.keys())}

# --- Query Endpoint ---
@app.post("/query", response_model=QueryResponse, summary="Query the RAG Pipeline")
async def query(request: QueryRequest):
    logging.info(f"Received query: {request.query} (top_k={request.top_k})")
    
    if not VECTORSTORES:
        raise HTTPException(status_code=503, detail="No vectorstores are loaded. Service is unavailable.")

    domain = get_domain(request.query)
    if not domain:
        raise HTTPException(status_code=404, detail="Could not find a relevant domain for the query.")
    
    vectorstore = VECTORSTORES[domain]

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
        retrieved_docs = retriever.invoke(request.query)
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents from vectorstore.")

    if not retrieved_docs:
        logging.warning("No relevant documents found.")
        return QueryResponse(answer="Based on the provided contexts, I cannot find the answer.", contexts=[])

    context_str = ""
    response_contexts = []
    for i, doc in enumerate(retrieved_docs):
        context_str += f"- Snippet {i+1} (from {doc.metadata.get('source', 'N/A')}, page {doc.metadata.get('page', 'N/A')}):\n"
        context_str += f"{doc.page_content}\n\n"
        
        response_contexts.append(ContextDoc(
            content=doc.page_content,
            source=doc.metadata.get('source', 'N/A'),
            page=doc.metadata.get('page', 'N/A')
        ))

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
    answer = call_llm(prompt_template)

    return QueryResponse(answer=answer, contexts=response_contexts)

# --- Run App Locally ---
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
