import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
# This is here for standalone testing, but main.py loads it first
load_dotenv() 

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Google Gemini
if GEMINI_API_KEY:
    try:
        # --- ADDED LOG ---
        logging.info("GEMINI_API_KEY found, configuring model...")
        
        # --- NEW LOG: Lets verify the key is being read ---
        logging.info(f"Using key: {GEMINI_API_KEY[:4]}...{GEMINI_API_KEY[-4:]}")

        genai.configure(api_key=GEMINI_API_KEY)
        # Using the model name you specified
        gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
        logging.info("Gemini 2.5 Flash model configured successfully.")
    except Exception as e:
        # --- ADDED LOG ---
        logging.error(f"Failed to configure Gemini. Error: {e}")
        gemini_model = None
else:
    gemini_model = None
    logging.warning("GEMINI_API_KEY not found. LLM will not be available.")


def call_llm(prompt: str, max_tokens: int = 500) -> str:
    """
    Calls the configured Gemini LLM with the given prompt.
    Returns the generated text or an error message.
    """
    
    # Priority 1: Try Gemini
    if gemini_model:
        try:
            logging.info("Attempting LLM call with Gemini...")
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    temperature=0.1, # Low temp for factual answers
                    top_p=0.9,
                )
            )
            
            # --- ADDED CHECK for blocked response ---
            if response.candidates:
                if response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    logging.warning(f"Gemini call succeeded but content was blocked. Finish Reason: {response.candidates[0].finish_reason}")
                    return f"Error: Gemini response was blocked or empty. Finish Reason: {response.candidates[0].finish_reason}"
            else:
                logging.warning(f"Gemini call succeeded but returned no candidates. Prompt Feedback: {response.prompt_feedback}")
                return f"Error: Gemini returned no candidates. Feedback: {response.prompt_feedback}"
                
        except Exception as e:
            # --- THIS IS THE MOST IMPORTANT CHANGE ---
            logging.error(f"Gemini API call FAILED. Specific Error: {e}")
            return f"Error: Gemini API call failed. Details: {e}"

    # If Gemini model is not configured
    logging.error("No LLM available or it failed.")
    return "Error: No LLM is configured or available. Please check GEMINI_API_KEY."

