import os
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (must come BEFORE you access them)
load_dotenv()

# Fetch the Gemini API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
print("Loaded GOOGLE_API_KEY:", api_key)  # Debug print

if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

# Configure Gemini with the provided API key
genai.configure(api_key=api_key)

# Initialize Gemini model once
MODEL_NAME = "gemini-1.5-pro"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")
    raise

def safe_generate(model, prompt, retries=3, sleep_time=20):
    """
    Attempt to generate a response from the Gemini model, with retries for rate limits and transient errors.
    """
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            else:
                logger.error("Gemini API returned an unexpected response structure.")
                raise ValueError("Gemini API returned an unexpected response structure.")
        except google.api_core.exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit. Retrying in {sleep_time} seconds... ({attempt + 1}/{retries})")
            time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"An error occurred during Gemini API call: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {sleep_time} seconds... ({attempt + 1}/{retries})")
                time.sleep(sleep_time)
            else:
                raise
    raise RuntimeError("Exceeded retry attempts due to persistent errors with Gemini API.")

def generate_recommendations(summary_text: str, additional_context: str = None) -> str:
    """
    Generate financial advice using Gemini given spending summary text.
    Optionally, additional context (user goals, age, etc.) can be provided for more personalization.
    """
    prompt = (
        "You are a financial advisor AI.\n\n"
        "Based on the user's spending summary below, provide:\n"
        "1. Monthly budget planning advice.\n"
        "2. Suggestions to reduce unnecessary spending.\n"
        "3. Personalized financial advice.\n\n"
    )
    if additional_context:
        prompt += f"Additional context: {additional_context}\n\n"

    prompt += f"Spending Summary:\n{summary_text}\n\nPlease provide concise, actionable advice."

    try:
        advice = safe_generate(model, prompt)
        return advice
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        return "Sorry, I couldn't generate recommendations at this time due to a technical issue."

# Example usage
if __name__ == "__main__":
    example_summary = """
    - Total spent in June: ₹20,000
    - Groceries: ₹5,000
    - Food Delivery: ₹6,000
    - Entertainment: ₹3,000
    - UPI Transfers to friends: ₹2,000
    - Subscriptions: ₹4,000
    """
    print(generate_recommendations(example_summary))