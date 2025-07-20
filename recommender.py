import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Load environment variable
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize Flash model
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_recommendations(summary_text: str, additional_context: str = None) -> str:
    prompt = (
        "You are a smart financial advisor AI.\n\n"
        "Given the following user's UPI spending breakdown:\n\n"
        f"{summary_text}\n\n"
        "Please provide:\n"
        "- A 2-line budget health overview\n"
        "- 2 unnecessary spending patterns with tips to reduce\n"
        "- Suggestions for better financial control\n"
        "- A brief motivational note on financial discipline\n\n"
        "Keep it personal, clear, and concise. Use bullet points if needed."
    )
    if additional_context:
        prompt += f"Context: {additional_context}\n\n"
    prompt += f"Spending Summary:\n{summary_text}\n\nKeep it short and actionable."

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except ResourceExhausted:
        return "🚫 You’ve reached the daily usage limit. Please try again tomorrow or upgrade your API quota."
    except Exception as e:
        return f"❌ An error occurred while generating recommendations: {e}"
