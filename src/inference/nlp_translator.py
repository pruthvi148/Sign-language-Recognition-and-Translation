import os
import google.generativeai as genai
from dotenv import load_dotenv

class T5Translator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Check API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use gemini-flash-latest which is highly capable and supports the generation methods needed
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            print(f"Error loading Gemini model: {e}")
            raise e
            
        # Keep instruction similar to the original T5 design but optimized for Cloud
        self.system_instruction = (
            "You are an expert American Sign Language (ASL) interpreter translating a news broadcast. "
            "I will provide you with a list of disjointed English glosses that were extracted "
            "from a live video of a news anchor by a PyTorch AI. Because it's a fast news broadcast, "
            "the AI is only detecting the most rigid nouns and verbs. "
            "Your job is to read these disjointed clues and brilliantly deduce what the anchor was conceptually saying, "
            "constructing it into a single, fluent, highly intelligent, and mature English sentence.\n"
            "Do not add conversational filler. Output ONLY the final translated sentence.\n\n"
            "Signs detected: "
        )

    def translate(self, words):
        """
        Translates a list of word dictionaries or strings into a proper sentence using Google Gemini.
        """
        if not words:
            return "No signs detected."

        # Extract just the string words
        extracted_words = []
        for w in words:
            if isinstance(w, dict):
                extracted_words.append(w["word"])
            else:
                extracted_words.append(w)
                
        if not extracted_words:
            return "No signs detected."
            
        input_text = self.system_instruction + " ".join(extracted_words)

        try:
            response = self.model.generate_content(input_text)
            return response.text.strip().replace("\n", " ")
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return f"[Google API Error: {e}] {' '.join(extracted_words)}"
