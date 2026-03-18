from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import requests 

app = FastAPI()

# This allows your GitHub Pages site to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change "*" to your github pages URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gemini (The Heavy Reader for long text)
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Setup Groq (Llama 3 - The Fast Tutor for quick chats)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class StudyRequest(BaseModel):
    question: str
    is_complex: bool = False

@app.post("/ask")
async def ask_ai(request: StudyRequest):
    try:
        # Route to GEMINI if the user pasted a lot of text (>100 chars)
        if request.is_complex and gemini_api_key:
            response = gemini_model.generate_content(
                f"You are a detailed professor. Explain this thoroughly: {request.question}"
            )
            # Formatting with HTML so it looks nice on the frontend
            return {"answer": f"<strong>🤖 [Gemini Node]:</strong><br><br>{response.text.replace('\n', '<br>')}"}
        
        # Route to GROQ (Llama 3) for fast, quick tutoring
        elif GROQ_API_KEY:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama3-8b-8192", 
                "messages": [{"role": "user", "content": f"You are a quick, helpful study tutor. Answer this concisely: {request.question}"}]
            }
            
            groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
            groq_data = groq_response.json()
            answer = groq_data["choices"][0]["message"]["content"]
            
            return {"answer": f"<strong>⚡ [Llama 3 Node]:</strong><br><br>{answer.replace('\n', '<br>')}"}
            
        else:
            return {"answer": "Error: API keys are not configured on the server."}

    except Exception as e:
        return {"answer": f"Backend Error: {str(e)}"}
