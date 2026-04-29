from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import tempfile
import os
import sys
import subprocess

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

try:
    from inference.pipeline import ContinuousRecognizer
    from inference.nlp_translator import T5Translator
except ImportError:
    pass

app = FastAPI(title="Sign Language Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
recognizer = None
translator = None

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, 'frontend', 'dist')

@app.on_event("startup")
def load_models():
    global recognizer, translator
    model_path = os.path.join(BASE_DIR, 'models', 'best_model.pth')
    classes_path = os.path.join(BASE_DIR, 'models', 'classes.json')

    # Auto-build frontend if dist doesn't exist
    frontend_dir = os.path.join(BASE_DIR, 'frontend')
    if not os.path.isdir(DIST_DIR) and os.path.isfile(os.path.join(frontend_dir, 'package.json')):
        print("Building frontend for the first time...")
        try:
            subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, shell=True, check=True)
            print("Frontend built successfully!")
        except Exception as e:
            print(f"Warning: Could not build frontend: {e}")

    if os.path.exists(model_path) and os.path.exists(classes_path):
        print("Loading Sign Language Models...")
        recognizer = ContinuousRecognizer(model_path=model_path, classes_path=classes_path)
    else:
        print("Warning: Model files not found. Inference will return mock data.")

    print("Loading NLP Translator...")
    try:
        translator = T5Translator()
    except ValueError as ve:
        print(f"Warning: {ve}")
    except Exception as e:
        print(f"Warning: NLP Translator could not be loaded: {e}")

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if not recognizer:
        return {"words": ["hello", "how", "you"], "sentence": "Hello, how are you? (Mock - model not loaded)"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        predictions = recognizer.predict(tmp_path)
        if predictions:
            if translator:
                sentence = "[LIVE] " + translator.translate(predictions)
            else:
                sentence = "[NLP Cloud Disconnected - Using Raw Fallback]: " + " ".join([d["word"] if isinstance(d, dict) else d for d in predictions])
        else:
            sentence = "No signs detected."
            
        return {"words": predictions, "sentence": sentence}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# --- Serve the compiled React frontend from /frontend/dist ---
if os.path.isdir(DIST_DIR):
    # Serve static assets (JS, CSS, images)
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(DIST_DIR, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Try to serve the exact file first
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return HTMLResponse("""
        <html><body style="background:#0f172a;color:white;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
        <div style="text-align:center">
            <h1>SignTranslate API is Running!</h1>
            <p>Frontend not built yet. Run this in your terminal:</p>
            <code style="background:#1e1b4b;padding:10px 20px;border-radius:8px;display:block;margin:20px auto">cd frontend && npm run build</code>
            <p>Then restart the server.</p>
        </div></body></html>
        """)
