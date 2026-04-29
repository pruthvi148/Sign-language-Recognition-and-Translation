# SignLive: ASL Recognition & Translation

SignLive is a modern **Hybrid Edge-Cloud** web application that translates American Sign Language (ASL) into grammatically perfect English sentences in real-time. It supports both live webcam feeds and pre-recorded video file uploads.

![SignLive Dashboard](frontend/src/assets/hero.png) *(Note: Add a screenshot of your dashboard here)*

## Architecture

This project combines local Computer Vision (Edge) with advanced Large Language Models (Cloud) to achieve seamless translation without the massive latency of sending raw video to the cloud.

1. **Computer Vision (MediaPipe):** Extracts precise 3D hand and pose coordinates (225 total landmarks per frame) locally to avoid uploading heavy video files.
2. **Machine Learning (PyTorch BiLSTM):** A custom Bidirectional Long Short-Term Memory neural network analyzes the extracted coordinate sequence over a 30-frame sliding window to predict disjointed English Glosses (e.g., `ceiling professor teacher bookstore`).
3. **NLP (Google Gemini API):** Acts as the "linguistic brain", mathematically restructuring the raw disjointed ASL glosses into fluent, grammatically perfect English sentences using `gemini-pro`.
4. **Backend (FastAPI):** High-performance asynchronous Python server managing the prediction pipelines.
5. **Frontend (React/Vite):** A professional, responsive 3-pane dashboard featuring simulated authentication, history tracking, and dark/light modes.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js & npm
- A Google Gemini API Key (Get one free at [Google AI Studio](https://aistudio.google.com/app/apikey))

### 1. Clone the Repository
```bash
git clone https://github.com/pruthvi148/Sign-language-Recognition-and-Translation.git
cd Sign-language-Recognition-and-Translation
```

### 2. Setup the Python Backend
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install google-generativeai python-dotenv
```

### 3. Configure the Environment
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Download the Model Weights
Ensure your pre-trained PyTorch weights (`best_model.pth`) are placed inside the `models/` directory.

### 5. Setup the React Frontend
The frontend requires an initial build so the Python server can serve it statically.
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## Usage

You can launch the entire application (Backend + Frontend) with a single command. 

```bash
python run_server.py
```
This will automatically:
- Start the Uvicorn/FastAPI server on `http://localhost:8000`
- Open your default web browser to the SignLive Dashboard.

### Features
- **Live Webcam Mode:** Perform signs directly into your camera for continuous real-time translation.
- **Upload Mode:** Drag and drop an `.mp4` video to extract and translate signs.
- **Dark/Light Theme:** Toggle the visual aesthetic of the dashboard.
- **Real-time Confidence Tracking:** View the AI's confidence percentage for every detected word.

## License
MIT License
