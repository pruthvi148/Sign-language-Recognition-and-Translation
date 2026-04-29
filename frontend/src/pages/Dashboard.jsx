import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Webcam from 'react-webcam';
import { 
  MessageSquare, LogOut, Moon, Sun, 
  Video, Trash2, Download, Upload, FileVideo, X
} from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'webcam'
  
  // Translation State
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [autoTranslatePending, setAutoTranslatePending] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Upload State
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  
  // Refs
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const fileInputRef = useRef(null);
  const displayVideoRef = useRef(null);

  useEffect(() => {
    const activeUser = localStorage.getItem('signlive_active_user');
    if (!activeUser) {
      navigate('/');
    } else {
      setUser(JSON.parse(activeUser));
    }
  }, [navigate]);

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
  }, [darkMode]);

  const handleLogout = () => {
    localStorage.removeItem('signlive_active_user');
    navigate('/');
  };

  // --- Upload Logic ---
  const handleFile = (file) => {
    if (file.type.startsWith('video/')) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('Please upload a valid video file (.mp4, .mov, etc)');
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleRemove = (e) => {
    if (e) e.stopPropagation();
    setVideoFile(null);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setResult(null);
  };

  // --- Webcam Logic ---
  const handleDataAvailable = useCallback(({ data }) => {
    if (data.size > 0) {
      setRecordedChunks((prev) => prev.concat(data));
    }
  }, [setRecordedChunks]);

  const handleStartCaptureClick = useCallback(() => {
    setRecordedChunks([]);
    setError(null);
    setResult(null);
    
    if (!webcamRef.current?.stream) {
      setError("Webcam stream is not available. Check your permissions.");
      return;
    }
    
    try {
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: 'video/webm'
      });
      mediaRecorderRef.current.addEventListener("dataavailable", handleDataAvailable);
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch(err) {
      setError("Failed to start MediaRecorder: " + err.message);
    }
  }, [webcamRef, setIsRecording, handleDataAvailable]);

  const handleStopCaptureClick = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setAutoTranslatePending(true);
    }
  }, [mediaRecorderRef, isRecording]);

  const handleTranslate = async (fileToTranslate) => {
    if (!fileToTranslate) return;
    
    setIsLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', fileToTranslate);
    
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
          const errText = await response.text();
          throw new Error(`Server Error [${response.status}]: ${errText.substring(0, 100)}`);
      }
      
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the translation API');
    } finally {
      setIsLoading(false);
      setAutoTranslatePending(false);
    }
  };

  useEffect(() => {
    if (!isRecording && recordedChunks.length > 0 && autoTranslatePending) {
      const blob = new Blob(recordedChunks, { type: "video/webm" });
      const file = new File([blob], "webcam_capture.webm", { type: "video/webm" });
      setRecordedChunks([]); 
      handleTranslate(file);
    }
  }, [isRecording, recordedChunks, autoTranslatePending]);

  const handleClear = () => {
    setResult(null);
    setError(null);
  };

  const handleDownload = () => {
    if (!result?.sentence) return;
    const element = document.createElement("a");
    const file = new Blob([result.sentence], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = "signlive_transcript.txt";
    document.body.appendChild(element);
    element.click();
  };

  const currentConfidence = result && result.words && result.words.length > 0 
    ? Math.round(result.words.reduce((acc, curr) => acc + (curr.conf || 0.8), 0) / result.words.length * 100) 
    : 0;

  if (!user) return null;

  return (
    <div className={`dashboard-wrapper ${darkMode ? 'dark' : 'light'}`}>
      <nav className="dash-nav">
        <div className="nav-brand">
          <div className="logo-icon-wrap">
            <MessageSquare size={24} color="#fff" />
          </div>
          <h1>SignLive</h1>
        </div>
        <div className="nav-actions">
          <div className="user-avatar">
            {user.name.charAt(0).toUpperCase()}
          </div>
          <span className="user-name">{user.name.split(' ')[0]}</span>
          <button className="icon-btn" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </nav>

      <main className="dash-main">
        <aside className="dash-sidebar left-sidebar">
          
          {/* Mode Toggle */}
          <div className="panel mode-toggle-panel">
            <div className="tabs">
              <button 
                className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => { setActiveTab('upload'); handleRemove(); }}
              >
                <Upload size={16} /> Upload Video
              </button>
              <button 
                className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
                onClick={() => { setActiveTab('webcam'); handleRemove(); }}
              >
                <Video size={16} /> Live Webcam
              </button>
            </div>
          </div>

          <div className="panel status-panel">
            <h3>STATUS</h3>
            <div className="status-indicator">
              <div className={`dot ${isRecording ? 'active' : isLoading ? 'loading' : 'idle'}`}></div>
              <span>{isRecording ? 'Recording...' : isLoading ? 'Translating...' : 'Idle'}</span>
            </div>
            
            <div className="last-detected">
              <span className="label">Last detected</span>
              <div className="detected-word">
                {result && result.words && result.words.length > 0 
                  ? result.words[result.words.length - 1].word || result.words[result.words.length - 1] 
                  : 'None'}
              </div>
            </div>

            <div className="confidence-meter">
              <div className="conf-header">
                <span className="label">Confidence</span>
                <span className="value">{currentConfidence}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${currentConfidence}%` }}></div>
              </div>
            </div>
          </div>

          <div className="panel controls-panel">
            <h3>CONTROLS</h3>
            {activeTab === 'webcam' ? (
              <button 
                className={`primary-btn ${isRecording ? 'stop-btn' : ''}`}
                onClick={isRecording ? handleStopCaptureClick : handleStartCaptureClick}
                disabled={isLoading}
              >
                <Video size={18} />
                {isRecording ? 'Stop Recognition' : 'Start Recognition'}
              </button>
            ) : (
              <button 
                className="primary-btn"
                onClick={() => handleTranslate(videoFile)}
                disabled={!videoFile || isLoading}
              >
                <MessageSquare size={18} />
                Translate Video
              </button>
            )}
            
            <button className="outline-btn" onClick={handleClear} disabled={!result}>
              <Trash2 size={18} />
              Clear Captions
            </button>
            <button className="outline-btn" onClick={handleDownload} disabled={!result}>
              <Download size={18} />
              Download Transcript
            </button>
          </div>

          <div className="panel support-panel">
            <h3>MEETING SUPPORT</h3>
            <p>Share this screen during Zoom, Teams, or Google Meet to display live sign language captions.</p>
          </div>
        </aside>

        <section className="dash-camera">
          <div className={`camera-container ${isRecording ? 'recording' : ''}`}>
             
             {activeTab === 'webcam' ? (
               <>
                 <Webcam
                    audio={false}
                    ref={webcamRef}
                    mirrored={true}
                    className="webcam-feed"
                    videoConstraints={{ facingMode: "user" }}
                 />
                 {!isRecording && !isLoading && (
                   <div className="camera-overlay">
                      <div className="cam-icon-circle">
                         <Video size={32} color="#a1a1aa" />
                      </div>
                      <h2>Camera is standby</h2>
                      <p>Click "Start Recognition" to begin</p>
                   </div>
                 )}
               </>
             ) : (
               <>
                 {!videoUrl ? (
                   <div 
                     className="camera-overlay upload-mode-overlay"
                     onClick={() => fileInputRef.current?.click()}
                   >
                     <div className="cam-icon-circle">
                        <Upload size={32} color="#a1a1aa" />
                     </div>
                     <h2>Upload Sign Language Video</h2>
                     <p>Click here to browse files (MP4, MOV)</p>
                     <input 
                       type="file" 
                       ref={fileInputRef} 
                       onChange={handleFileChange} 
                       accept="video/*" 
                       style={{ display: 'none' }}
                     />
                   </div>
                 ) : (
                   <div className="video-preview-mode">
                     <button className="remove-vid-btn" onClick={handleRemove}><X size={16}/></button>
                     <video 
                       src={videoUrl} 
                       controls autoPlay playsInline loop
                       className="uploaded-vid-feed"
                       ref={displayVideoRef}
                     ></video>
                   </div>
                 )}
               </>
             )}

             {isLoading && (
               <div className="camera-overlay loading-overlay">
                  <div className="spinner"></div>
                  <h2>Analyzing Gestures...</h2>
               </div>
             )}

             {error && (
               <div className="camera-error">
                 {error}
               </div>
             )}
          </div>
        </section>

        <aside className="dash-sidebar right-sidebar">
          <div className="panel captions-panel">
            <h3><MessageSquare size={16}/> LIVE CAPTIONS</h3>
            <div className="live-caption-box">
              <span className="tag">[Live Caption]</span>
              <p className="caption-text">
                {result ? result.sentence : 'Waiting for signs...'}
              </p>
            </div>

            <div className="word-history">
              <h3>WORD HISTORY</h3>
              <div className="history-list">
                {result && result.words ? result.words.map((w, i) => {
                  const wordText = typeof w === 'object' ? w.word : w;
                  const conf = typeof w === 'object' && w.conf ? Math.round(w.conf * 100) : 85;
                  return (
                    <div key={i} className="history-pill">
                      {wordText} <span className="conf-tag">{conf}%</span>
                    </div>
                  );
                }) : (
                  <div className="empty-history">No history yet</div>
                )}
              </div>
            </div>
            
            <div className="sidebar-footer">
              Share this window during meetings for live sign language captions
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
