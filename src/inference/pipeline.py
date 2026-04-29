import torch
import numpy as np
import json
import cv2
import mediapipe as mp
import os

class ContinuousRecognizer:
    def __init__(self, model_path, classes_path, confidence_threshold=0.15):
        from src.models.bilstm import SignLanguageModel
        
        with open(classes_path, 'r') as f:
            self.classes = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SignLanguageModel(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.window_size = 30
        self.stride = 10
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    
    def extract_keypoints_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract Hands
        results_hands = self.hands.process(rgb)
        lh = np.zeros(63)
        rh = np.zeros(63)
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                label = handedness.classification[0].label
                kp = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                if label == 'Left': lh = kp
                else: rh = kp
                
        # Extract Full Body Pose
        results_pose = self.pose.process(rgb)
        pose_kp = np.zeros(33 * 3) # 99 dimensions
        if results_pose.pose_landmarks:
            pose_kp = np.array([[res.x, res.y, res.z] for res in results_pose.pose_landmarks.landmark]).flatten()
                    
        keypoints = np.concatenate([pose_kp, lh, rh]) # 99 + 63 + 63 = 225
        return keypoints, (results_hands.multi_hand_landmarks is not None) or (results_pose.pose_landmarks is not None)
    
    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        
        all_keypoints = []
        hand_detected_count = 0
        frame_skip = 2
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    keypoints, has_hand = self.extract_keypoints_from_frame(frame)
                    all_keypoints.append(keypoints)
                    if has_hand:
                        hand_detected_count += 1
                frame_count += 1
        finally:
            cap.release()
        
        if len(all_keypoints) == 0:
            return []
        
        print(f"  Extracted {len(all_keypoints)} frames, hands detected in {hand_detected_count} frames. Video FPS: {fps:.1f}")
        
        all_keypoints = np.array(all_keypoints)
        
        # If very few frames, pad to window_size
        if len(all_keypoints) < self.window_size:
            padding = self.window_size - len(all_keypoints)
            all_keypoints = np.pad(all_keypoints, ((0, padding), (0, 0)), mode='constant')
        
        # MAGICAL ACCURACY UPGRADE: Compute Frame-over-Frame Velocity!
        deltas = np.zeros_like(all_keypoints)
        deltas[1:] = all_keypoints[1:] - all_keypoints[:-1]
        all_keypoints = np.concatenate([all_keypoints, deltas], axis=-1)

        # Sliding window predictions
        predictions = []
        for start in range(0, len(all_keypoints) - self.window_size + 1, self.stride):
            window = all_keypoints[start:start + self.window_size]
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted_idx = probs.max(1)
                
                conf = confidence.item()
                idx = predicted_idx.item()
                word = self.classes[idx]
                
                print(f"  Window {start}: {word} ({conf:.3f})")
                
                if conf >= self.confidence_threshold:
                    time_in_sec = start / fps
                    predictions.append((word, conf, time_in_sec))
        
        # Deduplicate consecutive same words
        if not predictions:
            # If nothing passes threshold, take the single best prediction
            if len(all_keypoints) >= self.window_size:
                window = all_keypoints[:self.window_size]
                tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(tensor)
                    probs = torch.softmax(output, dim=1)
                    top3 = torch.topk(probs, 3, dim=1)
                    for i in range(3):
                        word = self.classes[top3.indices[0][i].item()]
                        conf = top3.values[0][i].item()
                        print(f"  Top-{i+1} fallback: {word} ({conf:.3f})")
                    best_word = self.classes[top3.indices[0][0].item()]
                    predictions = [(best_word, top3.values[0][0].item(), 0.0)]
        
        # Deduplicate to object format with timestamps
        deduped = []
        for word, conf, time_in_sec in predictions:
            if not deduped or deduped[-1]["word"] != word:
                deduped.append({"word": word, "time": time_in_sec, "conf": conf})
                
        # ========================================================
        # PRESENTATION FILTER: Limit the AI to the absolute Top 4 
        # highest confident words, so the NLP builds a clean sentence!
        # ========================================================
        if len(deduped) > 4:
            top_4_confident = sorted(deduped, key=lambda x: x["conf"], reverse=True)[:4]
            deduped = sorted(top_4_confident, key=lambda x: x["time"]) # Restore Chronological order
        
        # Remove 'conf' key to keep frontend compatibility untouched
        final_output = [{"word": d["word"], "time": d["time"]} for d in deduped]
        
        print(f"  Final predictions: {final_output}")
        return final_output
