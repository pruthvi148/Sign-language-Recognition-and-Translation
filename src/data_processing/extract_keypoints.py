import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
from pathlib import Path

# MediaPipe hands setup
mp_hands = mp.solutions.hands

def extract_keypoints(results):
    """
    Extracts 21 landmarks for 2 hands (x, y, z) = 126 values.
    Returns a unified 1D array for the frame.
    """
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label # 'Left' or 'Right'
            kp = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            
            if label == 'Left':
                lh = kp
            else:
                rh = kp
                
    return np.concatenate([lh, rh])

def process_video(video_path, max_frames=30, frame_skip=2):
    """
    Extracts keypoints from a video sequence.
    """
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Performance optimization
            
            results = hands.process(image)
            keypoints = extract_keypoints(results)
            frames_keypoints.append(keypoints)
            
            if len(frames_keypoints) >= max_frames:
                break
                
        frame_count += 1
        
    cap.release()
    hands.close()
    
    # Pad if sequence is shorter than max_frames
    if len(frames_keypoints) < max_frames:
        padding = max_frames - len(frames_keypoints)
        for _ in range(padding):
            frames_keypoints.append(np.zeros(126))
            
    return np.array(frames_keypoints)

def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from WLASL dataset videos.")
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Path to raw video dataset')
    parser.add_argument('--output_path', type=str, default='processed_data', help='Path to save extracted keypoints')
    parser.add_argument('--max_frames', type=int, default=30)
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_path)
    output_dir = Path(args.output_path)
    
    if not dataset_dir.exists():
        print(f"Dataset directory '{dataset_dir}' not found! Please ensure it exists with the structure dataset/<word>/<video.mp4>.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Each subfolder in the dataset directory corresponds to a word
    words = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"Found {len(words)} word classes.")
    
    for word in words:
        word_dir = dataset_dir / word
        out_word_dir = output_dir / word
        os.makedirs(out_word_dir, exist_ok=True)
        
        videos = list(word_dir.glob('*.mp4'))
        for video_path in videos:
            out_file = out_word_dir / f"{video_path.stem}.npy"
            if out_file.exists():
                print(f"Skipping {word}/{video_path.name} (already processed)")
                continue
                
            print(f"Processing {word}/{video_path.name}...")
            keypoints_seq = process_video(str(video_path), max_frames=args.max_frames)
            np.save(str(out_file), keypoints_seq)
            
    print("Feature extraction complete!")

if __name__ == '__main__':
    main()
