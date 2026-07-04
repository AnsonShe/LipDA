import os
import pickle
import numpy as np
import librosa
import json
import argparse
from tqdm import tqdm

# --- Step 1: Isolate the MFCC Extraction Logic ---
def extract_mfcc_from_audio(audio_path, sr=22050):
    """
    Loads an audio file and extracts MFCC features (including deltas).
    
    Args:
        audio_path (str): Path to the audio file (.wav).
        sr (int): The target sample rate.

    Returns:
        np.ndarray or None: A numpy array of shape (39, T) where T is the
                            number of time steps, or None if extraction fails.
    """
    try:
        # Load audio file
        audio, current_sr = librosa.load(audio_path, sr=None)
        # Resample if necessary
        if current_sr != sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
        
        # Ensure audio is not empty
        if len(audio) < 1024: # Need enough audio for at least a few frames
             print(f"  -> Warning: Audio data is too short in {os.path.basename(audio_path)}")
             return None

        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
        delta_mfcc = librosa.feature.delta(mfcc_features)
        delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
        
        # Stack features
        mfcc_stacked = np.vstack([mfcc_features, delta_mfcc, delta2_mfcc])
        return mfcc_stacked

    except Exception as e:
        print(f"  -> Error processing audio file {os.path.basename(audio_path)}: {e}")
        return None

# --- Step 2: Logic to Align MFCCs to a Fixed Sequence Length ---
def align_mfcc_to_length(mfcc_raw, target_len):
    """
    Aligns a raw MFCC array (39, T) to a target sequence length by resampling indices.
    This ensures the output MFCC sequence has the same length as the video feature sequence.
    
    Args:
        mfcc_raw (np.ndarray): The raw MFCC features of shape (39, T).
        target_len (int): The desired sequence length (e.g., 75 from the video features).

    Returns:
        np.ndarray: The aligned MFCC features of shape (target_len, 39).
    """
    if mfcc_raw is None:
        # Return a zero array if MFCC extraction failed
        return np.zeros((target_len, 39), dtype=np.float32)

    # Transpose to (T, 39) for easier processing
    mfcc_raw = mfcc_raw.T 
    num_mfcc_frames = mfcc_raw.shape[0]
    
    # Generate indices for resampling
    indices = np.linspace(0, num_mfcc_frames - 1, num=target_len, dtype=int)
    
    # Select frames based on the new indices
    aligned_mfcc = mfcc_raw[indices]
    
    return aligned_mfcc

# --- Step 3: Main Processing Logic ---
def main(args):
    """
    Main function to generate new feature caches based on audio distortions.
    """
    print("="*60)
    print(" 🚀 Starting Audio-Distorted Feature Generation")
    print("="*60)
    print(f"[*] Loading video list from: {args.json_path}")
    print(f"[*] Reading original features from: {args.original_cache_dir}")
    print(f"[*] Reading distorted audio from: {args.audio_distortion_dir}")
    print(f"[*] Saving new features to: {args.output_dir}")
    print("-"*60)

    # Load the list of test videos
    with open(args.json_path, 'r') as f:
        test_video_list = json.load(f)

    # Main loop over all videos
    for video_info in tqdm(test_video_list, desc="Processing Videos"):
        video_name = video_info['name']

        # Path to the original feature file (contains landmarks and lip_roi)
        original_pkl_path = os.path.join(args.original_cache_dir, f"{video_name}.pkl")
        # Directory containing the distorted .wav files for this video
        audio_subdir = os.path.join(args.audio_distortion_dir, video_name)

        # --- Sanity Checks ---
        if not os.path.exists(original_pkl_path):
            print(f"\n[!] Warning: Original feature file not found for '{video_name}', skipping.")
            continue
        if not os.path.isdir(audio_subdir):
            print(f"\n[!] Warning: Audio distortion directory not found for '{video_name}', skipping.")
            continue

        # Load the visual features once per video
        try:
            with open(original_pkl_path, 'rb') as f:
                original_features = pickle.load(f)
            landmarks = original_features['landmarks']
            lip_roi = original_features['lip_roi']
            target_sequence_length = landmarks.shape[0]
        except Exception as e:
            print(f"\n[!] Error: Could not load original features for '{video_name}': {e}, skipping.")
            continue

        # Inner loop over all distorted audio files for the current video
        for audio_filename in os.listdir(audio_subdir):
            if not audio_filename.endswith('.wav'):
                continue
            
            # --- Determine distortion type and prepare output path ---
            base_name = os.path.splitext(audio_filename)[0]
            if base_name == 'original':
                distortion_type = 'baseline'
            else:
                # e.g., 'output_noise_light' -> 'noise_light'
                distortion_type = '_'.join(base_name.split('_')[1:])
            
            output_dist_dir = os.path.join(args.output_dir, distortion_type)
            os.makedirs(output_dist_dir, exist_ok=True)
            output_pkl_path = os.path.join(output_dist_dir, f"{video_name}.pkl")

            if os.path.exists(output_pkl_path):
                continue # Skip if already processed

            # --- Core Operation: Create the new feature file ---
            audio_file_path = os.path.join(audio_subdir, audio_filename)
            
            # 1. Extract new MFCC from the distorted audio
            mfcc_raw = extract_mfcc_from_audio(audio_file_path)
            
            # 2. Align the new MFCC to the video's sequence length
            mfcc_aligned = align_mfcc_to_length(mfcc_raw, target_sequence_length)
            
            # 3. Combine reused visual features with new audio feature
            new_features = {
                'landmarks': landmarks,
                'lip_roi': lip_roi,
                'mfcc': mfcc_aligned
            }
            
            # 4. Save the new feature dictionary to the target directory
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(new_features, f)

    print("\n" + "="*60)
    print(" ✅ Feature generation complete!")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate S2 feature caches for audio robustness evaluation.")
    
    parser.add_argument('--json_path', type=str, default='avlip/test_videos.json',
                        help='Path to the test set JSON file')
    parser.add_argument('--original_cache_dir', type=str, default='attribute_data/features_cache/test',
                        help='Path to the directory with original S2 feature .pkl files')
    parser.add_argument('--audio_distortion_dir', type=str, default='robust/wav',
                        help='Path to the root directory containing distorted .wav files')
    parser.add_argument('--output_dir', type=str, default='attribute_data/features_cache_audio_distorted',
                        help='Path to the base output directory for new feature caches')
    
    args = parser.parse_args()
    main(args)