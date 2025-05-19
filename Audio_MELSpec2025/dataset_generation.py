import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
import shutil
import soundfile as sf
import torchaudio
import random
import IPython.display as ipd
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')


class DatasetEnhancer:
    def __init__(self, input_dir='/kaggle/input/birdclef-2025', output_dir='/kaggle/working/enhanced-dataset',
                sample_size=1.0, random_seed=42):
        """
        Initialize the dataset enhancer.
        
        Args:
            input_dir: Directory containing original dataset
            output_dir: Directory to save enhanced dataset
            sample_size: Float between 0 and 1, representing the portion of the dataset to use
            random_seed: Random seed for reproducibility when sampling
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_audio_dir = os.path.join(input_dir, 'train_audio')
        self.output_train_audio_dir = os.path.join(output_dir, 'train_audio')
        self.chunk_len = 0.1  # Chunk length in seconds
        self.sample_size = min(max(sample_size, 0.0), 1.0)  # Ensure between 0 and 1
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Sampled dataframes - will be populated when needed
        self.sampled_train_df = None
        self.sampled_test_df = None
        
        # Initialize VAD model
        self._init_vad_model()
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_train_audio_dir, exist_ok=True)
    
    def _init_vad_model(self):
        """Initialize the Silero Voice Activity Detection model"""
        torch.set_num_threads(1)
        self.model, (self.get_speech_timestamps, _, _, _, _) = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad'
        )
    
    def visualize_audio_with_voice_detection(self, audio_path, title=None):
        """
        Visualize an audio file showing the audio power and voice detection segments.
        
        Args:
            audio_path: Path to the audio file
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure
        """
        # Load the audio file
        wav, sr = librosa.load(audio_path)
        
        # Calculate the sound power
        power = wav ** 2
        
        # Split the data into chunks and sum the energy in every chunk
        chunk = int(self.chunk_len * sr)
        
        pad = int(np.ceil(len(power) / chunk) * chunk - len(power))
        power = np.pad(power, (0, pad))
        power = power.reshape((-1, chunk)).sum(axis=1)
        
        # Detect speech segments
        speech_timestamps = self.get_speech_timestamps(torch.Tensor(wav), self.model)
        segmentation = np.zeros_like(wav)
        for st in speech_timestamps:
            segmentation[st['start']: st['end']] = 20
        
        # Create plot
        fig = plt.figure(figsize=(24, 3))
        
        # Set title
        if title is None:
            filename = os.path.basename(audio_path)
            fig.suptitle(f'Audio: {filename}')
        else:
            fig.suptitle(title)
        
        # Plot power in blue
        t = np.arange(len(power)) * self.chunk_len
        plt.plot(t, 10 * np.log10(power), 'b', label='Audio Power (dB)')
        
        # Plot voice segments in red
        t = np.arange(len(segmentation)) / sr
        plt.plot(t, segmentation, 'r', label='Voice Detection')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (dB) / Voice Detection')
        plt.legend()
        
        # Return the figure (doesn't show it yet)
        return fig
    
    def find_and_visualize_audio_sample(self, dataset_names, author='Fabio A. Sarria-S'):
        """
        Find an audio sample and visualize it with voice detection.
        
        Args:
            dataset_names: List of datasets to search for samples
            author: Author name to search for (default: Fabio A. Sarria-S)
            
        Returns:
            Audio display object and matplotlib figure
        """
        # Find a sample file
        sample_path = None
        sample_author = None
        sample_filename = None
        
        for dataset_name in dataset_names:
            if dataset_name.lower() == 'train':
                csv_path = os.path.join(self.input_dir, 'train.csv')
                audio_dir = self.train_audio_dir
            elif dataset_name.lower() == 'test':
                csv_path = os.path.join(self.input_dir, 'test.csv')
                audio_dir = os.path.join(self.input_dir, 'test_audio')
            else:
                continue
                
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Try to find a file by the specified author
                if 'author' in df.columns:
                    author_df = df[df.author == author] if author in df.author.values else None
                    
                    if author_df is not None and len(author_df) > 0:
                        sample_row = author_df.iloc[0]
                        sample_filename = sample_row.filename
                        sample_author = author
                        sample_path = os.path.join(audio_dir, sample_filename)
                        if os.path.exists(sample_path):
                            break
                
                # If no file by the author, use any file
                if sample_path is None and len(df) > 0:
                    sample_row = df.iloc[0]
                    sample_filename = sample_row.filename
                    sample_author = sample_row.author if 'author' in df.columns else 'Unknown'
                    sample_path = os.path.join(audio_dir, sample_filename)
                    if os.path.exists(sample_path):
                        break
        
        if sample_path is None:
            print("No suitable audio sample found.")
            return None, None
        
        # Visualize the sample
        title = f'{sample_filename} by {sample_author}'
        fig = self.visualize_audio_with_voice_detection(sample_path, title)
        
        # Create audio player
        audio = ipd.Audio(sample_path)
        
        # Return both the audio player and the figure
        return audio, fig
    
    def _sample_dataframe(self, df, sample_size):
        """
        Sample a portion of a dataframe, trying to maintain class distribution.
        
        Args:
            df: DataFrame to sample from
            sample_size: Float between 0 and 1, representing the portion to sample
            
        Returns:
            Sampled DataFrame
        """
        if sample_size >= 1.0:
            return df  # Return the entire dataframe
            
        # If we have primary_label column, try to maintain class distribution
        if 'primary_label' in df.columns:
            sampled_df = df.groupby('primary_label', group_keys=False).apply(
                lambda x: x.sample(max(1, int(len(x) * sample_size)), random_state=self.random_seed)
            )
            
            # If sampling resulted in an empty dataframe, just sample randomly
            if len(sampled_df) == 0:
                sampled_df = df.sample(max(1, int(len(df) * sample_size)), random_state=self.random_seed)
                
            return sampled_df
        else:
            # Simple random sampling if no class information
            return df.sample(max(1, int(len(df) * sample_size)), random_state=self.random_seed)
    
    def _get_sampled_train_df(self):
        """Get a sampled train dataframe"""
        if self.sampled_train_df is not None:
            return self.sampled_train_df
            
        train_csv_path = os.path.join(self.input_dir, 'train.csv')
        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
            self.sampled_train_df = self._sample_dataframe(train_df, self.sample_size)
            return self.sampled_train_df
        return None
    
    def _get_sampled_test_df(self):
        """Get a sampled test dataframe"""
        if self.sampled_test_df is not None:
            return self.sampled_test_df
            
        test_csv_path = os.path.join(self.input_dir, 'test.csv')
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            self.sampled_test_df = self._sample_dataframe(test_df, self.sample_size)
            return self.sampled_test_df
        return None
    
    def remove_human_voice(self, dataset_names=None, use_previous_output=False):
        """
        Remove human voice from recordings and create enhanced dataset.
        This is the first enhancement method.
        
        Args:
            dataset_names: List of strings with dataset names to process (e.g., ['train', 'test']).
                          If None, all available datasets will be processed.
            use_previous_output: If True, use the output of the previous enhancement as input.
        
        Returns:
            Path to enhanced dataset and sample file path
        """
        # Update input directory if using previous output
        if use_previous_output and os.path.exists(self.output_dir):
            original_input_dir = self.input_dir
            self.input_dir = self.output_dir
            self.train_audio_dir = os.path.join(self.input_dir, 'train_audio')
            
            # Create a new output directory for this enhancement
            self.output_dir = os.path.join(os.path.dirname(self.output_dir), 'voice_removed_dataset')
            self.output_train_audio_dir = os.path.join(self.output_dir, 'train_audio')
            
            # Create output directories
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_train_audio_dir, exist_ok=True)
        
        # Determine which datasets to process
        if dataset_names is None:
            # Default: process all available datasets
            dataset_names = []
            if os.path.exists(os.path.join(self.input_dir, 'train.csv')):
                dataset_names.append('train')
            if os.path.exists(os.path.join(self.input_dir, 'test.csv')):
                dataset_names.append('test')
        
        # Process selected datasets
        for dataset_name in dataset_names:
            if dataset_name.lower() == 'train':
                self._process_train_data(process_fabio_files=True)
            elif dataset_name.lower() == 'test':
                test_csv_path = os.path.join(self.input_dir, 'test.csv')
                if os.path.exists(test_csv_path):
                    self._process_test_data(test_csv_path)
        
        if self.sample_size < 1.0:
            print(f"Enhanced dataset created at {self.output_dir} for {', '.join(dataset_names)} (using {self.sample_size*100:.1f}% sample)")
        else:
            print(f"Enhanced dataset created at {self.output_dir} for {', '.join(dataset_names)}")
        
        # Play a sample audio file to verify
        # First make sure our processed files have been written to disk
        if os.path.exists(self.output_dir):
            print(f"Output directory exists: {self.output_dir}")
            
            # Print directory structure for debugging
            print("\nDirectory structure of output folder:")
            depth = 0
            for root, dirs, files in os.walk(self.output_dir):
                depth += 1
                if depth > 3:  # Limit depth to avoid too much output
                    continue
                    
                level = root.replace(self.output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root) or 'root'}/")
                sub_indent = ' ' * 4 * (level + 1)
                num_files = len(files)
                if num_files > 0:
                    print(f"{sub_indent}{num_files} files (showing up to 5)")
                    for i, f in enumerate(files[:5]):
                        print(f"{sub_indent}- {f}")
            
            # Check each dataset directory manually
            for dataset_name in dataset_names:
                if dataset_name.lower() == 'train':
                    audio_dir = os.path.join(self.output_dir, 'train_audio')
                    if os.path.exists(audio_dir):
                        print(f"Checking {audio_dir} for samples")
                        # Try to find any audio file
                        for root, dirs, files in os.walk(audio_dir):
                            for file in files:
                                if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                                    sample_path = os.path.join(root, file)
                                    print(f"Found audio file in train_audio: {sample_path}")
                                    return self.output_dir, sample_path
                elif dataset_name.lower() == 'test':
                    audio_dir = os.path.join(self.output_dir, 'test_audio')
                    if os.path.exists(audio_dir):
                        print(f"Checking {audio_dir} for samples")
                        # Try to find any audio file
                        for root, dirs, files in os.walk(audio_dir):
                            for file in files:
                                if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                                    sample_path = os.path.join(root, file)
                                    print(f"Found audio file in test_audio: {sample_path}")
                                    return self.output_dir, sample_path
            
            # Try to find a processed sample file
            sample_path = self._get_sample_audio_path(dataset_names)
            if sample_path:
                print(f"Found sample file from the voice-removed dataset: {sample_path}")
                return self.output_dir, sample_path
            else:
                print("Could not find a sample file. Searching for any audio file...")
                # Last resort: search for any audio file in the output directory
                for root, dirs, files in os.walk(self.output_dir):
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                            sample_path = os.path.join(root, file)
                            print(f"Found audio file via last resort search: {sample_path}")
                            return self.output_dir, sample_path
        
        print("WARNING: No sample audio file found.")
        return self.output_dir, None
    
    def create_fixed_duration_clips(self, duration_seconds=5, selection_method='random', 
                                   sample_rate=32000, dataset_names=None, use_previous_output=False):
        """
        Create a dataset with fixed-duration audio clips from each recording.
        
        Args:
            duration_seconds: Length of each audio clip in seconds
            selection_method: How to select the segment ('start', 'end', 'random')
            sample_rate: Sample rate for the output audio files
            dataset_names: List of strings with dataset names to process (e.g., ['train', 'test']).
                          If None, all available datasets will be processed.
            use_previous_output: If True, use the output of the previous enhancement as input.
            
        Returns:
            Path to enhanced dataset and sample file path
        """
        # Update input directory if using previous output
        if use_previous_output and os.path.exists(self.output_dir):
            original_input_dir = self.input_dir
            self.input_dir = self.output_dir
            self.train_audio_dir = os.path.join(self.input_dir, 'train_audio')
        
        # Set output directory specific to this enhancement
        clip_output_dir = os.path.join(os.path.dirname(self.output_dir), f'fixed_duration_{duration_seconds}sec')
        
        # Determine which datasets to process
        if dataset_names is None:
            # Default: process all available datasets
            dataset_names = []
            if os.path.exists(os.path.join(self.input_dir, 'train.csv')):
                dataset_names.append('train')
            if os.path.exists(os.path.join(self.input_dir, 'test.csv')):
                dataset_names.append('test')
                
        # Calculate minimum segment length in samples
        min_segment_samples = int(duration_seconds * sample_rate)
        
        # Process selected datasets
        for dataset_name in dataset_names:
            if dataset_name.lower() == 'train':
                # Process training data
                clip_train_audio_dir = os.path.join(clip_output_dir, 'train_audio')
                os.makedirs(clip_train_audio_dir, exist_ok=True)
                
                train_csv_path = os.path.join(self.input_dir, 'train.csv')
                if os.path.exists(train_csv_path):
                    self._process_fixed_duration_train(
                        train_csv_path, 
                        clip_output_dir,
                        clip_train_audio_dir,
                        duration_seconds, 
                        selection_method, 
                        sample_rate, 
                        min_segment_samples
                    )
            
            elif dataset_name.lower() == 'test':
                # Process test data
                test_csv_path = os.path.join(self.input_dir, 'test.csv')
                if os.path.exists(test_csv_path):
                    clip_test_audio_dir = os.path.join(clip_output_dir, 'test_audio')
                    os.makedirs(clip_test_audio_dir, exist_ok=True)
                    
                    self._process_fixed_duration_test(
                        test_csv_path, 
                        clip_output_dir,
                        clip_test_audio_dir,
                        duration_seconds, 
                        selection_method, 
                        sample_rate, 
                        min_segment_samples
                    )
        
        if self.sample_size < 1.0:
            print(f"Fixed-duration ({duration_seconds}s) dataset created at {clip_output_dir} for {', '.join(dataset_names)} (using {self.sample_size*100:.1f}% sample)")
        else:
            print(f"Fixed-duration ({duration_seconds}s) dataset created at {clip_output_dir} for {', '.join(dataset_names)}")
        
        # Update object state to use the new output directory for further enhancements
        self.output_dir = clip_output_dir
        self.output_train_audio_dir = os.path.join(clip_output_dir, 'train_audio')
        
        # Play a sample audio file to verify
        sample_path = self._get_sample_audio_path(dataset_names, clip_output_dir)
        if sample_path:
            print(f"Playing a sample file from the fixed-duration dataset: {sample_path}")
            return clip_output_dir, sample_path
            
        return clip_output_dir, None
    
    def _get_sample_audio_path(self, dataset_names, output_dir=None):
        """Get a sample audio file path to play for verification"""
        if output_dir is None:
            output_dir = self.output_dir
            
        print(f"Looking for sample files in: {output_dir}")
        
        # First try a direct search for any audio file in the output directory
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.ogg')):
                    sample_path = os.path.join(root, file)
                    print(f"Found sample file via direct search: {sample_path}")
                    return sample_path
        
        # If that didn't work, try to find a sample via CSV
        for dataset_name in dataset_names:
            if dataset_name.lower() == 'train':
                audio_dir = os.path.join(output_dir, 'train_audio')
                csv_path = os.path.join(output_dir, 'train.csv')
            elif dataset_name.lower() == 'test':
                audio_dir = os.path.join(output_dir, 'test_audio')
                csv_path = os.path.join(output_dir, 'test.csv')
            else:
                continue
                
            print(f"Checking {dataset_name} dataset - CSV: {csv_path}, Audio dir: {audio_dir}")
                
            # Check if the CSV exists
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    
                    # First try Fabio's files (since they had human voice)
                    if 'author' in df.columns:
                        fabio_df = df[df.author == 'Fabio A. Sarria-S']
                        
                        if len(fabio_df) > 0:
                            # Try first 5 files from Fabio
                            for idx, row in fabio_df.head(5).iterrows():
                                sample_filename = row.filename
                                sample_path = os.path.join(audio_dir, sample_filename)
                                if os.path.exists(sample_path):
                                    print(f"Found Fabio's sample file: {sample_path}")
                                    return sample_path
                    
                    # If no Fabio files or author column doesn't exist, try any file
                    if len(df) > 0:
                        # Try first 10 files from the dataset
                        for idx, row in df.head(10).iterrows():
                            sample_filename = row.filename
                            sample_path = os.path.join(audio_dir, sample_filename)
                            if os.path.exists(sample_path):
                                print(f"Found sample file: {sample_path}")
                                return sample_path
                except Exception as e:
                    print(f"Error reading CSV {csv_path}: {str(e)}")
                        
            # If CSV approach didn't work, try directory listing
            if os.path.exists(audio_dir):
                print(f"Trying direct directory listing of {audio_dir}")
                # List all subdirectories
                subdirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
                
                if subdirs:
                    # Get first subdir
                    subdir = subdirs[0]
                    subdir_path = os.path.join(audio_dir, subdir)
                    # List files in that subdir
                    try:
                        files = os.listdir(subdir_path)
                        for file in files:
                            if file.endswith(('.wav', '.mp3', '.ogg')):
                                sample_path = os.path.join(subdir_path, file)
                                print(f"Found sample via directory listing: {sample_path}")
                                return sample_path
                    except Exception as e:
                        print(f"Error listing directory {subdir_path}: {str(e)}")
                
        print("No suitable audio sample found after exhaustive search.")
        return None
    
    def play_audio_sample(self, audio_path):
        """Play an audio sample using IPython.display"""
        try:
            return ipd.Audio(audio_path)
        except Exception as e:
            print(f"Error playing audio file: {str(e)}")
            return None
    
    def _process_fixed_duration_train(self, train_csv_path, output_dir, output_audio_dir, 
                                     duration_seconds, selection_method, sample_rate, min_segment_samples):
        """
        Process training data to create fixed-duration clips.
        
        Args:
            train_csv_path: Path to training CSV
            output_dir: Base output directory
            output_audio_dir: Directory to save processed audio
            duration_seconds: Length of each clip in seconds
            selection_method: How to select the segment ('start', 'end', 'random')
            sample_rate: Sample rate for output audio
            min_segment_samples: Minimum segment length in samples
        """
        # Load training data - use sampled version if sample_size < 1
        if self.sample_size < 1.0:
            train_df = self._get_sampled_train_df()
            if train_df is None:
                train_df = pd.read_csv(train_csv_path)
        else:
            train_df = pd.read_csv(train_csv_path)
        
        # Copy the CSV file (full CSV, not just the sample)
        full_train_df = pd.read_csv(train_csv_path)
        full_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        
        print(f"Processing {len(train_df)} training files to create {duration_seconds}s clips ({selection_method})")
        
        # Process each file
        for idx, row in train_df.iterrows():
            input_path = os.path.join(self.train_audio_dir, row.filename)
            output_path = os.path.join(output_audio_dir, row.filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create fixed-duration clip
            self._create_fixed_duration_clip(
                input_path, 
                output_path, 
                selection_method, 
                sample_rate, 
                min_segment_samples
            )
    
    def _process_fixed_duration_test(self, test_csv_path, output_dir, output_audio_dir, 
                                   duration_seconds, selection_method, sample_rate, min_segment_samples):
        """
        Process test data to create fixed-duration clips.
        
        Args:
            test_csv_path: Path to test CSV
            output_dir: Base output directory
            output_audio_dir: Directory to save processed audio
            duration_seconds: Length of each clip in seconds
            selection_method: How to select the segment ('start', 'end', 'random')
            sample_rate: Sample rate for output audio
            min_segment_samples: Minimum segment length in samples
        """
        # Load test data - use sampled version if sample_size < 1
        if self.sample_size < 1.0:
            test_df = self._get_sampled_test_df()
            if test_df is None:
                test_df = pd.read_csv(test_csv_path)
        else:
            test_df = pd.read_csv(test_csv_path)
        
        # Copy the CSV file (full CSV, not just the sample)
        full_test_df = pd.read_csv(test_csv_path)
        full_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        print(f"Processing {len(test_df)} test files to create {duration_seconds}s clips ({selection_method})")
        
        # Process each file
        input_test_audio_dir = os.path.join(self.input_dir, 'test_audio')
        for idx, row in test_df.iterrows():
            input_path = os.path.join(input_test_audio_dir, row.filename)
            output_path = os.path.join(output_audio_dir, row.filename)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create fixed-duration clip
            self._create_fixed_duration_clip(
                input_path, 
                output_path, 
                selection_method, 
                sample_rate, 
                min_segment_samples
            )
    
    def _create_fixed_duration_clip(self, input_path, output_path, selection_method, sample_rate, min_segment_samples):
        """
        Create a fixed-duration clip from an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output audio file
            selection_method: How to select the segment ('start', 'end', 'random')
            sample_rate: Sample rate for output audio
            min_segment_samples: Minimum segment length in samples
        """
        # Check if output file is OGG format
        is_ogg = output_path.lower().endswith('.ogg')
        
        try:
            # For OGG files, use librosa+soundfile approach directly to avoid torchaudio codec issues
            if is_ogg:
                wav, sr = librosa.load(input_path, sr=sample_rate)
                
                # Select segment based on method
                if len(wav) <= min_segment_samples:
                    # If audio is shorter than required duration, pad with zeros
                    wav = np.pad(wav, (0, min_segment_samples - len(wav)))
                    segment = wav[:min_segment_samples]
                else:
                    if selection_method == 'start':
                        # Take from the beginning
                        segment = wav[:min_segment_samples]
                    elif selection_method == 'end':
                        # Take from the end
                        segment = wav[-min_segment_samples:]
                    elif selection_method == 'random':
                        # Take a random segment
                        max_start = len(wav) - min_segment_samples
                        start_idx = random.randint(0, max_start)
                        segment = wav[start_idx:start_idx + min_segment_samples]
                    else:
                        # Default to beginning
                        segment = wav[:min_segment_samples]
                
                # Save using soundfile
                sf.write(output_path, segment, sample_rate)
                return
            
            # For non-OGG files, use the torchaudio approach
            sig, orig_sr = torchaudio.load(input_path)
            
            # Resample if necessary
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
                sig = resampler(sig)
            
            # Get audio length in samples
            audio_length = sig.shape[1]
            
            # If audio is shorter than required duration, pad with zeros
            if audio_length <= min_segment_samples:
                sig = torch.cat([sig, torch.zeros(1, min_segment_samples - audio_length)], dim=1)
                segment = sig[:, :min_segment_samples]
            else:
                # Select segment based on method
                if selection_method == 'start':
                    # Take from the beginning
                    segment = sig[:, :min_segment_samples]
                elif selection_method == 'end':
                    # Take from the end
                    segment = sig[:, -min_segment_samples:]
                elif selection_method == 'random':
                    # Take a random segment
                    max_start = audio_length - min_segment_samples
                    start_idx = random.randint(0, max_start)
                    segment = sig[:, start_idx:start_idx + min_segment_samples]
                else:
                    # Default to beginning
                    segment = sig[:, :min_segment_samples]
            
            # Save the segment
            torchaudio.save(output_path, segment, sample_rate)
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            # If there's an error, try using librosa as fallback
            try:
                wav, sr = librosa.load(input_path, sr=sample_rate)
                wav = wav[:min_segment_samples]  # Take first segment as fallback
                if len(wav) < min_segment_samples:
                    wav = np.pad(wav, (0, min_segment_samples - len(wav)))
                sf.write(output_path, wav, sample_rate)
            except Exception as e2:
                print(f"Failed to process {input_path} with fallback method: {str(e2)}")
                # If we can't process it at all, create a silent audio file
                try:
                    silence = np.zeros(min_segment_samples)
                    sf.write(output_path, silence, sample_rate)
                    print(f"Created silent audio for {output_path}")
                except:
                    print(f"Failed to create even silent audio for {output_path}")
    
    def _process_audio_file(self, input_path, output_path):
        """
        Process an audio file to remove human voice segments.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio file
        """
        # Load the audio file
        wav, sr = librosa.load(input_path)
        
        # Detect speech segments using Silero VAD
        speech_timestamps = self.get_speech_timestamps(torch.Tensor(wav), self.model)
        
        if not speech_timestamps:
            # No speech detected, just copy the file
            sf.write(output_path, wav, sr)
            return
        
        # Create an array marking sections to keep (non-speech segments)
        keep_mask = np.ones_like(wav, dtype=bool)
        
        # Mark speech segments (with small buffers around them) for removal
        buffer_samples = int(0.2 * sr)  # 200ms buffer
        for ts in speech_timestamps:
            start = max(0, ts['start'] - buffer_samples)
            end = min(len(wav), ts['end'] + buffer_samples)
            keep_mask[start:end] = False
        
        # Keep only non-speech segments
        cleaned_audio = wav[keep_mask]
        
        # If we removed everything or almost everything, keep a portion of the original
        if len(cleaned_audio) < 0.1 * len(wav):
            # Save the first and last third of the audio without speech segments
            third = len(wav) // 3
            keep_sections = np.concatenate([wav[:third], wav[-third:]])
            sf.write(output_path, keep_sections, sr)
        else:
            # Save the cleaned audio
            sf.write(output_path, cleaned_audio, sr)
    
    def _process_train_data(self, process_fabio_files=True):
        """
        Process train dataset - copy CSV and audio files.
        For Fabio's recordings, remove human voice if process_fabio_files is True.
        
        Args:
            process_fabio_files: Whether to process Fabio's files to remove human voice
        """
        # Load train CSV - use sampled version if sample_size < 1
        train_csv_path = os.path.join(self.input_dir, 'train.csv')
        
        if self.sample_size < 1.0:
            train_df = self._get_sampled_train_df()
            if train_df is None:
                train_df = pd.read_csv(train_csv_path)
                
            # Copy the full CSV file (not just the sampled portion)
            full_train_df = pd.read_csv(train_csv_path)
            full_train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        else:
            train_df = pd.read_csv(train_csv_path)
            # Copy the original CSV
            train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        
        # Process files by author
        fabio_df = train_df[train_df.author == 'Fabio A. Sarria-S'].copy() if 'author' in train_df.columns else None
        
        # Process Fabio's recordings - remove human voice
        if fabio_df is not None and process_fabio_files and len(fabio_df) > 0:
            print(f'Processing {len(fabio_df)} recordings by Fabio A. Sarria-S to remove human voice')
            for idx, rec in fabio_df.iterrows():
                filename = rec.filename
                input_path = os.path.join(self.train_audio_dir, filename)
                output_path = os.path.join(self.output_train_audio_dir, filename)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process audio to remove human voice
                self._process_audio_file(input_path, output_path)
        
        # Copy other authors' files or all files if author info not available
        if fabio_df is not None:
            non_fabio_df = train_df[train_df.author != 'Fabio A. Sarria-S'].copy()
            print(f'Copying {len(non_fabio_df)} recordings by other authors')
            for idx, rec in non_fabio_df.iterrows():
                filename = rec.filename
                input_path = os.path.join(self.train_audio_dir, filename)
                output_path = os.path.join(self.output_train_audio_dir, filename)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Just copy the file if it's not by Fabio or if we're not processing Fabio's files
                if not os.path.exists(output_path):
                    shutil.copy2(input_path, output_path)
        else:
            # Process all files if we don't have author information
            print(f'Processing all {len(train_df)} recordings (author info not available)')
            for idx, rec in train_df.iterrows():
                filename = rec.filename
                input_path = os.path.join(self.train_audio_dir, filename)
                output_path = os.path.join(self.output_train_audio_dir, filename)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process all files to remove human voice
                self._process_audio_file(input_path, output_path)
        
        return os.path.join(self.output_dir, 'train.csv')
    
    def _process_test_data(self, test_csv_path):
        """
        Process test dataset if it exists
        
        Args:
            test_csv_path: Path to test CSV file
        
        Returns:
            Path to processed test CSV
        """
        # Load test CSV - use sampled version if sample_size < 1
        if self.sample_size < 1.0:
            test_df = self._get_sampled_test_df()
            if test_df is None:
                test_df = pd.read_csv(test_csv_path)
                
            # Copy the full CSV file (not just the sampled portion)
            full_test_df = pd.read_csv(test_csv_path)
            full_test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        else:
            test_df = pd.read_csv(test_csv_path)
            # Copy the original CSV
            test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        
        # Create test audio directory
        output_test_audio_dir = os.path.join(self.output_dir, 'test_audio')
        os.makedirs(output_test_audio_dir, exist_ok=True)
        
        # Copy test audio files
        input_test_audio_dir = os.path.join(self.input_dir, 'test_audio')
        if os.path.exists(input_test_audio_dir):
            # Process all test files to remove human voice
            print(f'Processing {len(test_df)} test recordings to remove human voice')
            for idx, row in test_df.iterrows():
                filename = row.filename
                input_path = os.path.join(input_test_audio_dir, filename)
                output_path = os.path.join(output_test_audio_dir, filename)
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process audio to remove human voice
                self._process_audio_file(input_path, output_path)
                
        return os.path.join(self.output_dir, 'test.csv')


# Example usage - Pipeline that follows the steps
# if __name__ == "__main__":
#     # Step 1: Initialize the enhancer with input and output directories
#     # and use only 1% of the data for processing
#     enhancer = DatasetEnhancer(
#         input_dir='/kaggle/input/birdclef-2025',
#         output_dir='/kaggle/working/enhanced-dataset',
#         sample_size=0.01  # Use only 1% of the data
#     )
#     
#     # Visualize a sample before processing to see voice detection
#     audio, fig = enhancer.find_and_visualize_audio_sample(['train'])
#     if fig:
#         display(fig)
#     if audio:
#         display(audio)
#     
#     # Step 2: Remove human voices from both train and test datasets
#     voice_removed_dir, voice_removed_sample = enhancer.remove_human_voice(
#         dataset_names=['train', 'test']
#     )
#     
#     # Play a sample after removing human voice
#     if voice_removed_sample:
#         print("Playing a sample audio after removing human voice:")
#         display(enhancer.play_audio_sample(voice_removed_sample))
#         
#         # Visualize the voice-removed sample
#         fig = enhancer.visualize_audio_with_voice_detection(
#             voice_removed_sample, 
#             "Voice-removed audio sample"
#         )
#         display(fig)
#     
#     # Step 3: Create 5-second clips from the human-voice-removed dataset
#     clips_dir, clip_sample = enhancer.create_fixed_duration_clips(
#         duration_seconds=5,
#         selection_method='start',
#         sample_rate=32000,
#         dataset_names=['train', 'test'],
#         use_previous_output=True  # Use the output from the voice removal step
#     )
#     
#     # Play a sample after creating 5-second clips
#     if clip_sample:
#         print("Playing a sample audio after creating 5-second clips:")
#         display(enhancer.play_audio_sample(clip_sample))
#         
#         # Visualize the 5-second clip
#         fig = enhancer.visualize_audio_with_voice_detection(
#             clip_sample,
#             "5-second clip sample"
#         )
#         display(fig)
#         
#     print("Enhancement pipeline completed successfully!")
#     print(f"Voice-removed dataset: {voice_removed_dir}")
#     print(f"Fixed-duration clips dataset: {clips_dir}")
#     print(f"Only processed {enhancer.sample_size*100:.1f}% of the data for demonstration purposes.")