import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as AT
from torchvision import models
from sklearn.metrics import roc_auc_score
import traceback

class SpecTor:
    def __init__(self, fft_params, sample=0.1, mode='train'):
        """
        Initialize the SpecTor class for processing audio data and generating spectrograms.
        
        Args:
            fft_params (dict): Dictionary containing FFT parameters:
                - n_fft: FFT window size
                - win_length: Window length
                - hop_length: Hop length
                - f_min: Minimum frequency
                - f_max: Maximum frequency
                - n_mels: Number of mel bands
                - sample_rate: Audio sample rate
            sample (float): Fraction of data to use (between 0 and 1)
            mode (str): 'train' or 'test' mode
        """
        self.fft_params = fft_params
        self.sample = sample
        self.mode = mode
        
        # Set paths for different data folders
        self.original_data_folder = "/kaggle/input/birdclef-2025/"
        self.original_train_audio_folder = "/kaggle/input/birdclef-2025/train_audio/"
        self.enhanced_train_audio_folder = "/kaggle/input/birdclef-2025-preprocessed-dataset-i/fixed_duration_5sec/"
        self.notebook_one_path = "/kaggle/input/clef-dataset-enhancer-part-i"
        self.original_train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes/"
        self.original_test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
        
        # Load class labels from original training data (for consistency)
        self.class_labels = sorted(os.listdir(self.original_train_audio_folder))
        
        # Load and process training metadata from enhanced dataset
        train_csv_path = os.path.join(self.enhanced_train_audio_folder, 'train.csv')
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training CSV not found at: {train_csv_path}")
            
        # Load the training metadata
        self.meta = pd.read_csv(train_csv_path)
        
        # Clean up the DataFrame
        self.meta = self._clean_dataframe(self.meta)
        
        # Sample the data if needed
        if self.sample < 1.0:
            self.meta = self._sample_dataframe(self.meta, self.sample)
        
        # Initialize mel spectrogram transform
        self.mel_spectrogram = AT.MelSpectrogram(
            sample_rate=fft_params['sample_rate'],
            n_fft=fft_params['n_fft'],
            win_length=fft_params['win_length'],
            hop_length=fft_params['hop_length'],
            center=True,
            f_min=fft_params['f_min'],
            f_max=fft_params['f_max'],
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            n_mels=fft_params['n_mels'],
            mel_scale="htk",
        )
        
        # Create dataset and dataloader
        self.dataset = BirdclefDataset(
            self.meta, 
            mode=self.mode,
            audio_folder=self.enhanced_train_audio_folder,  # Use enhanced audio folder
            class_labels=self.class_labels,
            mel_spectrogram=self.mel_spectrogram
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=24, 
            shuffle=(self.mode == 'train'),
            num_workers=1,
            drop_last=(self.mode == 'train')
        )
    
    def _clean_dataframe(self, df):
        """
        Clean and format the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert string representations of lists to actual lists
        for col in ['secondary_labels', 'type']:
            if col in df.columns:
                df[col] = df[col].apply(eval)
        
        # Ensure filename column exists and is properly formatted
        if 'filename' not in df.columns:
            raise ValueError("DataFrame must contain 'filename' column")
        
        # Ensure primary_label column exists
        if 'primary_label' not in df.columns:
            raise ValueError("DataFrame must contain 'primary_label' column")
        
        # Remove any rows with missing essential data
        df = df.dropna(subset=['filename', 'primary_label'])
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        return df
    
    def _sample_dataframe(self, df, sample_size):
        """
        Sample a portion of the dataframe while maintaining class distribution.
        
        Args:
            df: DataFrame to sample from
            sample_size: Float between 0 and 1, representing the portion to sample
            
        Returns:
            Sampled DataFrame
        """
        if sample_size >= 1.0:
            return df
            
        # Sample while maintaining class distribution
        sampled_df = df.groupby('primary_label', group_keys=False).apply(
            lambda x: x.sample(max(1, int(len(x) * sample_size)), random_state=42),
            include_groups=False
        )
        
        # If sampling resulted in an empty dataframe, just sample randomly
        if len(sampled_df) == 0:
            sampled_df = df.sample(max(1, int(len(df) * sample_size)), random_state=42)
            
        return sampled_df.reset_index(drop=True)
    
    def plot_spectrogram(self, specgram, title=None, ylabel="freq_bin"):
        """
        Plot a spectrogram.
        
        Args:
            specgram: Spectrogram data
            title: Optional title for the plot
            ylabel: Label for the y-axis
        """
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or "Spectrogram (db)")
        axs.set_ylabel(ylabel)
        axs.set_xlabel("frame")
        im = axs.imshow(specgram, origin="lower", aspect="auto")
        fig.colorbar(im, ax=axs)
        plt.show(block=False)
    
    def get_dataloader(self):
        """Return the dataloader for the dataset."""
        return self.dataloader
    
    def get_dataset(self):
        """Return the dataset."""
        return self.dataset


class BirdclefDataset(Dataset):
    def __init__(self, df, mode='train', audio_folder=None, class_labels=None, mel_spectrogram=None, target_length=5, verbose=0):
        """
        Initialize the BirdclefDataset.
        
        Args:
            df: DataFrame containing metadata
            mode: 'train' or 'test' mode
            audio_folder: Path to audio files
            class_labels: List of class labels
            mel_spectrogram: Pre-initialized mel spectrogram transform
            target_length: Target length of audio in seconds
            verbose: Logging level (-1: time only, 0: no prints, 1: all prints)
        """
        self.df = df
        self.mode = mode
        self.audio_folder = audio_folder
        self.class_labels = class_labels
        self.target_length = target_length
        self.verbose = verbose
        
        # For time estimation when verbose=-1
        self.start_time = None
        self.processed_items = 0
        
        # Use provided mel spectrogram or create default one
        self.mel_spectrogram = mel_spectrogram if mel_spectrogram is not None else AT.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            center=True,
            f_min=20,
            f_max=15000,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            n_mels=128,
            mel_scale="htk",
        )
        
        # Calculate target number of frames
        self.target_samples = int(target_length * 32000)
        self.target_frames = int((self.target_samples + self.mel_spectrogram.hop_length - 1) // self.mel_spectrogram.hop_length)

    def log(self, message, level=1):
        """Helper function to handle logging based on verbose level."""
        if self.verbose == level:
            print(message)

    def estimate_time(self, index):
        """Estimate remaining processing time."""
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
            return
            
        self.processed_items += 1
        if self.processed_items % 10 == 0:  # Update every 10 items
            elapsed_time = time.time() - self.start_time
            items_per_second = self.processed_items / elapsed_time
            remaining_items = len(self.df) - index
            estimated_remaining_time = remaining_items / items_per_second
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            seconds = int(estimated_remaining_time % 60)
            
            print(f"\rProcessing {index}/{len(self.df)} - Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")

    def normalize_std(self, spec, eps=1e-23):
        """
        Normalize spectrogram using standardization.
        
        Args:
            spec: Spectrogram tensor
            eps: Small value to avoid division by zero
            
        Returns:
            Normalized spectrogram
        """
        mean = torch.mean(spec)
        std = torch.std(spec)
        return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))
    
    def process_audio(self, sig, sr):
        """
        Process audio signal to fixed length.
        
        Args:
            sig: Audio signal tensor
            sr: Sample rate
            
        Returns:
            Processed audio signal
        """
        # Ensure audio is mono
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
        
        # Calculate exact number of samples needed for target frames
        target_samples = (self.target_frames - 1) * self.mel_spectrogram.hop_length + 1
        
        # Handle audio that's too short or too long
        if sig.shape[1] < target_samples:
            # Pad with zeros if too short
            padding = target_samples - sig.shape[1]
            sig = torch.nn.functional.pad(sig, (0, padding))
        else:
            # Take exact number of samples needed
            sig = sig[:, :target_samples]
        
        return sig
                
    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to get
            
        Returns:
            Tuple of (spectrogram, target) for train mode or (spectrogram, filename) for test mode
        """
        try:
            # Time estimation for verbose=-1
            if self.verbose == -1:
                self.estimate_time(index)
            
            # Load audio file
            filename = self.df.iloc[index].filename
            
            # Check if audio_folder is set
            if self.audio_folder is None:
                raise ValueError(f"audio_folder is not set for the dataset. File: {filename}")
            
            file_path = os.path.join(self.audio_folder, filename)
            self.log(f"\nProcessing file {index}: {filename}", 1)
            
            if not os.path.exists(file_path):
                error_msg = f"Audio file not found: {file_path}"
                self.log(f"Error: File not found at {file_path}", 1)
                raise FileNotFoundError(error_msg)
            
            self.log("1. Loading audio file...", 1)
            # Load and verify audio
            sig, sr = torchaudio.load(uri=file_path, backend="soundfile")
            if sig.numel() == 0:
                error_msg = f"Empty audio file: {file_path}"
                self.log(f"Error: Empty audio file detected", 1)
                raise ValueError(error_msg)
            self.log(f"   → Audio loaded successfully: shape={sig.shape}, sample_rate={sr}Hz", 1)
            
            self.log("2. Processing audio to fixed length...", 1)
            # Process audio to fixed length
            sig = self.process_audio(sig, sr)
            self.log(f"   → Audio processed to shape: {sig.shape}", 1)
            
            self.log("3. Normalizing audio...", 1)
            # Normalize audio
            max_val = torch.max(torch.abs(sig))
            if max_val == 0:
                error_msg = f"Audio file has zero amplitude: {file_path}"
                self.log(f"Error: Zero amplitude audio detected", 1)
                raise ValueError(error_msg)
            sig = sig / max_val
            
            # Add small random noise (only for training)
            if self.mode == 'train':
                self.log("4. Adding random noise for training...", 1)
                sig = sig + 1.5849e-05 * (torch.rand_like(sig) - 0.5)
            
            self.log("5. Generating mel spectrogram...", 1)
            # Generate mel spectrogram
            melspec = self.mel_spectrogram(sig)
            if torch.isnan(melspec).any() or torch.isinf(melspec).any():
                error_msg = f"Invalid values in mel spectrogram for file: {file_path}"
                self.log(f"Error: Invalid values in mel spectrogram", 1)
                raise ValueError(error_msg)
            
            self.log("6. Applying log transform and normalization...", 1)
            melspec = torch.log(melspec + 1e-9)  # Add small epsilon to avoid log(0)
            melspec = self.normalize_std(melspec)
            self.log(f"   → Spectrogram shape: {melspec.shape}", 1)
            
            # Verify final spectrogram shape
            expected_shape = (1, self.mel_spectrogram.n_mels, self.target_frames)
            if melspec.shape != expected_shape:
                error_msg = f"Unexpected mel spectrogram shape: {melspec.shape}, expected: {expected_shape}"
                self.log(f"Error: Unexpected spectrogram shape", 1)
                raise ValueError(error_msg)

            if self.mode == 'train':
                self.log("7. Creating target vector...", 1)
                # Extract target from filename (first part before the /)
                target = filename.split('/')[0]
                
                # Check if class_labels is set
                if self.class_labels is None:
                    raise ValueError(f"class_labels is not set for the dataset. File: {filename}")
                
                if target not in self.class_labels:
                    error_msg = f"Unknown class label: {target}"
                    self.log(f"Error: Unknown class label '{target}'", 1)
                    raise ValueError(error_msg)
                y = np.array([1 if item == target else 0 for item in self.class_labels])
                self.log(f"   → Target created for class: {target}", 1)
                self.log("Processing completed successfully!", 1)
                return melspec, y
            else:
                self.log("7. Preparing test output...", 1)
                self.log("Processing completed successfully!", 1)
                return melspec, filename
                
        except Exception as e:
            if self.verbose >= 0:  # Show errors for both verbose=0 and 1
                print(f"\nError processing file at index {index}: {str(e)}")
                print(f"File path: {file_path if 'file_path' in locals() else 'unknown'}")
                print(f"DataFrame row: {self.df.iloc[index].to_dict()}")
            raise
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.df)


class ModelTrainer:
    def __init__(self, class_labels, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the ModelTrainer class.
        
        Args:
            class_labels: List of class labels
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.class_labels = class_labels
        self.device = device
        self.model = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
        print(f"\nInitializing ModelTrainer on device: {device}")
        
    def load_model(self, pretrained=True):
        """Load and initialize the ResNet34 model."""
        print("\n=== Loading ResNet34 Model ===")
        try:
            class ResNet34Model(nn.Module):
                def __init__(self, num_classes, pretrained=True):
                    super().__init__()
                    print(f"→ Loading {'pretrained' if pretrained else 'untrained'} ResNet34...")
                    model = models.resnet34(pretrained=pretrained)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    self.model = model
                    print(f"→ Modified final layer to output {num_classes} classes")

                def forward(self, x):
                    # Convert single channel to 3 channels
                    x = torch.cat((x, x, x), 1)
                    x = self.model(x)
                    return x

            self.model = ResNet34Model(len(self.class_labels), pretrained=pretrained)
            self.model = self.model.to(self.device)
            print("→ Successfully moved model to", self.device)
            print("Model loading completed successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def verify_files(self, df, audio_folder):
        """Verify that audio files exist and filter out those that don't."""
        print("→ Verifying files exist...")
        
        valid_rows = []
        invalid_files = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
            try:
                filename = row['filename']
                
                # Check for potential path issues and try different variations
                potential_paths = [
                    os.path.join(audio_folder, filename),  # Standard path
                    os.path.join(audio_folder, os.path.basename(filename))  # Just filename without directories
                ]
                
                # If the filename has a class directory, also try with different path structures
                if '/' in filename:
                    class_dir, file = filename.split('/', 1)
                    potential_paths.append(os.path.join(audio_folder, class_dir, file))  # Nested structure
                    potential_paths.append(os.path.join(audio_folder, file))  # File without class dir
                
                # Try each potential path
                file_exists = False
                working_path = None
                
                for path in potential_paths:
                    if os.path.exists(path):
                        file_exists = True
                        working_path = path
                        break
                
                if file_exists:
                    valid_rows.append(idx)
                    # If the working path is different from the expected path, log this
                    expected_path = os.path.join(audio_folder, filename)
                    if working_path != expected_path:
                        print(f"ℹ️ File found at alternative path: {working_path}")
                        # Update the filename to match the working structure
                        relative_path = os.path.relpath(working_path, audio_folder)
                        df.at[idx, 'filename'] = relative_path
                else:
                    invalid_files.append(filename)
            except Exception as e:
                print(f"  Error checking file at index {idx}: {str(e)}")
                invalid_files.append(f"{row['filename']} (error: {str(e)})")
        
        if invalid_files:
            print(f"⚠️ Warning: {len(invalid_files)} files not found in {audio_folder}")
            # Safely show examples by directly accessing the list rather than using DataFrame indexing
            example_count = min(5, len(invalid_files))
            if example_count > 0:
                print(f"   Example missing files: {invalid_files[:example_count]}")
            print("\nPossible causes for missing files:")
            print("1. Files don't exist in the expected location")
            print("2. Path structure mismatch between CSV and actual files")
            print("3. Incorrect base audio_folder path")
            print("4. Class directories might be missing or named differently")
        
        print(f"→ {len(valid_rows)} valid files found out of {len(df)}")
        
        if not valid_rows:
            print("❌ No valid files found! Check your audio folder path and file structure.")
            return df.head(0)  # Return empty DataFrame with same structure
            
        # Only keep rows that have valid files
        return df.loc[valid_rows].reset_index(drop=True)

    def check_dataset_structure(self, audio_folder):
        """Check the structure of the dataset directory and suggest fixes."""
        print("\n=== Checking Dataset Structure ===")
        
        # Check if the audio folder exists
        if not os.path.exists(audio_folder):
            print(f"❌ Error: Audio folder not found at {audio_folder}")
            return False
        
        # List the contents of the audio folder
        try:
            contents = os.listdir(audio_folder)
            num_files = len([f for f in contents if os.path.isfile(os.path.join(audio_folder, f))])
            num_dirs = len([d for d in contents if os.path.isdir(os.path.join(audio_folder, d))])
            
            print(f"→ Audio folder contains {num_files} files and {num_dirs} directories")
            
            # Check for expected structure - expecting either audio files directly or in subdirectories
            if num_files == 0 and num_dirs == 0:
                print("❌ Error: Audio folder is empty")
                return False
            
            # Sample a few entries to check format
            sample_entries = random.sample(contents, min(5, len(contents)))
            print(f"→ Sample entries: {sample_entries}")
            
            # If we have directories, check if they contain audio files
            if num_dirs > 0:
                sample_dir = next((d for d in contents if os.path.isdir(os.path.join(audio_folder, d))), None)
                if sample_dir:
                    dir_contents = os.listdir(os.path.join(audio_folder, sample_dir))
                    print(f"→ Sample directory '{sample_dir}' contains {len(dir_contents)} files")
                    if len(dir_contents) > 0:
                        print(f"→ Sample files in '{sample_dir}': {dir_contents[:3]}")
            
            # Check for the train.csv file
            metadata_path = os.path.join(audio_folder, 'train.csv')
            if os.path.exists(metadata_path):
                print(f"✓ Found train.csv in {audio_folder}")
            else:
                print(f"❌ Warning: train.csv not found in {audio_folder}")
                # Check for other CSV files
                csv_files = [f for f in os.listdir(audio_folder) if f.endswith('.csv')]
                if csv_files:
                    print(f"→ Found other CSV files: {csv_files}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error checking dataset structure: {str(e)}")
            traceback.print_exc()
            return False

    def check_csv_structure(self, metadata_path):
        """Check the structure of the CSV file and ensure it has the correct format."""
        print("\n=== Checking CSV Structure ===")
        
        try:
            # Check if the file exists
            if not os.path.exists(metadata_path):
                print(f"❌ Error: CSV file not found at {metadata_path}")
                return None
            
            # Try to load the CSV file
            try:
                df = pd.read_csv(metadata_path)
                print(f"✓ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ Error loading CSV: {str(e)}")
                return None
            
            # Check required columns
            required_columns = ['filename']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Error: Missing required columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                return None
            
            # Check data format
            print(f"→ First 3 rows of the CSV:")
            print(df.head(3))
            
            # Check for null values in critical columns
            null_in_filename = df['filename'].isnull().sum()
            if null_in_filename > 0:
                print(f"⚠️ Warning: {null_in_filename} null values in 'filename' column")
                # Drop null values
                df = df.dropna(subset=['filename'])
                print(f"→ Dropped rows with null filenames. Remaining rows: {len(df)}")
            
            # Check filename format
            sample_filenames = df['filename'].head(5).tolist()
            print(f"→ Sample filenames: {sample_filenames}")
            
            # Check if filenames have expected format (typically class/filename.ext)
            format_issues = sum(1 for f in sample_filenames if '/' not in f)
            if format_issues > 0:
                print(f"⚠️ Warning: {format_issues} out of 5 sample filenames don't follow the 'class/filename.ext' format")
                print("→ This might cause issues with class label extraction")
                
                # Attempt to fix missing class directory structure in filenames
                if 'primary_label' in df.columns:
                    print("✓ Found 'primary_label' column - will use it to fix filenames")
                    df['filename'] = df.apply(lambda row: f"{row['primary_label']}/{row['filename']}" 
                                            if '/' not in row['filename'] else row['filename'], axis=1)
                    print("→ Fixed filenames by prepending class labels")
                    print("→ Sample fixed filenames:", df['filename'].head(5).tolist())
            
            return df
            
        except Exception as e:
            print(f"❌ Error checking CSV structure: {str(e)}")
            traceback.print_exc()
            return None

    def prepare_data(self, train_meta, val_split=0.2, batch_size=24):
        """Prepare training and validation data loaders."""
        print("\n=== Preparing Data ===")
        try:
            # Get the audio folder path
            audio_folder = "/kaggle/input/birdclef-2025-preprocessed-dataset-i/fixed_duration_5sec/"
            print(f"→ Using audio folder: {audio_folder}")
            
            # Check if audio folder exists
            if not os.path.exists(audio_folder):
                print(f"❌ Error: Audio folder not found at {audio_folder}")
                print("Checking parent directories...")
                
                # Try to find the correct directory
                parent_dir = "/kaggle/input/birdclef-2025-preprocessed-dataset-i/"
                if os.path.exists(parent_dir):
                    print(f"✓ Found parent directory: {parent_dir}")
                    contents = os.listdir(parent_dir)
                    print(f"Available directories: {[d for d in contents if os.path.isdir(os.path.join(parent_dir, d))]}")
                    print(f"Available files: {[f for f in contents if os.path.isfile(os.path.join(parent_dir, f))]}")
                    
                    # If we find a directory that might contain audio files, suggest it
                    audio_dirs = [d for d in contents if os.path.isdir(os.path.join(parent_dir, d))]
                    if audio_dirs:
                        print(f"ℹ️ Try using one of these directories instead: {audio_dirs}")
            
            # Check if train_meta is valid
            if train_meta is None or len(train_meta) == 0:
                raise ValueError("Invalid or empty metadata provided")
            
            print(f"→ Metadata contains {len(train_meta)} entries")
            
            # Check required columns exist
            if 'filename' not in train_meta.columns:
                raise ValueError("Metadata missing required 'filename' column")
                
            # Sample a few rows to verify format
            print("→ Sample rows from metadata:")
            print(train_meta.head(3))
            
            # Check dataset structure
            self.check_dataset_structure(audio_folder)
            
            # Split data into train and validation sets
            print(f"→ Splitting data with validation ratio: {val_split}")
            train_df, val_df = train_test_split(train_meta, test_size=val_split, random_state=42)
            print(f"→ Initial train set size: {len(train_df)}, Initial validation set size: {len(val_df)}")
            
            # Verify files exist and filter out those that don't
            train_df = self.verify_files(train_df, audio_folder)
            val_df = self.verify_files(val_df, audio_folder)
            print(f"→ Final train set size: {len(train_df)}, Final validation set size: {len(val_df)}")
            
            if len(train_df) == 0 or len(val_df) == 0:
                raise ValueError("Insufficient valid files found for training/validation. Check the audio_folder path.")
            
            if len(train_df) < batch_size or len(val_df) < batch_size:
                print(f"⚠️ Warning: Dataset is smaller than batch size ({batch_size})")
                # Adjust batch size if needed
                adjusted_batch_size = min(batch_size, min(len(train_df), len(val_df)))
                if adjusted_batch_size < batch_size:
                    print(f"→ Adjusting batch size from {batch_size} to {adjusted_batch_size}")
                    batch_size = adjusted_batch_size
            
            # Create datasets
            print("→ Creating datasets...")
            train_dataset = BirdclefDataset(
                train_df, 
                mode='train',
                audio_folder=audio_folder,
                class_labels=self.class_labels,
                verbose=0  # Minimal output during dataset creation
            )
            
            val_dataset = BirdclefDataset(
                val_df, 
                mode='train',  # Using 'train' mode to get labels
                audio_folder=audio_folder,
                class_labels=self.class_labels,
                verbose=0  # Minimal output during dataset creation
            )
            
            # Create data loaders
            print("→ Creating data loaders...")
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2, 
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=1, 
                drop_last=True
            )
            
            print("Data preparation completed successfully!")
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            traceback.print_exc()  # Print the full traceback
            raise

    def train(self, epochs=10, learning_rate=0.001):
        """Train the model."""
        print("\n=== Starting Training ===")
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            print(f"→ Using Adam optimizer with learning rate: {learning_rate}")
            
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-------------------")
                
                # Training phase
                self.model.train()
                running_loss = 0.0
                pred_train, label_train = [], []
                
                print("Training phase:")
                for batch_idx, (melspecs, labels) in enumerate(tqdm(self.train_loader)):
                    melspecs, labels = melspecs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(melspecs)
                    loss = criterion(outputs, labels.to(torch.float32))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    pred_train.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
                    label_train.append(labels.detach().cpu().numpy())
                
                # Validation phase
                print("\nValidation phase:")
                self.model.eval()
                running_loss_val = 0.0
                pred_val, label_val = [], []
                
                with torch.no_grad():
                    for melspecs, labels in tqdm(self.val_loader):
                        melspecs, labels = melspecs.to(self.device), labels.to(self.device)
                        outputs = self.model(melspecs)
                        loss = criterion(outputs, labels.to(torch.float32))
                        running_loss_val += loss.item()
                        
                        pred_val.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
                        label_val.append(labels.detach().cpu().numpy())
                
                # Calculate metrics
                train_loss = running_loss / len(self.train_loader)
                val_loss = running_loss_val / len(self.val_loader)
                train_auc = self.calculate_auc(label_train, pred_train)
                val_auc = self.calculate_auc(label_val, pred_val)
                
                # Store metrics
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_auc'].append(train_auc)
                self.history['val_auc'].append(val_auc)
                
                print(f"\nEpoch Summary:")
                print(f"→ Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"→ Train AUC: {train_auc:.2f}%, Val AUC: {val_auc:.2f}%")
            
            print("\nTraining completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def calculate_auc(self, labels, predictions):
        """Calculate AUC score."""
        try:
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            return roc_auc_score(labels, predictions, average='macro') * 100
        except Exception as e:
            print(f"Error calculating AUC: {str(e)}")
            return 0.0

    def plot_metrics(self):
        """Plot training metrics."""
        print("\n=== Plotting Training Metrics ===")
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot AUC scores
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_auc'], label='Train AUC')
            plt.plot(self.history['val_auc'], label='Validation AUC')
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC (%)')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            print("Metrics plotted successfully!")
            
        except Exception as e:
            print(f"Error plotting metrics: {str(e)}")
            raise

    def save_model(self, save_path):
        """Save the trained model."""
        print(f"\n=== Saving Model to {save_path} ===")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history,
                'class_labels': self.class_labels
            }, save_path)
            print("Model saved successfully!")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

# Example usage:
"""
# Define FFT parameters
fft_params = {
    'n_fft': 1024,
    'win_length': 1024,
    'hop_length': 512,
    'f_min': 20,
    'f_max': 15000,
    'n_mels': 128,
    'sample_rate': 32000
}

# Set paths
original_data_folder = "/kaggle/input/birdclef-2025/"
original_train_audio_folder = "/kaggle/input/birdclef-2025/train_audio/"
enhanced_train_audio_folder = "/kaggle/input/birdclef-2025-preprocessed-dataset-i/fixed_duration_5sec/"
enhanced_train_audio_csv = "/kaggle/input/birdclef-2025-preprocessed-dataset-i/fixed_duration_5sec/train.csv"

# Get class labels from original dataset
class_labels = sorted(os.listdir(original_train_audio_folder))
print(f"Found {len(class_labels)} class labels")

# Alternative 1: Use SpecTor to load and process data
try:
    # Create SpecTor instance
    train_processor = SpecTor(fft_params, sample=1.0, mode='train')
    
    # Initialize trainer with class labels from SpecTor
    trainer = ModelTrainer(class_labels=train_processor.class_labels)
    
    # Load model
    trainer.load_model(pretrained=True)
    
    # Prepare data using the metadata from SpecTor
    trainer.prepare_data(train_processor.meta, val_split=0.2, batch_size=24)
    
    # Train model
    trainer.train(epochs=10, learning_rate=0.001)
    
    # Plot metrics
    trainer.plot_metrics()
    
    # Save model
    trainer.save_model('enhanced_model_spector.pth')
    
except Exception as e:
    print(f"Error with Alternative 1: {str(e)}")
    traceback.print_exc()

# Alternative 2: Load CSV directly and use ModelTrainer
try:
    # Create trainer instance
    trainer = ModelTrainer(class_labels=class_labels)
    
    # Load metadata directly with structure check
    metadata_path = enhanced_train_audio_csv
    print(f"Checking CSV at: {metadata_path}")
    
    # Check CSV structure
    train_meta = trainer.check_csv_structure(metadata_path)
    
    if train_meta is not None and len(train_meta) > 0:
        print(f"Loaded metadata with {len(train_meta)} entries")
        
        # Load model
        trainer.load_model(pretrained=True)
        
        # Prepare data with the loaded metadata
        trainer.prepare_data(train_meta, val_split=0.2, batch_size=24)
        
        # Train model
        trainer.train(epochs=10, learning_rate=0.001)
        
        # Plot metrics
        trainer.plot_metrics()
        
        # Save model
        trainer.save_model('enhanced_model_direct.pth')
    else:
        print("❌ Failed to load valid metadata. Cannot continue.")
        
except Exception as e:
    print(f"Error with Alternative 2: {str(e)}")
    traceback.print_exc()
""" 


