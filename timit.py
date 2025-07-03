# -*- coding: utf-8 -*-
"""phoneme_recognition_fixed.py

Fixed version addressing major issues in the original code
"""
!pip install torchinfo
import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import kagglehub
import editdistance
from torchinfo import summary
import csv



# Step 1: Define phoneme mapping (TIMIT uses 61 phonemes, mapped to 39 for evaluation)
PHONE_MAP = {
    'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ah': 'ah', 'uw': 'uw', 'uh': 'uh',
    'aa': 'aa', 'ey': 'ey', 'ay': 'ay', 'oy': 'oy', 'aw': 'aw', 'ow': 'ow', 'er': 'er',
    'l': 'l', 'r': 'r', 'y': 'y', 'w': 'w', 'm': 'm', 'n': 'n', 'ng': 'ng', 'v': 'v',
    'f': 'f', 'dh': 'dh', 'th': 'th', 'z': 'z', 's': 's', 'zh': 'zh', 'sh': 'sh',
    'jh': 'jh', 'ch': 'ch', 'b': 'b', 'p': 'p', 'd': 'd', 't': 't', 'g': 'g', 'k': 'k',
    'dx': 't', 'nx': 'n', 'hv': 'hh', 'hh': 'hh', 'bcl': 'sil', 'pcl': 'sil',
    'tcl': 'sil', 'gcl': 'sil', 'kcl': 'sil', 'dcl': 'sil', 'q': 'sil', 'epi': 'sil',
    'pau': 'sil', 'h#': 'sil', '#h': 'sil'
}

PHONE_CLASSES = sorted(set(PHONE_MAP.values()))
PHONE_TO_IDX = {phone: idx for idx, phone in enumerate(PHONE_CLASSES)}
PHONE_TO_IDX['blank'] = len(PHONE_CLASSES)  # Blank class for CTC loss
NUM_PHONEMES = len(PHONE_CLASSES) + 1  # 39 phonemes after mapping + 1 blank = 40 total

# Step 2: Fixed TIMIT Dataset with proper frame alignment
class TIMITDataset(Dataset):
    def __init__(self, data_dir, split='TRAIN', transform=None, max_frames=None, global_mean=None, global_std=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_frames = max_frames
        self.file_list = []
        self.phoneme_labels = []
        self.load_timit()

        #---New---
        self.global_mean = global_mean
        self.global_std = global_std

    def load_timit(self):
        split_dir = os.path.join(self.data_dir, self.split)
        for dialect in os.listdir(split_dir):
            dialect_path = os.path.join(split_dir, dialect)
            if not os.path.isdir(dialect_path):
                continue
            for speaker in os.listdir(dialect_path):
                speaker_path = os.path.join(dialect_path, speaker)
                if not os.path.isdir(speaker_path):
                    continue
                for wav_file in os.listdir(speaker_path):
                    if wav_file.endswith('.WAV'):
                        base_name = os.path.splitext(wav_file)[0]
                        phn_file = os.path.join(speaker_path, base_name + '.PHN')
                        if os.path.exists(phn_file):
                            self.file_list.append(os.path.join(speaker_path, wav_file))
                            self.phoneme_labels.append(phn_file)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav_path = self.file_list[idx]
        phn_path = self.phoneme_labels[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.squeeze().numpy()

        # Extract MFCC features with proper parameters
        hop_length = int(sample_rate * 0.01)  # 10ms hop
        win_length = int(sample_rate * 0.025)  # 25ms window

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=51,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=512
        )
        mfcc = np.transpose(mfcc)

        # Compute delta and delta-delta features
        delta = librosa.feature.delta(mfcc, axis=0)
        delta_delta = librosa.feature.delta(mfcc, order=2, axis=0)

        # Concatenate features
        features = np.concatenate((mfcc, delta, delta_delta), axis=1)  # Shape: (frames, 153)


        #----New---- Normalization on all items
        if self.global_mean is not None and self.global_std is not None:
            features = (features - self.global_mean.numpy()) / (self.global_std.numpy() + 1e-8)
        else:
            # Fallback to utterance-level normalization
            features = (features - np.mean(features, axis=0, keepdims=True)) / (np.std(features, axis=0, keepdims=True) + 1e-8)


        # Load phoneme labels
        phonemes = []
        with open(phn_path, 'r') as f:
            for line in f:
                start, end, phoneme = line.strip().split()
                start, end = int(start), int(end)
                phoneme = PHONE_MAP.get(phoneme, 'sil')
                phonemes.append((start, end, phoneme))

        # Create frame-level phoneme sequence (NOT frame-level alignment)
        # For CTC, we need the phoneme sequence without repetitions
        phoneme_sequence = []
        for start, end, phoneme in phonemes:
            phoneme_sequence.append(PHONE_TO_IDX[phoneme])

        # Remove consecutive duplicates for CTC
        clean_sequence = []
        prev_phone = None
        for phone in phoneme_sequence:
            if phone != prev_phone:
                clean_sequence.append(phone)
                prev_phone = phone

        # Apply max_frames constraint if specified
        if self.max_frames and len(features) > self.max_frames:
            features = features[:self.max_frames]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(clean_sequence, dtype=torch.long)

        return features, labels


# Compute Mean for normalization:
def compute_global_cmvn(dataset):
    total_sum = 0
    total_square_sum = 0
    total_frames = 0

    for features, _ in tqdm(dataset, desc="Computing CMVN"):
        # features shape: [T, F]
        total_sum += features.sum(dim=0)
        total_square_sum += (features ** 2).sum(dim=0)
        total_frames += features.shape[0]

    mean = total_sum / total_frames
    std = torch.sqrt(total_square_sum / total_frames - mean ** 2 + 1e-8)
    return mean, std

# Step 3: Simplified BiLSTM Model (better than LSTM-P for this task)
class LSTMPhonemeRecognizer(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_layers, num_classes, dropout=0.2):
        super(LSTMPhonemeRecognizer, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.classifier = nn.Linear(proj_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (batch_size, seq_len, num_classes)
        return logits

# Step 4: Custom collate function for variable length sequences
def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    features, labels = zip(*batch)

    # Get sequence lengths
    feature_lengths = [len(f) for f in features]
    label_lengths = [len(l) for l in labels]

    # Pad features
    max_feature_len = max(feature_lengths)
    padded_features = []
    for f in features:
        pad_len = max_feature_len - len(f)
        if pad_len > 0:
            padded_f = torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)
        else:
            padded_f = f
        padded_features.append(padded_f)

    # Stack features
    features_tensor = torch.stack(padded_features)

    # Concatenate labels (required for CTC)
    labels_tensor = torch.cat(labels)

    return features_tensor, labels_tensor, torch.tensor(feature_lengths), torch.tensor(label_lengths)

# Step 5: Fixed Training Function
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=10, patience=10):
    model.train()
    best_per = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels, feature_lengths, label_lengths = batch_data
            features = features.to(device)
            labels = labels.to(device)
            feature_lengths = feature_lengths.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(features)  # (batch_size, seq_len, num_classes)

            # Prepare for CTC loss
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs = log_probs.transpose(0, 1)  # (seq_len, batch_size, num_classes)

            # Compute CTC loss
            loss = criterion(log_probs, labels, feature_lengths, label_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

        if epoch%3 == 0:
            per = evaluate_model(model, test_loader, device)

            #----New----
            scheduler.step(per) #DO NOT PASS PER IF USING StepLR!!!

            # Save best model
            if per < best_per:
                best_per = per
                torch.save(model.state_dict(), 'best_model.pt')
                print(f'Saved best model with PER: {best_per:.2f}%')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 3

            # Early stopping
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break
            #----New----
            model.train()

    return best_per




# Step 6: Fixed Evaluation Function
def ctc_decode_greedy(log_probs, lengths, blank_idx):
    """Simple greedy CTC decoding"""
    batch_size, seq_len, num_classes = log_probs.shape
    decoded_sequences = []

    for b in range(batch_size):
        seq_len_b = lengths[b]
        # Get most likely path
        best_path = torch.argmax(log_probs[b, :seq_len_b], dim=1)

        # CTC decode: remove blanks and collapse repeats
        decoded = []
        prev_token = None

        for token in best_path:
            token = token.item()
            if token != blank_idx and token != prev_token:
                decoded.append(token)
            prev_token = token

        decoded_sequences.append(decoded)

    return decoded_sequences

# Step 6: Optimized CTC Beam Search Decoder
def ctc_decode_beam(log_probs, lengths, blank_idx, beam_width=5, prob_threshold=-20.0):
    """Optimized CTC beam search decoding without external libraries.

    Args:
        log_probs: (batch_size, seq_len, num_classes) tensor of log probabilities
        lengths: (batch_size,) tensor of sequence lengths
        blank_idx: Index of the blank token
        beam_width: Number of beams to keep at each step
        prob_threshold: Log probability threshold to prune beams (default: -20.0)

    Returns:
        List of lists, where each inner list is a decoded sequence of token indices
    """
    batch_size, seq_len, num_classes = log_probs.shape
    decoded_sequences = []

    for b in range(batch_size):
        seq_len_b = lengths[b].item()
        log_probs_b = log_probs[b, :seq_len_b].cpu().numpy()  # (seq_len, num_classes)

        # Initialize beams: (sequence, log_prob)
        beams = [(tuple(), 0.0)]
        for t in range(seq_len_b):
            curr_probs = log_probs_b[t]  # (num_classes,)
            new_beams = []

            # Process each beam
            for prefix, prefix_prob in beams:
                # Compute probabilities for all extensions
                ext_probs = prefix_prob + curr_probs  # Vectorized: (num_classes,)

                # Add extensions to new_beams
                for c in range(num_classes):
                    new_prob = ext_probs[c]
                    if c == blank_idx:
                        new_beams.append((prefix, new_prob))
                    else:
                        new_prefix = prefix + (c,)
                        if prefix and prefix[-1] == c:
                            new_beams.append((prefix, new_prob))  # Extend with blank
                            new_beams.append((new_prefix, new_prob))  # Extend with repeat
                        else:
                            new_beams.append((new_prefix, new_prob))

            # Prune to top beam_width beams, ensuring at least one survives
            if new_beams:
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
                if len(beams) > beam_width:
                    # Apply threshold only if enough beams exist
                    beams = [b for b in beams if b[1] >= beams[0][1] + prob_threshold][:beam_width]
                else:
                    beams = beams[:beam_width]
            else:
                # Fallback: add the best extension to prevent empty beams
                best_c = np.argmax(curr_probs)
                beams = [(beams[0][0] + (best_c,) if best_c != blank_idx else beams[0][0], beams[0][1] + curr_probs[best_c])]

        # Select best beam and collapse CTC output
        best_prefix, _ = max(beams, key=lambda x: x[1])
        decoded = []
        prev_token = None
        for token in best_prefix:
            if token != blank_idx and token != prev_token:
                decoded.append(token)
            prev_token = token
        decoded_sequences.append(decoded)

    return decoded_sequences

def evaluate_model(model, test_loader, device):
    model.eval()
    total_per = 0
    num_sequences = 0
    blank_idx = PHONE_TO_IDX['blank']

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Evaluating'):
            features, labels, feature_lengths, label_lengths = batch_data
            features = features.to(device)

            # Forward pass
            logits = model(features)
            log_probs = torch.log_softmax(logits, dim=2)

            # Decode predictions
            # decoded_preds = ctc_decode_beam(log_probs, feature_lengths, blank_idx, beam_width=10)
            decoded_preds = ctc_decode_greedy(log_probs, feature_lengths, blank_idx)

            # Convert concatenated labels back to sequences
            label_start = 0
            for i, label_len in enumerate(label_lengths):
                pred_seq = decoded_preds[i]
                true_seq = labels[label_start:label_start + label_len].tolist()

                # Compute PER
                per = compute_per(true_seq, pred_seq)
                total_per += per
                num_sequences += 1

                label_start += label_len

    avg_per = total_per / num_sequences if num_sequences > 0 else 100.0
    print(f'Average Phoneme Error Rate (PER): {avg_per:.2f}%')
    return avg_per

def compute_per(reference, hypothesis):
    """Compute Phoneme Error Rate using edit distance"""
    if len(reference) == 0:
        return 100.0 if len(hypothesis) > 0 else 0.0

    edit_dist = editdistance.eval(reference, hypothesis)
    per = (edit_dist / len(reference)) * 100
    return per

def store_test_data(test_dataset, batch_size=10):

    # Load the whole dataset in one batch
    full_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,  # One big batch
        shuffle = False,
        collate_fn = collate_fn     # Your padding collate function
    )

    # Get the batch
    inputs, labels, input_lengths, label_lengths = next(iter(full_loader))

    # Save features to CSV (one frame per row)
    with open("features.csv", mode="w", newline="") as f_feat:
        writer_feat = csv.writer(f_feat)
        for sample in inputs:  # sample shape: [T, F]
            for frame in sample:
                writer_feat.writerow(frame.tolist())

    # Save labels to CSV (one label per row, blank line between sequences)
    with open("labels.csv", mode="w", newline="") as f_lbl:
        writer_lbl = csv.writer(f_lbl)
        idx = 0
        for length in label_lengths:
            length = length.item()
            for i in range(length):
                writer_lbl.writerow([labels[idx + i].item()])
            idx += length
            writer_lbl.writerow([])  # Blank line to separate label sequences

def save_tensor_to_csv(tensor, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in tensor.detach().cpu().numpy():
            writer.writerow(row)

def save_model_weights(model, out_dir="lstm_weights"):
    os.makedirs(out_dir, exist_ok=True)

    lstm = model.lstm
    proj_size = lstm.proj_size
    num_layers = lstm.num_layers

    for layer in range(num_layers):
        prefix = f'_l{layer}'

        # Extract gate weights
        weight_ih = getattr(lstm, f'weight_ih{prefix}')
        weight_hh = getattr(lstm, f'weight_hh{prefix}')
        bias_ih = getattr(lstm, f'bias_ih{prefix}')
        bias_hh = getattr(lstm, f'bias_hh{prefix}')

        # Split gates
        gates = ['i', 'f', 'g', 'o']
        w_ih_chunks = torch.chunk(weight_ih, 4, dim=0)
        w_hh_chunks = torch.chunk(weight_hh, 4, dim=0)
        b_ih_chunks = torch.chunk(bias_ih, 4, dim=0)
        b_hh_chunks = torch.chunk(bias_hh, 4, dim=0)

        for g, w_ih, w_hh, b_ih, b_hh in zip(gates, w_ih_chunks, w_hh_chunks, b_ih_chunks, b_hh_chunks):
            save_tensor_to_csv(w_ih, f"{out_dir}/layer{layer}_weight_ih_{g}.csv")
            save_tensor_to_csv(w_hh, f"{out_dir}/layer{layer}_weight_hh_{g}.csv")
            print(f"Input weight shape {g}:", w_ih.shape)
            print(f"Hidden weight shape {g}:", w_hh.shape)

            # Summed bias
            bias_sum = b_ih + b_hh
            print(f"Bias shape {g}:", bias_sum.shape)
            save_tensor_to_csv(bias_sum.unsqueeze(0), f"{out_dir}/layer{layer}_bias_{g}.csv")

        # Save projection matrix if proj_size > 0
        if proj_size > 0:
            weight_hr = getattr(lstm, f'weight_hr{prefix}')
            print("Projection weight shape:", weight_hr.shape)
            save_tensor_to_csv(weight_hr, f"{out_dir}/layer{layer}_weight_proj.csv")


    # Save linear classifier weights and bias
    print("Classifier weight shape:", model.classifier.weight.shape)
    save_tensor_to_csv(model.classifier.weight, f"{out_dir}/classifier_weight.csv")
    save_tensor_to_csv(model.classifier.bias.unsqueeze(0), f"{out_dir}/classifier_bias.csv")

    print(f"Saved all weights to '{out_dir}/'")


# Step 7: Main Execution

# Configuration
data_dir = kagglehub.dataset_download("mfekadu/darpa-timit-acousticphonetic-continuous-speech") + "/data/"
batch_size = 16  # Reduced batch size
num_epochs = 60
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Number of phoneme classes: {NUM_PHONEMES}")

# Load datasets
train_dataset = TIMITDataset(data_dir, split='TRAIN')
test_dataset = TIMITDataset(data_dir, split='TEST')


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# store_test_data(test_dataset, 20)


print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Initialize model
model = LSTMPhonemeRecognizer(
    input_size=51*3,  # 51 MFCC + 51 delta + 51 delta-delta
    hidden_size=1024,
    proj_size=512,
    num_layers=1,
    num_classes=NUM_PHONEMES,
    dropout=0.2
).to(device)

print("Model Summary:")
summary(model, input_size=(batch_size, 300, 153))

# Calculate and print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nParameter Count:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# Loss and optimizer
criterion = nn.CTCLoss(blank=PHONE_TO_IDX['blank'], reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-5

# Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

print("Starting training...")
best_per = train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs, patience=15)

print("Loading best model for weight saving...")
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt'))
    print("Best model loaded successfully")
else:
    print("Warning: best_model.pt not found, saving weights from final model state")


print("Evaluating model...")
per = evaluate_model(model, test_loader, device)

save_model_weights(model, out_dir="lstm_weights_best")
