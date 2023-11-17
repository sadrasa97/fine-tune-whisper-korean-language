from datasets import Dataset, Audio
import librosa
import os
import json
import soundfile as sf
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, Wav2Vec2Processor,
                          WhisperProcessor, WhisperForConditionalGeneration, 
                          TrainingArguments,Seq2SeqTrainingArguments, Seq2SeqTrainer)
import evaluate
metric = evaluate.load("wer")
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
# Load the necessary models and processors
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="korean", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="korean", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

# Function to load audio from a file
def load_audio(file_path):
    waveform, _ = librosa.load(file_path, sr=16000)
    return waveform

# Function to save a waveform to a file
def save_waveform(waveform, file_path):
    sf.write(file_path, waveform, 16000)

# Function to segment an audio waveform
def segment_audio(waveform, segment_length=30):
    rate = 16000  # Sample rate for librosa
    segment_samples = segment_length * rate
    return [waveform[i:i + segment_samples] for i in range(0, len(waveform), segment_samples)]

# Function to process each audio file and save segments
def process_and_save_segments(audio_file, json_file, directory_path):
    waveform = load_audio(audio_file)
    segments = segment_audio(waveform)
    segment_files = []
    for i, segment in enumerate(segments):
        segment_file_path = f"{directory_path}\\segment_{i}_{os.path.basename(audio_file)}"
        save_waveform(segment, segment_file_path)
        segment_files.append(segment_file_path)
    sentence = load_sentences_from_json(json_file)
    return segment_files, sentence

# Function to load sentences from a JSON file
def load_sentences_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    sentences = [utterance.get('standard_form', 'No Standard Form') for utterance in json_data.get('utterance', [])]
    return '\n'.join(sentences)
MAX_TOKEN_LENGTH = 2500
# Define file paths and process them
directory_path = "C:\\Users\\sadra\\Desktop\\whisfine"
audio_files = []
json_files = []
for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):
        audio_files.append(os.path.join(directory_path, filename))
    elif filename.endswith(".json"):
        json_files.append(os.path.join(directory_path, filename)) 

# Process files and create dataset entries
data_entries = []
for audio_file, json_file in zip(audio_files, json_files):
    if os.path.isfile(audio_file) and os.path.isfile(json_file):
        segment_files, sentence = process_and_save_segments(audio_file, json_file, directory_path)
        for segment_file in segment_files:
            data_entries.append({"audio": segment_file, "sentence": sentence})

# Create the dataset with paths to audio segments
dataset = Dataset.from_dict({"audio": [entry["audio"] for entry in data_entries],
                             "sentence": [entry["sentence"] for entry in data_entries]})

# Cast the 'audio' column to Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Function to split dataset into training and testing sets
def train_test_split(dataset: Dataset, train_ratio=0.8, seed=42):
    dataset_length = len(dataset)
    train_size = int(train_ratio * dataset_length)
    test_size = dataset_length - train_size
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, dataset_length))
    return train_dataset, test_dataset

train_dataset, test_dataset = train_test_split(dataset, train_ratio=0.8, seed=42)
print("Train dataset length:", len(train_dataset))
print("Test dataset length:", len(test_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, drop_last=True)
print('Dataset structure:', dataset)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    print("Processing audio:", audio)
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    encoded_text = tokenizer(batch["sentence"], truncation=True, max_length=MAX_TOKEN_LENGTH).input_ids
    print("Processed text:", batch["sentence"])
    batch["labels"] = encoded_text
    return batch

dataset = dataset.map(prepare_dataset)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)


        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("Data collator initialized with processor:", processor)    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = TrainingArguments(
    output_dir="./whisper_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir='./logs',
    logging_steps=10,
)

# Custom training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
MAX_MODEL_INPUT_LENGTH = 2500

# Custom training loop
for epoch in range(training_args.num_train_epochs):
    for batch in train_dataset:
        # Process the audio
        processed_audio = feature_extractor(batch["audio"]["array"], 
                                           return_tensors="pt", 
                                           padding=True, 
                                           sampling_rate=16000,
                                           truncation=True,
                                           max_length=MAX_MODEL_INPUT_LENGTH)
        
        input_features = processed_audio['input_features']

        #Print the shape of input features
        input_features_reshaped = input_features.reshape(1, -1)
        # Process the labels
        labels = tokenizer(batch["sentence"], 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True,
                           max_length=MAX_MODEL_INPUT_LENGTH).input_ids
        # Forward pass through the model
        try:

            outputs = model(input_features=input_features_reshaped, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:

            break  


print("Training complete!")

