from datasets import Dataset, Audio
import librosa
import os
import json
import soundfile as sf
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from datasets import DatasetDict

# Load the necessary models and processors
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="korean", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="korean", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")


# Define functions for audio processing
def load_audio(file_path):
    # Add a print statement to confirm the file being loaded
    print(f"Loading audio file: {file_path}")
    waveform, _ = librosa.load(file_path, sr=16000)
    return waveform

def save_waveform(waveform, file_path):
    # Add a print statement to confirm the file being saved
    print(f"Saving waveform to: {file_path}")
    sf.write(file_path, waveform, 16000)

def load_sentences_from_json(json_file):
    # Add a print statement to confirm the JSON file being loaded
    print(f"Loading sentences from JSON: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return ' '.join([utterance.get('standard_form', '') for utterance in json_data.get('utterance', [])])


# Process files and create dataset entries
directory_path = "D:\\Finetune whisper"  #Replace with your own path
data_entries = []

for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):
        audio_path = os.path.join(directory_path, filename)
        json_path = os.path.splitext(audio_path)[0] + ".json"
        if os.path.isfile(json_path):
            sentence = load_sentences_from_json(json_path)
            data_entries.append({"audio": audio_path, "sentence": sentence})

if not data_entries:
    raise ValueError("No data entries found. Check your dataset and file paths.")

# Create dataset
dataset = Dataset.from_dict({
    "audio": [entry["audio"] for entry in data_entries],
    "sentence": [entry["sentence"] for entry in data_entries]
}).cast_column("audio", Audio(sampling_rate=16000))

def train_test_split(dataset: Dataset, train_ratio=0.8, seed=42):
    dataset_length = len(dataset)
    train_size = int(train_ratio * dataset_length)
    test_size = dataset_length - train_size

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)

    # Split the dataset into train and test sets
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, dataset_length))

    # Return as a DatasetDict for better handling in Hugging Face's Trainer
    split_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return split_datasets

# Use the function to split the dataset
split_datasets = train_test_split(dataset, train_ratio=0.8, seed=42)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']
print("Train dataset length:", len(train_dataset))
print("Test dataset length:", len(test_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, drop_last=True)
print('Dataset structure:', dataset)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# Validate dataset split
if len(train_dataset) == 0 or len(test_dataset) == 0:
    raise ValueError("Train or Test dataset is empty after splitting. Check your dataset.")

# Dataset processing functions
def prepare_dataset(batch):
    try:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids[:448]
        return batch
    except Exception as e:
        print(f"Error in processing: {e}")
        return None

# Adjustments for the DataLoader and the dataset processing
batch_size = 8  # Adjust as needed

# Apply the processing function and filter out None values
dataset = dataset.map(prepare_dataset)
dataset = dataset.filter(lambda x: x["input_features"] is not None and x["labels"] is not None)

# Splitting the dataset
split_datasets = train_test_split(dataset, train_ratio=0.8, seed=42)
train_dataset = split_datasets['train']
test_dataset = split_datasets['test']

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

print('dataset after prepare',dataset)
# Filter out None values if any errors occurred during processing
dataset = dataset.filter(lambda x: x is not None)
print(dataset)
from datasets import load_metric

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
wer_metric = load_metric("wer")
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        """
        Data collator that dynamically pads the batch so that all data 
        samples have the same size.

        Args:
            processor: A processor that combines a feature extractor 
                       and a tokenizer, used to process audio and text data.
        """
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of features (samples) into a batch for training or evaluation.

        Args:
            features: A list of dictionaries, each containing 'input_features' and 'labels'.

        Returns:
            A dictionary with batched 'input_features' and 'labels'.
        """
        # Pad the input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Create a mask for the labels
        labels = labels_batch["input_ids"]
        labels_attention_mask = labels_batch["attention_mask"]

        # Replace padding token id's with -100 so they are ignored in the loss computation
        labels[labels_attention_mask == 0] = -100

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Flatten the lists
    labels = [label for sublist in labels for label in sublist]
    preds = [pred for sublist in preds for pred in sublist]
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')
    wer = wer_metric.compute(predictions=[preds], references=[labels])
    return {"accuracy": accuracy, "f1": f1, "wer": wer}


# Define a custom callback for logging
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Check if the logs contain evaluation metrics
        if logs is not None and 'eval_loss' in logs:
            print(f"Epoch: {state.epoch}, Step: {state.global_step}")
            print(f"Eval Loss: {logs['eval_loss']}")
            if 'eval_accuracy' in logs:
                print(f"Accuracy: {logs['eval_accuracy']}")
            if 'eval_f1' in logs:
                print(f"F1 Score: {logs['eval_f1']}")
            if 'eval_wer' in logs:
                print(f"WER: {logs['eval_wer']}")
# Training configuration
training_args = TrainingArguments(
    output_dir="./whisper_model",
    num_train_epochs=24, 
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    learning_rate=1e-3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps", 
    save_strategy="steps",  
    save_steps=500,  
    load_best_model_at_end=True, 
)

# Early Stopping
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

try:
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}: {batch}")
        if i >= 5: 
            break
except IndexError as e:
    print(f"IndexError caught: {e}")

trainer.train()
print("Training complete!")


model.save_pretrained("./whisper_model_trained-large-v2")
