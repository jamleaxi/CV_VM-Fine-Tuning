import os
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
import evaluate

# 1. ENVIRONMENT CONFIGURATION & VRAM GUARDRAILS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Core Low-Compute Hyperparameters
BATCH_SIZE = 4            # Low batch size guarantees no 8GB Out-Of-Memory (OOM) errors
IMAGE_RESOLUTION = 384    # TrOCR default resolution
EPOCHS = 5                # Small epochs for a mini-lab demonstration

# Use a versioned run directory so each fine-tuned model is preserved separately.
RUN_VERSION = datetime.now().strftime("v%Y.%m.%d_%H%M%S")
BASE_OUTPUT_DIR = "./final_custom_trocr_model"
VERSIONED_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_VERSION)
os.makedirs(VERSIONED_OUTPUT_DIR, exist_ok=True)

# Load the standard Character Error Rate (CER) metric via Hugging Face Evaluate
cer_metric = evaluate.load("cer")

# 2. CUSTOM DATASET LOADER AND METRICS COMPUTATION
class HandWrittenDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, max_target_length=128):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['file_name'])
        image = Image.open(img_name).convert("RGB")
        
        # Preprocess image to normalized pixel values tensor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Preprocess text to token IDs
        text = self.df.iloc[idx]['text']
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length, 
                                          truncation=True).input_ids
        
        # Replace padding token id by -100 so PyTorch's CrossEntropyLoss ignores it
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Unwrap tuple output (e.g., when model returns extra tensors alongside sequences)
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # Replace any negative values (e.g., -100 padding) before decoding to avoid OverflowError
    pred_ids = np.where(pred_ids >= 0, pred_ids, processor.tokenizer.pad_token_id)

    # 1. Decode predicted token IDs back into text strings
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    # 2. Replace the -100 padding tokens we used earlier back to the pad_token_id 
    # so the tokenizer can decode the ground truth correctly
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # 3. Compute the Character Error Rate
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # You can return multiple metrics here (like word error rate) if needed
    return {"cer": cer}

# 3. INITIALIZE SMALL MODEL & PROCESSOR
# trocr-small uses a tiny DeiT encoder and a tiny UniLM text decoder (~60M total parameters)
model_checkpoint = "microsoft/trocr-small-handwritten"
processor = TrOCRProcessor.from_pretrained(model_checkpoint)

if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint)

    # Configure vocabulary token settings necessary for sequence generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # 4. INSTANTIATE DATASETS
    # Split a 100-line CSV into 80% train / 20% test
    train_dataset = HandWrittenDataset(csv_file="metadata/train_metadata.csv", img_dir="data/train_crops", processor=processor)
    eval_dataset = HandWrittenDataset(csv_file="metadata/test_metadata.csv", img_dir="data/test_crops", processor=processor)

    # 5. DEFINE TRAINING ARGUMENTS FOR 8GB VRAM
    training_args = Seq2SeqTrainingArguments(
        output_dir=VERSIONED_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        fp16=torch.cuda.is_available(), # Crucial: Uses half-precision execution to slash VRAM usage by ~50%
        dataloader_num_workers=2,       # Keeps CPU data pipelining efficient without choke
        report_to="none"                # Shuts off third-party logging overheads
    )

    # 6. INITIALIZE TRAINING LOOP
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. EXECUTE FINE-TUNING
    print("--- Starting Fine-Tuning Execution ---")
    train_result = trainer.train()

    # --- SAVE THE EVALUATION METRICS HISTOGRAM ---
    # This extracts the logs (loss, learning rate, CER per epoch)
    history = trainer.state.log_history

    # Save to a JSON file so students can parse it for graphs
    log_file_path = os.path.join(VERSIONED_OUTPUT_DIR, "training_metrics.json")
    with open(log_file_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Training metrics history successfully saved to {log_file_path}!")

    # Save the final evaluation results and the CER history to a versioned JSON file.
    evaluation_results = trainer.evaluate()
    cer_history = [
        {
            "epoch": entry.get("epoch"),
            "eval_cer": entry.get("eval_cer"),
        }
        for entry in history
        if "eval_cer" in entry
    ]

    cer_results_path = os.path.join(VERSIONED_OUTPUT_DIR, "cer_results.json")
    with open(cer_results_path, "w") as f:
        json.dump(
            {
                "run_version": RUN_VERSION,
                "output_dir": VERSIONED_OUTPUT_DIR,
                "final_evaluation": evaluation_results,
                "cer_history": cer_history,
            },
            f,
            indent=4,
        )

    print(f"CER results successfully saved to {cer_results_path}!")

    # 8. SAVE CHECKPOINT
    model.save_pretrained(VERSIONED_OUTPUT_DIR)
    processor.save_pretrained(VERSIONED_OUTPUT_DIR)
    print(f"Model fine-tuned successfully and saved locally to {VERSIONED_OUTPUT_DIR}!")