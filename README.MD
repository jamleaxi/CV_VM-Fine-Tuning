# Handwritten Text Recognition Fine-Tuning

This repository contains a small fine-tuning workflow for handwritten text recognition using Hugging Face TrOCR.

The main script, `finetune_htr.py`, loads a pre-trained handwritten TrOCR checkpoint, prepares image-text pairs from CSV metadata, computes Character Error Rate (CER), and saves the fine-tuned model locally.

## Overview

- Model: `microsoft/trocr-small-handwritten`
- Frameworks: PyTorch, Transformers, Pandas, Pillow, Evaluate
- Output folder: `./final_custom_trocr_model/<versioned_run>/`
- Training logs: `./final_custom_trocr_model/<versioned_run>/training_metrics.json`

## Expected Data Layout

The script expects the following files and folders:

- `metadata/train_metadata.csv`
- `metadata/test_metadata.csv`
- `data/train_crops/`
- `data/test_crops/`

Each CSV should contain at least these columns:

- `file_name`: image filename relative to the corresponding image folder
- `text`: ground-truth transcription for the image

## Setup

Install the required Python packages in your environment:

```bash
pip install -r requirements.txt
```

If you are using a CUDA-enabled system, install the appropriate PyTorch build for your platform first.

## Training

Run the fine-tuning script from the repository root:

```bash
python finetune_htr.py
```

The script uses a small batch size and mixed precision when CUDA is available, which helps keep memory usage low on consumer GPUs.

## Outputs

After training completes, the script writes:

- Fine-tuned model weights and processor files to a timestamped subdirectory under `./final_custom_trocr_model/`
- Training history to `./final_custom_trocr_model/<versioned_run>/training_metrics.json`
- CER evaluation results to `./final_custom_trocr_model/<versioned_run>/cer_results.json`

## Citation

If you use this repository in your work, please cite it as follows:

```bibtex
@misc{emberda_handwritten_text_recognition_2026,
	title        = {Handwritten Text Recognition Fine-Tuning with TrOCR},
	author       = {Emberda, Eric John},
	year         = {2026},
	howpublished = {GitHub repository},
	note         = {Accessed: 2026-05-29},
	url          = {https://github.com/ericjohnemberda/handwritten_text_recognition}
}
```

## License

See [LICENSE](LICENSE) for the project license.