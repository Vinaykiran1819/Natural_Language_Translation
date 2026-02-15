<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

# Language Translation with Transformer

A production-ready **Neural Machine Translation (NMT)** system built from scratch using the **Transformer** architecture. This project demonstrates end-to-end machine translation from raw parallel text to trained model and inference—supporting any source-to-target language pair.

The default configuration trains an **English → Italian** model on the OPUS Books dataset. The codebase is designed to be language-agnostic: swapping the dataset and updating configuration lets you train translation models for any language pair.

> **Citation:** This implementation is based on [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Using Different Languages](#using-different-languages)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

---

## Features

| Feature | Description |
|---------|-------------|
| **Transformer Architecture** | Full encoder-decoder with multi-head self-attention and cross-attention |
| **GPU-Accelerated Training** | Optimized for CUDA; falls back to CPU when unavailable |
| **Language-Agnostic** | Supports any source - target language pair via configurable datasets |
| **Multiple Inference Modes** | CLI, interactive script, Streamlit web app, and Jupyter notebook |
| **Automatic Tokenizers** | Word-level tokenizers built from training data on first run |
| **Checkpointing & Resumption** | Per-epoch saves; best model by BLEU; resume from any checkpoint |
| **Decoding Strategies** | Greedy, Top-k, and Top-p (nucleus) sampling for diverse outputs |

---

## Architecture Overview

The model follows the original **Transformer** (Vaswani et al., 2017) design:

| Component | Description |
|-----------|-------------|
| **Encoder** | 6 stacked blocks, each with Multi-Head Self-Attention + Feed-Forward Network (Residual + Layer Norm) |
| **Decoder** | 6 stacked blocks with Self-Attention, Cross-Attention (encoder-decoder), and Feed-Forward Network |
| **Embeddings** | Word-level embeddings + sinusoidal positional encoding |
| **Projection** | Linear layer + log-softmax to target vocabulary |

**Default hyperparameters:** `d_model=512`, `h=8` heads, `N=6` layers, `d_ff=2048`, `seq_len=350`.

---

## Example Output

```
Enter English sentence (or 'exit'): Hello, how are you today?

English: Hello, how are you today?
Italian: Ciao, come stai oggi? (95.2% 87.1% 92.3% ...)
==================================================
```

---

## Project Workflow

### Model Perspective

```
Input Text → Tokenizer → [SOS] + tokens + [EOS] + [PAD]
    ↓
Source Embedding + Positional Encoding
    ↓
Encoder (Self-Attention + FFN) × N → Encoder Output
    ↓
Decoder (Self-Attention + Cross-Attention + FFN) × N
    ↓
Projection Layer → Log-Softmax → Next-Token Prediction
    ↓
Iterative Decoding (Greedy / Top-k / Top-p) → Output Text
```

### Dataset Perspective

```
Raw Parallel Dataset (e.g., OPUS Books)
    ↓
Train/Validation Split (90% / 10%)
    ↓
Build or Load Tokenizers (source & target)
    ↓
BilingualDataset: (src_text, tgt_text) → (encoder_input, decoder_input, labels, masks)
    ↓
DataLoader (batched, shuffled)
    ↓
Training Loop → Validation (BLEU, CER, WER) → Checkpoints
```

---

## Installation

### Prerequisites

- **Python** 3.8+
- **PyTorch** 1.9+ (CUDA 11+ recommended for GPU training)
- **GPU** (optional): ~8GB VRAM for default `batch_size=8`; reduce to 4 for smaller GPUs

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/My_Transformer.git
cd My_Transformer

# Create a virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install PyTorch (choose one)
# With CUDA 11.8:
pip install torch torchvision torchaudio

# CPU-only (slower training):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

---

## Quick Start

> **First time?** Run [Training](#training) first to produce checkpoints and tokenizers. Inference requires these artifacts.

### Option A: Inference (after training)

**CLI** — Auto-loads the latest checkpoint in `weights/`:

```bash
python translate.py
```

Enter an English sentence when prompted. Type `exit` to quit.

**Web app** — Auto-loads the latest checkpoint in `weights/`:

```bash
streamlit run app.py
```

**Jupyter** — Edit `inference.ipynb` to point to your checkpoint, then run the cells.

### Option B: Full pipeline (train → infer)

```bash
# 1. Train (downloads data, builds tokenizers, saves to weights/)
python train.py

# 2. Translate
python translate.py
```

---

## Training

Training is GPU-aware: it uses CUDA when available and falls back to CPU otherwise.

### Dataset

The default setup uses **OPUS Books** (`opus_books`) from Hugging Face—a parallel corpus of books translated between language pairs. The `en-it` split provides English–Italian sentence pairs. The dataset is downloaded automatically on first run.

### Basic training (English → Italian)

```bash
python train.py
```

This will:

1. Download the OPUS Books `en-it` dataset via Hugging Face
2. Build or load `tokenizer_en.json` and `tokenizer_it.json` in the project root
3. Train for 20 epochs (configurable in `config.py`)
4. Save checkpoints to `weights/tmodel_00.pth`, `tmodel_01.pth`, ...
5. Save the best model (by BLEU) as `weights/tmodel_BEST_MODEL.pth`
6. Log loss and validation metrics to TensorBoard

### Resume from checkpoint

Edit `config.py`:

```python
"preload": "latest"   # resume from most recent checkpoint
# or
"preload": "10"       # resume from epoch 10 (tmodel_10.pth)
```

### View training logs

```bash
tensorboard --logdir runs/tmodel_GCP_v1
```

Open http://localhost:6006 to monitor loss, BLEU, CER, and WER.

---

## Inference

### CLI (translate.py)

```bash
python translate.py
```

Supports **Top-k** and **Top-p (nucleus)** sampling for more diverse outputs. Modify the call inside the script:

```python
translate(sentence, top_k=50, top_p=0.9, temperature=0.8)
```

### Web app (Streamlit)

```bash
streamlit run app.py
```

### Programmatic usage

```python
from translate import translate
translate("Hello, how are you?", top_k=50, top_p=0.9, temperature=0.8)
```

### Decoding strategies

| Strategy | Use case |
|----------|----------|
| **Greedy** | Deterministic, fast (used in `app.py` and validation) |
| **Top-k + Top-p** | More diverse outputs (used in `translate.py`) |
| **Temperature** | Lower = more conservative; higher = more random |

---

## Evaluation Metrics

During validation, the following metrics are logged to TensorBoard:

| Metric | Description |
|--------|-------------|
| **BLEU** | N-gram overlap with reference translations (higher is better) |
| **CER** | Character Error Rate (lower is better) |
| **WER** | Word Error Rate (lower is better) |

The model with the highest BLEU is saved as `weights/tmodel_BEST_MODEL.pth`.

---

## Configuration

Key options in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Training batch size (reduce for low GPU memory) |
| `num_epochs` | 20 | Number of training epochs |
| `lr` | 1e-4 | Learning rate |
| `seq_len` | 350 | Maximum sequence length |
| `d_model` | 512 | Model dimension |
| `lang_src` | `"en"` | Source language code |
| `lang_tgt` | `"it"` | Target language code |
| `datasource` | `"opus_books"` | Hugging Face dataset name |
| `model_folder` | `"weights"` | Checkpoint directory |
| `model_basename` | `"tmodel_"` | Checkpoint filename prefix |
| `preload` | `"None"` | Resume: `"latest"`, epoch index (e.g. `"10"`), or `"None"` |
| `experiment_name` | `"runs/tmodel_GCP_v1"` | TensorBoard log directory |

---

## Project Structure

```
My_Transformer/
├── config.py          # Hyperparameters and paths
├── model.py           # Transformer encoder-decoder implementation
├── dataset.py         # BilingualDataset and causal mask
├── train.py           # Training loop, tokenizer building, validation
├── translate.py       # CLI inference (Top-k, Top-p sampling)
├── app.py             # Streamlit web interface (greedy decoding)
├── inference.ipynb    # Jupyter inference examples
├── requirements.txt   # Python dependencies
├── tokenizer_en.json  # Source tokenizer (generated on first train)
├── tokenizer_it.json  # Target tokenizer (generated on first train)
├── weights/           # Model checkpoints (tmodel_00.pth, tmodel_BEST_MODEL.pth, ...)
└── runs/              # TensorBoard logs
```

---

## Using Different Languages

To train a model for another language pair (e.g., French → German):

1. **Update `config.py`:**

   ```python
   "lang_src": "fr",
   "lang_tgt": "de",
   "datasource": "opus_books"  # or another dataset with "fr-de" split
   ```

2. **Use a compatible dataset:** Hugging Face `datasets` supports many pairs. For custom data, adapt `get_ds()` in `train.py` to load your own parallel corpus.

3. **Run training:** Tokenizers will be created as `tokenizer_fr.json` and `tokenizer_de.json` on first run.

4. **Ensure tokenizer names in config:** `tokenizer_file` is `"tokenizer_{0}.json"`, so it will pick the correct files based on `lang_src` and `lang_tgt`.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **`FileNotFoundError` for weights/tokenizers** | Run `python train.py` first to generate tokenizers and checkpoints. |
| **`translate.py` returns nothing** | Ensure `weights/` contains at least one `tmodel_*.pth` file. |
| **Streamlit app fails to load** | Ensure `weights/` contains at least one `tmodel_*.pth` checkpoint. The app auto-loads the latest. |
| **CUDA out of memory** | Reduce `batch_size` in `config.py` (e.g., 4 or 2). |
| **Training very slow on CPU** | Expected; GPU is recommended. Consider cloud GPU (Colab, GCP, AWS) for full training. |
| **Dataset download fails** | Check internet connection; Hugging Face may require login for some datasets. |

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- `datasets` (Hugging Face)
- `tokenizers`
- `torchmetrics`
- `tensorboard`
- `tqdm`
- `streamlit` (for web app)

---

## Acknowledgments

- [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [OPUS](https://opus.nlpl.eu/) — Parallel corpora via Hugging Face `datasets`
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) — Fast tokenization

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
