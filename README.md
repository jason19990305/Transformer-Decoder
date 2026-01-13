# Transsformer Decoder Example

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)


HackMD Article : https://hackmd.io/@bGCXESmGSgeAArScMaBxLA/rJvmAClSbl


This is a **PyTorch** implementation of a **Self-Attention** mechanism and a **Simple Transformer** model. This project demonstrates the core components of the Transformer architecture, including Scaled Dot-Product Attention, Positional Encoding, and a basic text completion task using a custom tokenizer.

It includes a complete pipeline from tokenization to training and inference on a toy dataset.

## 📂 Project Structure

```text
.
├── main.py                 # Main training and inference script
├── transformer.py          # SimpleTransformer model & Self-Attention mechanism
├── positional_encoding.py  # Sinusoidal Positional Encoding implementation
├── word_embedding.py       # Word Embedding layer wrapper
├── tokenizer.py            # Simple word-level tokenizer
└── README.md               # Project documentation
```

## 🚀 Installation

### 1. Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### 2. Install Dependencies
This project primarily uses **PyTorch**.

```bash
# Install core dependencies
pip install torch numpy matplotlib
```

## 🖥️ Usage

### Text Completion Demo
The `main.py` script trains a clear Simple Transformer on a small corpus ("red apple is sweet", "blue sky is high", etc.) and demonstrates text completion.

```bash
python main.py
```

*   **Training**: Trains the model for 2000 epochs to learn the sequence patterns.
*   **Inference**: Generates text completions for prompts like "red apple" and "red fire", using `<eos>` token for variable length generation.

<img width="1779" height="423" alt="image" src="https://github.com/user-attachments/assets/de5e11d7-413b-4039-b8ee-0e6fd41195c3" />



### Component Visualization
You can run individual modules to test and visualize detailed components:

*   **Positional Encoding Heatmap**:
    ```bash
    python positional_encoding.py
    ```
    Generates a heatmap `positional_encoding_heatmap.png` visualizing the sinusoidal position embeddings.

## 💡 Technical Highlights

- **Self-Attention**: Implements the core attention mechanism:
 
  Includes causal masking to prevent attending to future tokens.
  
- **Positional Encoding**: Uses standard sinusoidal encoding to inject order information into the sequence.


- **Custom Tokenizer**: A simple `WordTokenizer` that handles vocabulary building, encoding, and decoding with `<eos>` support.

- **Variable Length Generation**: The inference loop supports generating sequences of varying lengths by monitoring the End-Of-Sequence (`<eos>`) token.
