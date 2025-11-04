# JAX LLM Implementation

A JAX-based implementation of large language model inference with support for LLaMA models, featuring efficient transformer architectures, KV caching, and advanced sampling techniques.

## Features

- **Pure JAX Implementation**: Leverages JAX's JIT compilation and auto-differentiation for efficient computation
- **LLaMA Model Support**: Load and run LLaMA 1B and other variants
- **Advanced Sampling Methods**:
  - Beam search with entropy-based Chain-of-Thought (CoT) injection
  - Nucleus (top-p) sampling
  - Top-k sampling
- **Efficient KV Caching**: Optimized key-value cache for faster inference
- **Rotary Positional Embeddings (RoPE)**: State-of-the-art position encoding
- **Grouped Query Attention (GQA)**: Support for models with different numbers of key-value heads
- **Visualization Tools**: Includes Mandelbrot fractal generation for testing JAX functionality

## Files Overview

- **`content.py`**: Main implementation with full LLaMA model loading, custom tokenizer, and inference
- **`cotent.py`**: Simplified transformer implementation with entropy-based CoT injection
- **`entrophy.py`**: Core transformer and sampling utilities with entropy calculations
- **`mandle.py`**: JAX functionality demo using Mandelbrot set generation

## Installation

### Prerequisites

- Python 3.8+
- JAX and JAX[cuda] (for GPU support)
- PyTorch (for loading model checkpoints)
- SentencePiece (for tokenization)

### Install Dependencies

```bash
pip install jax jaxlib
pip install torch
pip install sentencepiece
pip install matplotlib  # for visualization in mandle.py
```

For GPU support:
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Basic Text Generation

```python
from content import load_llama_weights, generate, compute_freqs_cis, tokenize, decode_tokens
import jax.numpy as jnp

# Load model
model_path = "/path/to/llama/checkpoints"
weights, model_params = load_llama_weights(model_path)
freqs_cis = compute_freqs_cis(
    model_params.hidden_dim // model_params.n_heads,
    model_params.max_seq_len
)

# Tokenize input
input_text = "Once upon a time"
input_tokens = jnp.array(tokenize(input_text))[None, :]

# Generate with nucleus sampling
generated = generate(
    input_tokens,
    model_params,
    weights,
    freqs_cis,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9
)

print(decode_tokens(generated[0].tolist()))
```

### Beam Search with CoT

```python
from content import beam_search

# Generate with beam search
generated = beam_search(
    model_params,
    weights,
    freqs_cis,
    input_tokens,
    beam_size=4,
    max_new_tokens=50,
    entropy_threshold=0.5
)

print(decode_tokens(generated.tolist()))
```

## Model Architecture

The implementation supports transformer models with:

- **Multi-head attention** with rotary position embeddings
- **Grouped query attention** for efficient inference
- **RMS normalization** instead of LayerNorm
- **SwiGLU activation** in feed-forward layers
- **KV caching** for autoregressive generation

### Key Components

- `compute_freqs_cis()`: Generates rotary embedding frequencies
- `apply_rotary_emb()`: Applies RoPE to queries and keys
- `attention()`: Multi-head attention with KV cache
- `feed_forward()`: SwiGLU feed-forward network
- `transformer_layer()`: Complete transformer block
- `xfmr()`: Full transformer model

## Configuration

To use your own model checkpoint, update the model path in the main functions:

```python
model_path = "/path/to/your/llama/checkpoints"
```

The model expects the following structure:
```
model_path/
  ├── params.json           # Model configuration
  ├── consolidated.00.pth   # Model weights
  └── tokenizer.model       # Tokenizer vocabulary
```

## Performance Tips

1. **JIT Compilation**: Most functions are decorated with `@jax.jit` for speed
2. **Static Arguments**: Shape-related parameters are marked as `static_argnames`
3. **Batch Processing**: Use batched inputs for better throughput
4. **Precision**: Models support both float32 and bfloat16 computation

## Entropy-Based Chain-of-Thought

The beam search implementation includes an innovative entropy-based CoT injection mechanism:

- Monitors the entropy of output probability distributions
- Automatically inserts CoT tokens when entropy drops below threshold
- Encourages the model to "think" during complex reasoning tasks

## Examples

### Run the Demo

```bash
python content.py
```

### Visualize JAX Functionality

```bash
python mandle.py
```

This generates a Mandelbrot fractal using JAX's vectorized operations.

## Project Status

This is an experimental implementation for research and educational purposes. The code demonstrates:

- JAX programming patterns for ML
- Efficient transformer implementation
- Advanced sampling techniques
- LLaMA architecture details

## Contributing

Contributions welcome! Areas for improvement:

- Add comprehensive test suite
- Support for more model formats
- Quantization support
- Better error handling
- Configuration file support

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Based on the LLaMA architecture from Meta AI
- Implements concepts from various papers on transformer optimization
- Uses JAX for high-performance numerical computing
