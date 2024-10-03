import jax
import jax.numpy as jnp
from jax import random, nn, lax
from functools import partial
from typing import NamedTuple, Optional, Dict, List, Tuple
import math
import os
import json
import torch
from pathlib import Path
import sentencepiece as spm
import base64
from dataclasses import dataclass

# Define custom types
class LayerWeights(NamedTuple):
    wq: jax.Array  # shape: (hidden_dim, hidden_dim)
    wk: jax.Array  # shape: (hidden_dim, hidden_dim)
    wv: jax.Array  # shape: (hidden_dim, hidden_dim)
    wo: jax.Array  # shape: (hidden_dim, hidden_dim)
    w1: jax.Array  # shape: (4*hidden_dim, hidden_dim)
    w2: jax.Array  # shape: (hidden_dim, 4*hidden_dim)
    w3: jax.Array  # shape: (4*hidden_dim, hidden_dim)
    attention_norm: jax.Array  # shape: (hidden_dim,)
    ffn_norm: jax.Array  # shape: (hidden_dim,)

@dataclass(frozen=True)
class ModelParams:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    max_seq_len: int

class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array

DEFAULT_MASK_VALUE = -1e10
COT_TOKEN = 99999  # Example token ID for "[CoT]"

class CustomLlamaTokenizer:
    def __init__(self, model_path: str):
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        
        with open(f"{model_path}/tokenizer.model", "r") as f:
            for line in f:
                token, id_str = line.strip().split()
                token_bytes = base64.b64decode(token)
                id = int(id_str)
                self.token_to_id[token_bytes] = id
                self.id_to_token[id] = token_bytes

    def tokenize(self, text: str) -> List[int]:
        tokens = []
        text_bytes = text.encode('utf-8')
        i = 0
        while i < len(text_bytes):
            for j in range(len(text_bytes), i, -1):
                sub_bytes = text_bytes[i:j]
                if sub_bytes in self.token_to_id:
                    tokens.append(self.token_to_id[sub_bytes])
                    i = j
                    break
            else:
                tokens.append(self.token_to_id.get(b' ', 0))  # Use  token if available, else 0
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_chunks = [self.id_to_token.get(id, b' ') for id in token_ids]
        return b''.join(byte_chunks).decode('utf-8', errors='replace')

# Initialize the tokenizer
tokenizer = CustomLlamaTokenizer("/home/batmanosama/.llama/checkpoints/Llama3.2-1B-Instruct")

def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jax.Array:
    """Compute frequency cis for rotary embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).reshape(end, -1)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.exp(1j * freqs)

def rms_norm(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Compute RMS normalization."""
    return x * weight * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5)

def create_norm_fn(dim: int, key: jax.random.PRNGKey) -> tuple[callable, jax.Array]:
    """Create a normalization function with initialized weights."""
    weight = jax.random.normal(key, (dim,)) * 0.02 + 1.0
    return lambda x: rms_norm(x, weight), weight

def initialize_weights(model_params: ModelParams, key: jax.random.PRNGKey) -> Dict[str, jax.Array]:
    """Initialize weights for the transformer model."""
    keys = random.split(key, 10)
    dim = model_params.hidden_dim
    
    norm_fn, norm_weight = create_norm_fn(dim, keys[1])
    weights = {
        'tok_embeddings': random.normal(keys[0], (model_params.vocab_size, dim)) * 0.02,
        'norm': norm_weight,
        'output': random.normal(keys[2], (dim, model_params.vocab_size)) * 0.02,
    }
    
    for i in range(model_params.n_layers):
        layer_key = random.split(keys[i+3], 9)
        attention_norm_fn, attention_norm_weight = create_norm_fn(dim, layer_key[7])
        ffn_norm_fn, ffn_norm_weight = create_norm_fn(dim, layer_key[8])
        weights[f'layers.{i}'] = LayerWeights(
            wq=random.normal(layer_key[0], (dim, dim)) * 0.02,
            wk=random.normal(layer_key[1], (dim, dim)) * 0.02,
            wv=random.normal(layer_key[2], (dim, dim)) * 0.02,
            wo=random.normal(layer_key[3], (dim, dim)) * 0.02,
            w1=random.normal(layer_key[4], (dim * 4, dim)) * 0.02,
            w2=random.normal(layer_key[5], (dim, dim * 4)) * 0.02,
            w3=random.normal(layer_key[6], (dim * 4, dim)) * 0.02,
            attention_norm=attention_norm_weight,
            ffn_norm=ffn_norm_weight,
        )
    
    return weights

@partial(jax.jit, static_argnames=("dtype",))
def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to input tensors."""
    batch_size, seq_len, n_heads, head_dim = xq.shape
    xq_r = xq.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
    xk_r = xk.reshape(batch_size, seq_len, xk.shape[2], head_dim // 2, 2)
    
    freqs_cis = freqs_cis[:seq_len].reshape(1, seq_len, 1, head_dim // 2, 2)
    
    def rotate_half(x):
        x1, x2 = x[..., 0], x[..., 1]
        return jnp.stack([-x2, x1], axis=-1)

    xq_out = xq_r * freqs_cis + rotate_half(xq_r) * freqs_cis[..., ::-1]
    xk_out = xk_r * freqs_cis + rotate_half(xk_r) * freqs_cis[..., ::-1]
    
    xq_out = xq_out.reshape(batch_size, seq_len, n_heads, head_dim)
    xk_out = xk_out.reshape(batch_size, seq_len, xk.shape[2], head_dim)
    
    return xq_out.astype(dtype), xk_out.astype(dtype)

@partial(jax.jit, static_argnames=('model_params', 'layer_idx'))
def attention(
    x: jax.Array,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None
) -> tuple[jax.Array, KVCache]:
    """Compute multi-head attention."""
    batch_size, seq_len, _ = x.shape
    head_dim = model_params.hidden_dim // model_params.n_heads
    max_seq_len = model_params.max_seq_len

    # Compute Q, K, V
    xq = jnp.dot(x, layer_weights.wq.T).reshape(batch_size, seq_len, model_params.n_heads, head_dim)
    xk = jnp.dot(x, layer_weights.wk.T).reshape(batch_size, seq_len, model_params.n_kv_heads, head_dim)
    xv = jnp.dot(x, layer_weights.wv.T).reshape(batch_size, seq_len, model_params.n_kv_heads, head_dim)

    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis, x.dtype)
    
    # Update KV cache
    keys, values = kvcache
    keys = jax.lax.dynamic_update_slice(keys, xk, (0, cur_pos, 0, 0))
    values = jax.lax.dynamic_update_slice(values, xv, (0, cur_pos, 0, 0))
    
    # Prepare for attention computation
    xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seq_len, head_dim)
    keys = jnp.transpose(keys, (0, 2, 1, 3))  # (bs, n_kv_heads, max_seq_len, head_dim)
    values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_kv_heads, max_seq_len, head_dim)
    
    # Repeat keys and values to match the number of query heads
    n_rep = model_params.n_heads // model_params.n_kv_heads
    keys = jnp.repeat(keys, n_rep, axis=1)
    values = jnp.repeat(values, n_rep, axis=1)
    
    # Compute attention scores
    scores = jnp.matmul(xq, keys.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    scores = scores.astype(jnp.float32)
    
    # Create causal mask
    mask_value = jnp.finfo(scores.dtype).min
    causal_mask = jnp.tril(jnp.ones((max_seq_len, max_seq_len)))
    causal_mask = jnp.where(causal_mask == 0, mask_value, 0)
    causal_mask = causal_mask[None, None, :, :]
    
    # Apply causal mask
    causal_mask_slice = jax.lax.dynamic_slice(
        causal_mask, (0, 0, cur_pos, 0), (1, 1, seq_len, max_seq_len)
    )
    causal_mask_slice = jnp.broadcast_to(causal_mask_slice, scores.shape)
    scores = jnp.where(causal_mask_slice == 0, scores, mask_value)
    
    # Apply attention mask if provided
    if attn_mask is not None:
        attn_mask = jnp.broadcast_to(attn_mask[:, None, None, :], scores.shape)
        scores = jnp.where(attn_mask, scores, mask_value)
    
    attn_weights = nn.softmax(scores, axis=-1).astype(x.dtype)
    
    # Use dynamic_slice for values
    values_to_use = jax.lax.dynamic_slice(
        values, (0, 0, 0, 0), (batch_size, model_params.n_heads, cur_pos + seq_len, head_dim)
    )
    context = jnp.matmul(attn_weights, values_to_use)
    
    context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
    output = jnp.dot(context, layer_weights.wo.T)
    
    return output, KVCache(keys[:, :model_params.n_kv_heads], values[:, :model_params.n_kv_heads])

@jax.jit
def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
    """Compute feed-forward layer with grouped linear layers."""
    h1 = jnp.dot(x, layer_weights.w1.T)
    h2 = jnp.dot(x, layer_weights.w3.T)
    h = nn.silu(h1) * h2
    # Reshape h to (batch_size, seq_len, 4, hidden_dim // 4)
    h = h.reshape(*h.shape[:-1], 4, -1)
    # Transpose h to (batch_size, seq_len, hidden_dim // 4, 4)
    h = jnp.transpose(h, (0, 1, 3, 2))
    # Reshape h back to (batch_size, seq_len, hidden_dim)
    h = h.reshape(*h.shape[:-2], -1)
    return jnp.dot(h, layer_weights.w2.T)

@partial(jax.jit, static_argnames=('model_params', 'layer_idx', 'cur_pos'))
def transformer_layer(
    x: jax.Array,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None
) -> tuple[jax.Array, KVCache]:
    """Compute a single transformer layer."""
    h = rms_norm(x, layer_weights.attention_norm)
    attn_output, new_kvcache = attention(h, layer_weights, model_params, cur_pos, layer_idx, freqs_cis, kvcache, attn_mask)
    x = x + attn_output  # Add residual connection
    h = rms_norm(x, layer_weights.ffn_norm)
    x = x + feed_forward(h, layer_weights)
    return x, new_kvcache

@partial(jax.jit, static_argnames=('model_params', 'cur_pos'))
def xfmr(
    tokens: jax.Array,
    model_params: ModelParams,
    weights: Dict[str, jax.Array],
    freqs_cis: jax.Array,
    cur_pos: int,
    kvcache: Optional[List[KVCache]] = None
) -> tuple[jax.Array, List[KVCache]]:
    """Compute the transformer model."""
    batch_size, seq_len = tokens.shape
    h = weights['tok_embeddings.weight'][tokens]
    
    if kvcache is None:
        kvcache = [KVCache(jnp.zeros((batch_size, model_params.max_seq_len, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)),
                           jnp.zeros((batch_size, model_params.max_seq_len, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)))
                   for _ in range(model_params.n_layers)]
    
    new_kvcache = []
    for i in range(model_params.n_layers):
        layer_prefix = f'layers.{i}.'
        layer_weights = LayerWeights(
            wq=weights[f'{layer_prefix}attention.wq.weight'],
            wk=weights[f'{layer_prefix}attention.wk.weight'],
            wv=weights[f'{layer_prefix}attention.wv.weight'],
            wo=weights[f'{layer_prefix}attention.wo.weight'],
            w1=weights[f'{layer_prefix}feed_forward.w1.weight'],
            w2=weights[f'{layer_prefix}feed_forward.w2.weight'],
            w3=weights[f'{layer_prefix}feed_forward.w3.weight'],
            attention_norm=weights[f'{layer_prefix}attention_norm.weight'],
            ffn_norm=weights[f'{layer_prefix}ffn_norm.weight'],
        )
        
        h, layer_kvcache = transformer_layer(h, layer_weights, model_params, cur_pos, i, freqs_cis, kvcache[i])
        new_kvcache.append(layer_kvcache)
    
    h = rms_norm(h, weights['norm.weight'])
    logits = jnp.dot(h, weights['output.weight'].T)
    
    return logits, new_kvcache

def calculate_entropy(probs: jax.Array) -> jax.Array:
    """Calculate entropy of the probability distribution."""
    return -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)

def get_cot_token() -> int:
    """Return the CoT token."""
    return COT_TOKEN

@jax.jit
def top_k_sampling(logits: jax.Array, top_k: int, temperature: float = 1.0) -> jax.Array:
    """Perform top-k sampling on logits."""
    scaled_logits = logits / temperature
    top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, top_k)
    probs = jax.nn.softmax(top_k_logits, axis=-1)
    return top_k_indices, probs

def beam_search(
    model_params: ModelParams,
    weights: Dict[str, jax.Array],
    freqs_cis: jax.Array,
    input_tokens: jax.Array,
    beam_size: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
    alpha: float = 0.6,
    entropy_threshold: float = 0.5,
    cot_token: int = COT_TOKEN
) -> jax.Array:
    """Perform beam search with entropy-based CoT injection."""
    batch_size, seq_len = input_tokens.shape
    max_len = seq_len + max_new_tokens
    
    # Initialize beam
    beam_tokens = jnp.pad(input_tokens, ((0, beam_size - 1), (0, max_new_tokens)), constant_values=0)
    beam_scores = jnp.zeros((beam_size,))
    
    # Initialize KV cache
    kvcache = [KVCache(jnp.zeros((beam_size, max_len, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)),
                       jnp.zeros((beam_size, max_len, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)))
               for _ in range(model_params.n_layers)]
    
    def body_fun(state, cur_pos):
        beam_tokens, beam_scores, kvcache = state
        
        # Forward pass
        logits, new_kvcache = xfmr(beam_tokens, model_params, weights, freqs_cis, cur_pos, kvcache)
        next_token_logits = logits[:, cur_pos-1, :]
        
        # Calculate log probabilities
        log_probs = jax.nn.log_softmax(next_token_logits, axis=-1)
        
        # Calculate entropy
        probs = jnp.exp(log_probs)
        entropy = calculate_entropy(probs)
        
        # Adjust scores based on length penalty
        scores = beam_scores[:, None] + log_probs
        length_penalty = ((5 + cur_pos + 1) / 6) ** alpha
        scores = scores / length_penalty
        
        # Get top-k next tokens and their scores
        flat_scores = scores.reshape(-1)
        top_scores, top_indices = jax.lax.top_k(flat_scores, beam_size)
        
        # Calculate new beam tokens and scores
        new_tokens = top_indices % model_params.vocab_size
        new_beam_scores = top_scores
        
        # Update beam tokens
        beam_indices = top_indices // model_params.vocab_size
        new_beam_tokens = beam_tokens.at[:, cur_pos].set(new_tokens)
        new_beam_tokens = new_beam_tokens.at[:].set(new_beam_tokens[beam_indices])
        
        # Update KV cache for new beam
        new_kvcache = [(KVCache(k.at[:].set(k[beam_indices]), v.at[:].set(v[beam_indices]))) for k, v in new_kvcache]
        
        # Inject CoT token if entropy is below threshold
        def inject_cot():
            cot_beam_tokens = new_beam_tokens.at[:, cur_pos+1].set(cot_token)
            return cot_beam_tokens, new_beam_scores, new_kvcache, cur_pos + 2
        
        def keep_beam():
            return new_beam_tokens, new_beam_scores, new_kvcache, cur_pos + 1
        
        new_state = jax.lax.cond(jnp.mean(entropy) < entropy_threshold, inject_cot, keep_beam)
        return new_state, new_state[0][:, cur_pos:cur_pos+1]  # Return the state and only the new tokens for scan
    
    # Run beam search
    init_state = (beam_tokens, beam_scores, kvcache)
    final_state, token_sequence = jax.lax.scan(body_fun, init_state, jnp.arange(seq_len, max_len))
    
    # Select the best beam
    best_beam_idx = jnp.argmax(final_state[1])
    best_sequence = final_state[0][best_beam_idx, :max_len]
    
    return best_sequence

@partial(jax.jit, static_argnames=("model_params", "max_new_tokens", "temperature", "top_p"))
def generate(
    input_tokens: jax.Array,
    model_params: ModelParams,
    weights: Dict[str, jax.Array],
    freqs_cis: jax.Array,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9
) -> jax.Array:
    """Generate tokens using nucleus sampling."""
    def body_fun(state, _):
        tokens, cur_pos, kvcache = state
        logits, new_kvcache = xfmr(tokens[:, -1:], model_params, weights, freqs_cis, cur_pos, kvcache)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Nucleus sampling
        sorted_logits, sorted_indices = jax.lax.top_k(next_token_logits, model_params.vocab_size)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.concatenate([jnp.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], axis=-1)
        next_token_logits = jnp.where(sorted_indices_to_remove, -jnp.inf, sorted_logits)
        
        next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits, axis=-1)
        next_token = jnp.take_along_axis(sorted_indices, next_token[:, None], axis=-1).squeeze(-1)
        
        return (jnp.concatenate([tokens, next_token[:, None]], axis=-1), cur_pos + 1, new_kvcache), next_token
    
    init_state = (input_tokens, input_tokens.shape[1], None)
    final_state, token_sequence = jax.lax.scan(body_fun, init_state, None, length=max_new_tokens)
    return jnp.concatenate([input_tokens, token_sequence.T], axis=-1)

def load_llama_weights(model_path: str) -> Tuple[Dict[str, jax.Array], ModelParams]:
    """Load LLaMA weights from the downloaded checkpoint."""
    model_path = Path(model_path)
    params_path = model_path / "params.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Load the PyTorch state dict
    state_dict = torch.load(model_path / "consolidated.00.pth", map_location="cpu")

    # Determine the hidden dimension and intermediate dimension from the weights
    hidden_dim = state_dict['layers.0.attention.wq.weight'].shape[0]
    intermediate_dim = state_dict['layers.0.feed_forward.w1.weight'].shape[0]

    print(f"Model dimensions: hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}")

    # Update model parameters based on the loaded JSON and weights
    model_params = ModelParams(
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        n_kv_heads=params.get("n_kv_heads", params["n_heads"]),
        vocab_size=params["vocab_size"],
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=params.get("max_seq_len", 2048)
    )

    # Convert PyTorch tensors to JAX arrays
    weights = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                value = value.float()
            weights[key] = jnp.array(value.numpy())

    return weights, model_params

def tokenize(text: str) -> List[int]:
    """Tokenize the input text using the custom LLaMA tokenizer."""
    return tokenizer.tokenize(text)

def decode_tokens(tokens: List[int]) -> str:
    """Decode the tokens back to text using the custom LLaMA tokenizer."""
    return tokenizer.decode(tokens)

def main():
    # Load LLaMA weights and model parameters
    model_path = "/home/batmanosama/.llama/checkpoints/Llama3.2-1B-Instruct"
    weights, model_params = load_llama_weights(model_path)

    # Compute frequency cis
    freqs_cis = compute_freqs_cis(model_params.hidden_dim // model_params.n_heads, model_params.max_seq_len)

    # Create some example input data
    input_text = "Once upon a time"
    input_tokens = tokenize(input_text)
    input_tokens = jnp.array(input_tokens)[None, :]  # Add batch dimension

    # Generate text using beam search
    generated_tokens = beam_search(
        model_params,
        weights,
        freqs_cis,
        input_tokens,
        beam_size=4,
        max_new_tokens=50,
        entropy_threshold=0.5
    )

    # Generate text using nucleus sampling
    generated_tokens_nucleus = generate(
        input_tokens,
        model_params,
        weights,
        freqs_cis,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )

    # Decode generated tokens
    generated_text = decode_tokens(generated_tokens)
    generated_text_nucleus = decode_tokens(generated_tokens_nucleus)

    print("Beam Search Output:", generated_text)
    print("Nucleus Sampling Output:", generated_text_nucleus)

if __name__ == "__main__":
    main()