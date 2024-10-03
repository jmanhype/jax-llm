import jax
import jax.numpy as jnp
from jax import random, nn, lax
from functools import partial
from typing import NamedTuple, Optional, Dict, List, Tuple
import math

# Define custom types
class LayerWeights(NamedTuple):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    attention_norm: jax.Array
    ffn_norm: jax.Array

class ModelParams(NamedTuple):
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
            w1=random.normal(layer_key[4], (dim, dim * 4)) * 0.02,
            w2=random.normal(layer_key[5], (dim * 4, dim)) * 0.02,
            w3=random.normal(layer_key[6], (dim, dim * 4)) * 0.02,
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

@partial(jax.jit, static_argnames=("model_params", "cur_pos", "layer_idx"))
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
    n_rep = model_params.n_heads // model_params.n_kv_heads
    head_dim = model_params.hidden_dim // model_params.n_heads

    xq = jnp.dot(x, layer_weights.wq.T).reshape(batch_size, seq_len, model_params.n_heads, head_dim)
    xk = jnp.dot(x, layer_weights.wk.T).reshape(batch_size, seq_len, model_params.n_kv_heads, head_dim)
    xv = jnp.dot(x, layer_weights.wv.T).reshape(batch_size, seq_len, model_params.n_kv_heads, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis, x.dtype)
    
    keys, values = kvcache
    keys = jnp.concatenate([keys, xk], axis=1)
    values = jnp.concatenate([values, xv], axis=1)
    
    xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seq_len, head_dim)
    keys = jnp.transpose(keys, (0, 2, 1, 3))  # (bs, n_kv_heads, total_seq_len, head_dim)
    values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_kv_heads, total_seq_len, head_dim)
    
    scores = jnp.matmul(xq, keys.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    scores = scores.astype(jnp.float32)  # Always do attention softmax in float32
    
    if attn_mask is not None:
        mask = attn_mask[None, None, cur_pos:cur_pos+seq_len, :keys.shape[2]]
        mask = jnp.broadcast_to(mask, scores.shape)
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    
    scores = nn.softmax(scores, axis=-1).astype(x.dtype)
    output = jnp.matmul(scores, values)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
    output = jnp.dot(output, layer_weights.wo.T)
    
    return output, KVCache(keys, values)

@jax.jit
def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
    """Compute feed-forward layer."""
    hidden_dim = x.shape[-1]
    intermediate_dim = layer_weights.w1.shape[-1]
    
    h = jnp.dot(x, layer_weights.w1)
    h = nn.silu(h)
    h = h * jnp.dot(x, layer_weights.w3)
    return jnp.dot(h, layer_weights.w2)

@partial(jax.jit, static_argnames=("model_params", "cur_pos"))
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
    h, new_kvcache = attention(h, layer_weights, model_params, cur_pos, layer_idx, freqs_cis, kvcache, attn_mask)
    x = x + h
    h = rms_norm(x, layer_weights.ffn_norm)
    x = x + feed_forward(h, layer_weights)
    return x, new_kvcache

@partial(jax.jit, static_argnames=("model_params", "cur_pos"))
def xfmr(
    tokens: jax.Array,
    model_params: ModelParams,
    weights: Dict[str, jax.Array],
    freqs_cis: jax.Array,
    cur_pos: int,
    kvcache: Optional[List[KVCache]] = None,
    attn_mask: Optional[jax.Array] = None
) -> tuple[jax.Array, List[KVCache]]:
    """Compute the transformer model."""
    batch_size, seq_len = tokens.shape
    h = weights['tok_embeddings'][tokens]
    
    if kvcache is None:
        kvcache = [KVCache(jnp.zeros((batch_size, 0, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)),
                           jnp.zeros((batch_size, 0, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)))
                   for _ in range(model_params.n_layers)]
    
    new_kvcache = []
    for i in range(model_params.n_layers):
        h, layer_kvcache = transformer_layer(h, weights[f'layers.{i}'], model_params, cur_pos, i, freqs_cis, kvcache[i], attn_mask)
        new_kvcache.append(layer_kvcache)
    
    h = rms_norm(h, weights['norm'])
    logits = jnp.dot(h, weights['output'])
    
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
    
    # Initialize beam
    beam_tokens = jnp.tile(input_tokens, (beam_size, 1))
    beam_scores = jnp.zeros((beam_size,))
    
    # Initialize KV cache
    kvcache = [KVCache(jnp.zeros((beam_size, 0, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)),
                       jnp.zeros((beam_size, 0, model_params.n_kv_heads, model_params.hidden_dim // model_params.n_heads)))
               for _ in range(model_params.n_layers)]
    
    def body_fun(state, _):
        beam_tokens, beam_scores, kvcache, cur_pos = state
        
        # Forward pass
        logits, new_kvcache = xfmr(beam_tokens, model_params, weights, freqs_cis, cur_pos, kvcache)
        next_token_logits = logits[:, -1, :]
        
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
        new_beam_tokens = jnp.concatenate([beam_tokens, top_indices[:, None] % model_params.vocab_size], axis=1)
        new_beam_scores = top_scores
        
        # Update KV cache for new beam
        beam_indices = top_indices // model_params.vocab_size
        new_kvcache = [(KVCache(k[beam_indices], v[beam_indices])) for k, v in new_kvcache]
        
        # Inject CoT token if entropy is below threshold
        def inject_cot():
            cot_beam_tokens = jnp.concatenate([new_beam_tokens, jnp.full((beam_size, 1), cot_token)], axis=1)
            return cot_beam_tokens, new_beam_scores, new_kvcache, cur_pos + 2
        
        def keep_beam():
            return new_beam_tokens, new_beam_scores, new_kvcache, cur_pos + 1
        
        return jax.lax.cond(jnp.mean(entropy) < entropy_threshold, inject_cot, keep_beam)
    
    # Run beam search
    init_state = (beam_tokens, beam_scores, kvcache, seq_len)
    final_state, _ = jax.lax.scan(body_fun, init_state, None, length=max_new_tokens)
    
    # Select the best beam
    best_beam_idx = jnp.argmax(final_state[1])
    best_sequence = final_state[0][best_beam_idx]
    
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
        logits, new_kvcache = xfmr(tokens, model_params, weights, freqs_cis, cur_pos, kvcache)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Nucleus sampling
        sorted_logits, sorted_indices = jax.lax.top_k(next_token_logits, model_params.vocab_size)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.concatenate([jnp.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, :-1]], axis=-1)
        next_token_logits = jnp.where(sorted_indices_to_remove, -jnp.inf, sorted_logits)
        
        next_token = jax.random.categorical(jax.random.PRNGKey(0), next_token_logits, axis=-1)
        next_token = jnp.take_along_axis(sorted_indices, next_token[:, None], axis=-1).squeeze(-1)
        
        return (jnp.concatenate([tokens, next_token[:, None]], axis=-1), cur_pos + 1, new_kvcache)
    
    init_state = (input_tokens, input_tokens.shape[1], None)
    final_state, _ = jax.lax.scan(body_fun, init_state, None, length=max_new_tokens)
    return final_state[0]

def main():
    # Model parameters (adjust these to match LLaMA 1B)
    model_params = ModelParams(
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,
        vocab_size=32000,
        hidden_dim=2048,
        intermediate_dim=5504,
        max_seq_len=2048
    )

    # Load LLaMA weights (you'll need to implement this function)
    weights = load_llama_weights("/path/to/llama/1b/weights")

    # Compute frequency cis
    freqs_cis = compute_freqs_cis(model_params.hidden_dim // model_params.n_heads, model_params.max_seq_len)

    # Create some example input data
    input_text = "Once upon a time"
    input_tokens = tokenize(input_text)  # You'll need to implement this function
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

    # Decode generated tokens (you'll need to implement this function)
    generated_text = decode_tokens(generated_tokens)
    generated_text_nucleus = decode_tokens(generated_tokens_nucleus)

    print("Beam Search Output:", generated_text)
    print("Nucleus Sampling Output:", generated_text_nucleus)

if __name__ == "__main__":
    main()