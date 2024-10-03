import jax
import jax.numpy as jnp
from jax import random, nn, lax
from functools import partial
from typing import NamedTuple, Optional, Dict, List

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

def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jax.Array:
    """Compute frequency cis for rotary embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).reshape(end, -1)  # Changed this line
    freqs = jnp.concatenate([freqs, freqs], axis=-1)  # Added this line
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
    
    # Ensure freqs_cis has the correct shape for broadcasting
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
        # Adjust attn_mask shape to match scores
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

# Example usage
def main():
    # Model parameters
    model_params = ModelParams(
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=32000,
        hidden_dim=256,
        intermediate_dim=1024,  # This is typically 4 times the hidden_dim
        max_seq_len=1024
    )

    # Initialize weights and compute frequency cis
    key = random.PRNGKey(0)
    weights = initialize_weights(model_params, key)
    freqs_cis = compute_freqs_cis(model_params.hidden_dim // model_params.n_heads, model_params.max_seq_len)

    # Create some example input data
    batch_size = 1
    seq_len = 16
    input_tokens = random.randint(key, (batch_size, seq_len), 0, model_params.vocab_size)

    # Create an attention mask (optional, set to None if not needed)
    attn_mask = jnp.tril(jnp.ones((model_params.max_seq_len, model_params.max_seq_len)), k=0).astype(bool)

    # Run the model
    logits, kvcache = xfmr(input_tokens, model_params, weights, freqs_cis, cur_pos=0, attn_mask=attn_mask)

    print(f"Input shape: {input_tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of KV caches: {len(kvcache)}")
    print(f"KV cache shapes: {kvcache[0].k.shape}, {kvcache[0].v.shape}")

if __name__ == "__main__":
    main()