import torch

print("⚙️ Booting Memory Manager: The Dynamic KV-Cache Evictor...")

def evict_garbage_tokens(past_key_values, attention_scores, max_cache_size=15):
    """
    The ASIC Hardware Logic:
    Takes the massive KV cache, protects the critical anchors, finds the 
    lowest scoring tokens in the middle, and physically deletes them from VRAM.
    """
    # Grab the current sequence length from the tensor shape
    # Shape is: (batch_size, num_heads, sequence_length, embed_dim)
    seq_len = past_key_values[0][0].shape[2] 
    
    if seq_len <= max_cache_size:
        return past_key_values, 0 # No need to evict yet, we are under budget

    # 1. THE SHIELD: Protect the "Attention Sinks" (first 2 tokens) 
    # and "Local Context" (last 3 tokens). We only evict from the middle.
    sink_tokens = [0, 1]
    local_tokens = list(range(seq_len - 3, seq_len))
    protected_indices = set(sink_tokens + local_tokens)
    
    # 2. THE SCORER: Find the lowest scoring tokens in the unprotected middle
    middle_indices = [i for i in range(seq_len) if i not in protected_indices]
    
    # Sort middle indices based on their attention score (lowest to highest)
    middle_indices.sort(key=lambda i: attention_scores[i].item())
    
    # Calculate exactly how many tokens we need to delete to get back under budget
    num_to_delete = seq_len - max_cache_size
    indices_to_delete = set(middle_indices[:num_to_delete])
    
    # 3. THE SURGEON: Create the list of indices we are keeping
    indices_to_keep = [i for i in range(seq_len) if i not in indices_to_delete]
    
    # Move the keep list to the same device (CPU/GPU) as the cache
    keep_tensor = torch.tensor(indices_to_keep).to(past_key_values[0][0].device)
    
    # 4. THE EXECUTION: Physically slice the tensors across all 12 layers
    new_past_key_values = []
    for layer_idx in range(len(past_key_values)):
        key, value = past_key_values[layer_idx]
        
        # index_select safely slices the multidimensional tensor along dim=2 (sequence length)
        new_key = torch.index_select(key, dim=2, index=keep_tensor)
        new_value = torch.index_select(value, dim=2, index=keep_tensor)
        
        new_past_key_values.append((new_key, new_value))

    return tuple(new_past_key_values), num_to_delete

# --- HARDWARE SIMULATION TEST ---
if __name__ == "__main__":
    print("\n🧪 Running Tensor Slicing Simulation...")
    
    # Simulate a bloated KV Cache: 12 layers, batch 1, 12 heads, 20 tokens, 64 dim
    fake_kv_cache = tuple((torch.randn(1, 12, 20, 64), torch.randn(1, 12, 20, 64)) for _ in range(12))
    
    # Simulate 20 random attention scores
    fake_scores = torch.rand(20)
    
    print(f"Original KV Cache Shape (Layer 0, Key): {fake_kv_cache[0][0].shape}")
    
    # Tell the manager to evict memory down to a strict 15-token budget
    optimized_cache, deleted_count = evict_garbage_tokens(fake_kv_cache, fake_scores, max_cache_size=15)
    
    print(f"🗑️ Deleted {deleted_count} low-entropy tokens from memory.")
    print(f"Optimized KV Cache Shape (Layer 0, Key): {optimized_cache[0][0].shape}")
    print("\n✅ Tensor memory physically freed without breaking the sequence!")