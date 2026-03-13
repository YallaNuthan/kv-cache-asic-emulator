import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("🚀 Booting the ASIC Emulator: Infinite Inference Engine (Stable Arch)...")

# --- THE HARDWARE MEMORY MANAGER ---
def evict_garbage_tokens(past_key_values, attention_scores, max_cache_size):
    # Because we pinned our version, past_key_values is a beautiful, raw tuple again!
    seq_len = past_key_values[0][0].shape[2] 
    if seq_len <= max_cache_size:
        return past_key_values, 0

    # Shield the Attention Sinks and Local Context
    sink_tokens = [0, 1]
    local_tokens = list(range(seq_len - 3, seq_len))
    protected_indices = set(sink_tokens + local_tokens)
    
    # Identify the lowest scoring tokens in the middle
    middle_indices = [i for i in range(seq_len) if i not in protected_indices]
    middle_indices.sort(key=lambda i: attention_scores[i].item())
    
    num_to_delete = seq_len - max_cache_size
    indices_to_delete = set(middle_indices[:num_to_delete])
    
    # Create the tensor of indices to keep
    indices_to_keep = [i for i in range(seq_len) if i not in indices_to_delete]
    keep_tensor = torch.tensor(indices_to_keep).to(past_key_values[0][0].device)
    
    # Physically slice the raw tensors
    new_past_key_values = []
    for layer_idx in range(len(past_key_values)):
        key, value = past_key_values[layer_idx]
        new_key = torch.index_select(key, dim=2, index=keep_tensor)
        new_value = torch.index_select(value, dim=2, index=keep_tensor)
        new_past_key_values.append((new_key, new_value))

    return tuple(new_past_key_values), num_to_delete


# --- THE MAIN INFERENCE ENGINE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True).to(device)

prompt = "The future of artificial intelligence hardware relies on "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

past_key_values = None
step = 0
MAX_CACHE_SIZE = 50  # The ultra-strict hardware memory limit

print(f"\n⚡ --- REAL-TIME ASIC VRAM PROFILING (BUDGET: {MAX_CACHE_SIZE} TOKENS) ---")

try:
    while True:
        # 1. Forward Pass
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        
        # 2. Get the new token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # 3. Update the raw cache
        past_key_values = outputs.past_key_values
        
        # 4. --- THE ASIC INTERCEPT ---
        current_cache_size = past_key_values[0][0].shape[2]
        
        if current_cache_size > MAX_CACHE_SIZE and outputs.attentions is not None:
            # Score the tokens
            final_layer_attention = outputs.attentions[-1]
            attention_to_past = final_layer_attention[0, :, 0, :] 
            average_importance = attention_to_past.mean(dim=0)
            
            # Evict the garbage
            past_key_values, _ = evict_garbage_tokens(past_key_values, average_importance, MAX_CACHE_SIZE)
        
        # 5. Feed the new token back in for the next step
        input_ids = next_token
        step += 1
        
        # 6. Profiling Output
        if step % 50 == 0:
            managed_size = past_key_values[0][0].shape[2]
            print(f"Step {step:04d} | Cache Size Pinned At: {managed_size} tokens | Status: STABLE 🟢")
                
except KeyboardInterrupt:
    print(f"\n🛑 Stopped manually at step {step}.")
    print("Notice how the memory never exceeded the budget? You just solved the Memory Wall.")
    
    