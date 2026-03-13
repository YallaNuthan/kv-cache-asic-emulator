import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("🚀 Booting Vanilla Baseline (Waiting for OOM crash)...")

# 1. Enforce Hardware Acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("⚠️ WARNING: No GPU detected. This will run on system RAM and may take a long time to crash.")

# 2. Load the Model & Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 3. The Seed Prompt
prompt = "The future of artificial intelligence hardware relies on "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

past_key_values = None
step = 0

print("\n📊 --- REAL-TIME VRAM PROFILING ---")

try:
    # 4. The Infinite Generation Loop
    while True:
        # Forward pass (use_cache=True forces the model to save every token to memory)
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        
        # Grab the newly generated token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Update the massive cache and feed the new token back in
        past_key_values = outputs.past_key_values
        input_ids = next_token
        
        step += 1
        
        # Profile the memory every 50 steps
        if step % 50 == 0:
            if device == "cuda":
                mem_alloc = torch.cuda.memory_allocated() / (1024**2) # Convert to Megabytes
                print(f"Step {step:04d} | VRAM Used: {mem_alloc:.2f} MB")
            else:
                print(f"Step {step:04d} | Generating...")
                
except torch.cuda.OutOfMemoryError:
    print(f"\n❌ FATAL: CUDA Out of Memory at step {step}!")
    print("The KV Cache grew too large and hit the Memory Wall.")
    print("This is the exact hardware bottleneck our ASIC will solve.")
except KeyboardInterrupt:
    print(f"\n🛑 Stopped manually at step {step}.")