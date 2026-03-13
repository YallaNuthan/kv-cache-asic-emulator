import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("🧠 Booting ASIC Logic: The Entropy Scorer...")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# FIX: We must explicitly alter the config during initialization to expose the matrices
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True).to(device)

# A test prompt to analyze
prompt = "The quick brown fox jumped over the lazy dog, but the dog did not seem to care very much."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Forward Pass 
with torch.no_grad():
    outputs = model(input_ids)

# Extract Attention Weights
# Shape of outputs.attentions[-1]: (batch_size, num_heads, sequence_length, sequence_length)
final_layer_attention = outputs.attentions[-1]

# Calculate Token Importance
# We look at the very last token generated, and average how much attention 
# it paid to every preceding token across all 12 attention heads.
attention_to_past = final_layer_attention[0, :, -1, :]  # Shape: (12, seq_len)
average_importance = attention_to_past.mean(dim=0)      # Shape: (seq_len)

# Rank the tokens
print("\n📊 --- TOKEN IMPORTANCE SCORES (ASIC VIEW) ---")
print(f"{'Token':<15} | {'Importance Score':<15}")
print("-" * 35)

# Pair tokens with their scores and sort them to find the "Garbage"
token_scores = []
for i in range(len(tokens)):
    score = average_importance[i].item()
    token_scores.append((tokens[i], score))

# Sort from lowest score (useless) to highest score (critical)
token_scores.sort(key=lambda x: x[1])

for token, score in token_scores:
    # Clean up the weird 'Ġ' character GPT-2 uses for spaces
    clean_token = token.replace('Ġ', '') 
    print(f"{clean_token:<15} | {score:.6f}")

print("\n🗑️ The top of this list is the 'garbage' our ASIC will dynamically delete!")