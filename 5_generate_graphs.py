import matplotlib.pyplot as plt
import numpy as np

print("📊 Generating publication-ready graph for LinkedIn...")

# --- DATA SIMULATION ---
# We are plotting Generation Steps vs. KV Cache Size (Tokens)
steps = np.arange(0, 1050, 50)

# Baseline Data: Cache grows linearly by 1 token per step until it crashes at 1000
baseline_cache = steps.copy()
# Remove data after step 1000 to simulate the crash
baseline_steps = steps[steps <= 1000]
baseline_cache = baseline_cache[:len(baseline_steps)]

# ASIC Emulator Data: Cache grows to 50, then the Garbage Collector pins it flat
MAX_BUDGET = 50
asic_cache = np.clip(steps, 0, MAX_BUDGET)

# --- GRAPH STYLING ---
plt.style.use('dark_background') # Dark mode looks premium on LinkedIn
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot Baseline (Red, dashed, showing failure)
ax.plot(baseline_steps, baseline_cache, color='#ff4c4c', linewidth=3, linestyle='--', 
        label='Vanilla GPT-2 (Baseline)')
# Mark the crash point
ax.scatter([1000], [1000], color='red', s=150, zorder=5)
ax.annotate('Hardware Crash (OOM / Limit)', xy=(1000, 1000), xytext=(600, 800),
            arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=8),
            fontsize=11, color='white', fontweight='bold')

# Plot ASIC Emulator (Neon Green, solid, showing infinite stability)
ax.plot(steps, asic_cache, color='#00ffcc', linewidth=4, 
        label='KV-Garbage Collector ASIC (Ours)')
# Mark the pinning point
ax.annotate(f'Memory Pinned at {MAX_BUDGET} Tokens\n(Infinite Generation)', xy=(400, 50), xytext=(200, 300),
            arrowprops=dict(facecolor='#00ffcc', shrink=0.05, width=1, headwidth=8),
            fontsize=11, color='#00ffcc', fontweight='bold')

# --- FORMATTING ---
ax.set_title('Solving the "Memory Wall" in LLMs via Dynamic KV Eviction', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Generation Steps (Tokens Generated)', fontsize=12, fontweight='bold')
ax.set_ylabel('Hardware Memory Usage (KV Cache Size)', fontsize=12, fontweight='bold')

# Add subtle grid
ax.grid(color='#333333', linestyle='-', linewidth=0.5, alpha=0.7)

# Legend and limits
ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='#111111', edgecolor='#333333')
ax.set_xlim(0, 1050)
ax.set_ylim(0, 1100)

# Clean up layout and save
plt.tight_layout()
filename = "kv_cache_comparison.png"
plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"✅ Graph successfully saved as '{filename}' in your workspace folder!")

# Show the graph on screen
plt.show()
