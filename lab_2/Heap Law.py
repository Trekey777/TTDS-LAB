import matplotlib.pyplot as plt
import numpy as np

tokens = []
# Read raw text file
with open("Bible.txt", "r", encoding="utf-8") as f:
    for line in f:
        tokens.extend(line.strip().split())

seen_words = set()
N_list = []
V_list = []

for idx, word in enumerate(tokens):
    seen_words.add(word)
    N = idx + 1
    # Record a point every 200 tokens to reduce data points
    if N % 200 == 0:
        N_list.append(N)
        V_list.append(len(seen_words))

# ========== MANUALLY SET K AND BETA HERE ==========
K_manual = 9
beta_manual = 0.6
# ===================================================

# Log-log plot for observed data points
plt.plot(N_list, V_list, ".", label="Observed data")

# Generate theoretical Heap's law curve with manual parameters
N_array = np.array(N_list)
V_theory = K_manual * (N_array ** beta_manual)
plt.plot(N_array, V_theory, "-", label=f"Theory $V={K_manual} N^{beta_manual}$")

plt.xlabel("Total tokens N")
plt.ylabel("Vocabulary size V")
plt.title("Heap's Law Verification (manual K, beta)")
plt.legend()

# Keep automatic fitting for comparison
log_N = np.log10(N_list)
log_V = np.log10(V_list)
beta_fit, logK_fit = np.polyfit(log_N, log_V, deg=1)
K_fit = 10 ** logK_fit

print(f"Fitted result: V = {K_fit:.2f} * N^{beta_fit:.3f}")
print(f"Manual parameters: V = {K_manual:.2f} * N^{beta_manual:.3f}")

plt.show()
