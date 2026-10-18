import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

token_positions = defaultdict(list)
tokens = []
with open("abstracts.wiki.txt", "r", encoding="utf-8") as f:
    for line in f:
        line_tokens = line.strip().split()
        tokens.extend(line_tokens)

for idx, word in enumerate(tokens):
    token_positions[word].append(idx)

N_total_tokens = len(tokens)
distances = []
for word, pos_list in token_positions.items():
    if len(pos_list) == 2:
        n1, n2 = pos_list
        d = n2 - n1
        distances.append(d)

distances = np.array(distances)

kde = gaussian_kde(distances)
x_range = np.linspace(0, np.max(distances), 2000)
y_kde = kde(x_range)

plt.figure(figsize=(8,6))
plt.plot(x_range, y_kde, color="#222222", label="Empirical KDE density")
plt.xlabel("distance (d)")
plt.ylabel("density")
plt.title("Clumping / Contagion test for twice-occurring words")
plt.legend(loc="upper right")
plt.grid(False)
plt.show()
