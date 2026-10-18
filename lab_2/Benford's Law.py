import math
import matplotlib.pyplot as plt
from collections import Counter

# 1. Load your word counts
counts = []
with open("terms.freq", "r") as f:
    for line in f:
        cnt_str, word = line.strip().split()
        cnt = int(cnt_str)
        counts.append(cnt)

# 2. Function to get leading digit of a positive integer
def leading_digit(n: int) -> int:
    # e.g. 729 -> 7, 15 -> 1, 4821 ->4
    while n >= 10:
        n = n // 10
    return n

# collect all leading digits
lead_digits = [leading_digit(c) for c in counts]
counter = Counter(lead_digits)

# 3. Empirical probability from data
total = len(lead_digits)
empirical = {}
for d in range(1,10):
    empirical[d] = counter.get(d, 0) / total

# 4. Benford theoretical probability
benford = {}
for d in range(1,10):
    benford[d] = math.log10(1 + 1/d)

# print table
print("digit | empirical | benford theory")
print("------|-----------|---------------")
for d in range(1,10):
    print(f"{d:5d} | {empirical[d]:.4f} | {benford[d]:.4f}")

# 5. Plot comparison
digits = list(range(1,10))
emp_vals = [empirical[d] for d in digits]
ben_vals = [benford[d] for d in digits]

plt.bar([x-0.2 for x in digits], emp_vals, width=0.4, label="Empirical (word counts)")
plt.bar([x+0.2 for x in digits], ben_vals, width=0.4, label="Benford predicted")
plt.xticks(digits)
plt.xlabel("Leading digit d")
plt.ylabel("Probability")
plt.title("Benford's Law Check on Word Frequency Counts")
plt.legend()
plt.show()
