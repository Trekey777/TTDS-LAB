import matplotlib.pyplot as plt

frequencies = []

with open("terms.freq") as f:
    for line in f:
        count,word = line.split()
        frequencies.append(int(count))

total = sum(frequencies)
probabilities = [count / total for count in frequencies]
# range[1,x)
ranks = range(1, len(frequencies) + 1)

plt.loglog(ranks, probabilities, ".")
plt.xlabel("Rank")
plt.ylabel("Probability")
plt.title("Zipf's Law")
plt.show()