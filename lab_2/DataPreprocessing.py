"""
TTDS Preprocessing — using already-downloaded local files
"""
import re
import nltk
from nltk.stem import PorterStemmer
from pathlib import Path

stemmer = PorterStemmer()

# ============================================================
# 1. File paths — adjust these if your files are elsewhere
# ============================================================
DATA_DIR = Path(".")          # folder where Bible.txt etc. live
OUT_DIR  = Path("output")
OUT_DIR.mkdir(exist_ok=True)

COLLECTIONS = {
    "bible":    DATA_DIR / "Bible.txt",
    "quran":    DATA_DIR / "quran.txt",
    "wikipedia": DATA_DIR / "abstracts.wiki.txt",
}

# Savoy stoplist — put englishST.txt in the same folder,
# or download it from http://members.unine.ch/jacques.savoy/clef/index.html
STOPLIST_FILE = DATA_DIR / "englishST.txt"

# ============================================================
# 2. Load stopwords
# ============================================================
def load_stopwords(path):
    sw = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                sw.add(w)
    return sw

stopwords = load_stopwords(STOPLIST_FILE)
print(f"Loaded {len(stopwords)} stopwords.\n")

# ============================================================
# 3. Preprocessing
# ============================================================
def preprocess(text):
    # (a) Case folding
    text = text.lower()
    # (b) Tokenisation: strip punctuation, keep alphanumerics
    text = re.sub(r"[^a-z0-9]", " ", text)
    tokens = text.split()
    # (c) Stopword removal
    tokens = [t for t in tokens if t not in stopwords]
    # (d) Porter stemming
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# ============================================================
# 4. Process each collection
# ============================================================
stats = {}

for name, src_path in COLLECTIONS.items():
    out_path = OUT_DIR / f"{name}_processed.txt"
    print(f"=== {name} ===")

    raw_tokens = 0
    vocab = set()

    with open(src_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            toks = preprocess(line)
            raw_tokens += len(toks)
            vocab.update(toks)
            fout.write(" ".join(toks) + "\n")

    stats[name] = (raw_tokens, len(vocab), out_path)
    print(f"  tokens: {raw_tokens:,} | vocab: {len(vocab):,}")
    print(f"  -> {out_path}\n")

# ============================================================
# 5. Summary
# ============================================================
print("=" * 55)
print(f"{'Collection':<12} {'Tokens':>12} {'Vocab':>10}")
print("-" * 55)
for name, (tok, voc, _) in stats.items():
    print(f"{name:<12} {tok:>12,} {voc:>10,}")
