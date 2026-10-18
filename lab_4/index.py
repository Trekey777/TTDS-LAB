# data preprocessing
import re
from tarfile import REGULAR_TYPES
from typing import Any

from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from pathlib import Path
import xml.etree.ElementTree as ET

# Initialize the Porter
stemmer = PorterStemmer()

DATA_DIR = Path("./collections")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTIONS = ["sample","trec.sample"]

IS_REMOVE_STOP_WORDS = True
STOPLIST_FILE = r"./englishST.txt"
stop_words_pattern = re.compile(r"^\s*rank:\s+\d+\s+([^s]+)\s+\d+\s*")

# rank: 82	m	80609

def load_stopwords(path) -> set:

    sw = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = stop_words_pattern.match(line)
            if match:
                sw.add(match.group(1))
        return sw

# print(load_stopwords(STOPLIST_FILE))
def sample_process_xml(path) -> dict[str, str]:
    tree = ET.parse(path)
    root = tree.getroot()

    documents = {}

    for doc in root.findall("DOC"):
        docno = doc.find("DOCNO").text.strip()
        text = "".join(doc.find("Text").itertext()).strip()
        documents[docno] = text

    # sort documents by docno
    documents = dict(sorted(
        documents.items(),
        key=lambda item: int(item[0])))

    return documents

def trec_sample_process_xml(path) -> dict[str, str]:
    tree = ET.parse(path)
    root = tree.getroot()
    documents = {}
    for doc in root.findall("DOC"):
        docno = doc.find("DOCNO").text.strip()
        # text = "".join(doc.find("HEADLINE").itertext()).strip()
        text = "".join(doc.find("TEXT").itertext()).strip()
        documents[docno] = text
    # sort documents by docno
    documents = dict(sorted(
        documents.items(),
        key=lambda item: int(item[0])))

    return documents
# print(process_xml(COLLECTIONS["sample"]))
def tokenize(doc_dict, stop_words = None) -> dict[str, list[str]]:
    for index, text in doc_dict.items():
        text = text.lower()
        text = re.sub("[^a-z0-9]", " ", text)
        tokens = text.split()
        if stop_words:
            tokens = [t for t in tokens if t not in stop_words]
        tokens = [stemmer.stem(t) for t in tokens]
        doc_dict[index] = tokens
    return doc_dict

def create_inverted_index(doc_dict) -> dict[Any, defaultdict[Any, list[Any]]]:
    inverted_index = defaultdict(lambda: defaultdict(list))
    for docno, tokens in doc_dict.items():
        for index, token in enumerate(tokens):
            inverted_index[token][docno].append(index + 1)

    inverted_index = dict(sorted(inverted_index.items()))
    return inverted_index

def save_inverted_index(inverted_index, path):
    with open(path, "w", encoding="utf-8") as f:
        for token, postings in inverted_index.items():
            f.write(f"{token}:\n")

            for docno, positions in postings.items():
                positions_str = ",".join(map(str, positions))
                f.write(f"\t{docno}: {positions_str}\n")
            f.write("\n")



stop_words = load_stopwords(STOPLIST_FILE)

if __name__ == "__main__":
    if IS_REMOVE_STOP_WORDS:
        stop_words = None
    else:
        stop_words = load_stopwords(STOPLIST_FILE)
    for name in COLLECTIONS:
        input_path = DATA_DIR / f"{name}.xml"
        if stop_words:
            output_path = OUTPUT_DIR / f"{name}_stop_words_index.txt"
        else:
            output_path = OUTPUT_DIR / f"{name}.index.txt"
        if name == "sample":
            documents = sample_process_xml(input_path)
        else:
            documents = trec_sample_process_xml(input_path)
        doc_dict = tokenize(documents, stop_words)
        inverted_index = create_inverted_index(doc_dict)
        save_inverted_index(inverted_index, output_path)