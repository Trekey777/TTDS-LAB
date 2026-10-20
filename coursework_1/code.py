"""
    TTDS COURSEWORK_1
"""
import math
import re
from collections import defaultdict
from nltk.stem import PorterStemmer
from pathlib import Path
import xml.etree.ElementTree as ET

# Initialize the Porter
stemmer = PorterStemmer()
IS_REMOVE_STOP_WORDS = True

if not IS_REMOVE_STOP_WORDS:
    stopwords = []
else:
    with open("./ttds_2023_english_stop_words.txt", "r") as f:
        lines = f.readlines()
        stopwords = [line.strip().lower() for line in lines]
"""
    IO Function
"""
def read_xml_file(file_name: str) -> dict[str, str]:
    tree = ET.parse(file_name)
    root = tree.getroot()
    doc_dict = {}
    # Get TEXT and HEADLINE in XML File
    for doc in root.findall("DOC"):
        doc_id = doc.find("DOCNO").text.strip()
        text = doc.find("HEADLINE").text.strip()
        text += "".join(doc.find("TEXT").itertext()).strip()
        doc_dict[doc_id] = text

    # sort documents by doc_id
    doc_dict = dict(sorted(
        doc_dict.items(),
        key=lambda item: int(item[0])))
    return doc_dict

def load_inverted_index(file_name: str) -> dict:
    inverted_index = {}

    with open(file_name, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError("Inverted index file is empty")

        label, value = first_line.split(":", 1)
        if label.strip() != "number of docs":
            raise ValueError(f"Unexpected first line: {first_line!r}")

        inverted_index["number of docs"] = int(value.strip())
        term = None

        for line in f:
            if not line.strip():
                continue

            if not line[0].isspace():
                term, df_text = line.strip().split(":", 1)
                term = term.strip()
                inverted_index[term] = {
                    "df": int(df_text.strip()),
                    "postings": {},
                }
            else:
                if term is None:
                    raise ValueError("Posting found before a term header")

                doc_id, positions_text = line.strip().split(":", 1)
                inverted_index[term]["postings"][doc_id.strip()] = [
                    int(position.strip())
                    for position in positions_text.split(",")
                    if position.strip()
                ]

    return inverted_index


def save_inverted_index(inverted_index: dict, file_name: str) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"number of docs:{inverted_index['number of docs']}\n")

        for token, data in inverted_index.items():
            if token == "number of docs":
                continue

            f.write(f"{token}:{data['df']}\n")

            for doc_id, positions in data["postings"].items():
                positions_text = ",".join(map(str, positions))
                f.write(f"\t{doc_id}: {positions_text}\n")

# support text query
def read_text_query(file_path) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        queries = f.readlines()
        query_list = list(map(str.strip, queries))
    return query_list

    # support boolean,phrase,proximity search
def read_normal_query(file_path) -> list:
    query_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n, q = line.strip().split(":", maxsplit = 1)
            query_list.append(q.strip())
    return query_list


class Index:
    """
        1. text_parse: lowercase, remove_stopwords, tokenize, stem
        2. build_inverted_index: build inverted index
    """
    def __init__(self, doc_dict:dict[str, str]):
        self.documents_number = len(doc_dict)
        self.doc_dict = doc_dict

    @staticmethod
    def text_parser(text:str, stemmer, stopwords) -> list[str]:
        text = text.lower()
        # tokenization
        tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
        # remove stopwords
        if stopwords:
            tokens = [token for token in tokens if token not in stopwords]
        # stem
        tokens = [stemmer.stem(t) for t in tokens]
        return tokens

    def doc_parser(self) -> dict[str,list[str]]:
        tokens_dict: dict[str, list[str]] = {}
        for doc_id, text in self.doc_dict.items():
            tokens_dict[doc_id] = Index.text_parser(text, stemmer, stopwords)
        return tokens_dict

    def build_inverted_index(self) -> dict:
        raw_index = defaultdict(
            lambda: {
                "df": 0,
                "postings": defaultdict(list),
            }
        )

        tokens_dict = self.doc_parser()

        for doc_id, tokens in tokens_dict.items():
            seen_tokens = set()

            for position, token in enumerate(tokens, start=1):
                raw_index[token]["postings"][doc_id].append(position)

                if token not in seen_tokens:
                    raw_index[token]["df"] += 1
                    seen_tokens.add(token)

        inverted_index = {
            "number of docs": self.documents_number,
            **{
                token: {
                    "df": data["df"],
                    "postings": dict(data["postings"]),
                }
                for token, data in sorted(raw_index.items())
            },
        }

        return inverted_index


class SearchEngine:
    """
    Search engine supporting the following features
        1. read_query
        2. query_parse
        3. boolean search
        4. phrase search
        5. proximity search
        6. ranked IR based on TFIDF
        7. calculate cosine similarity
    """
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index
        self.all_docs_set = {
            doc_id
            for token, info in self.inverted_index.items()
            if token != "number of docs"
            for doc_id in info["postings"]
        }
        self.term_pattern = re.compile(r"^([a-zA-Z0-9]+)$")
        self.phrase_pattern = re.compile(r'"([A-Za-z]+)\s+([A-Za-z]+)"')
        self.proximity_pattern = re.compile(r"#(\d+)\(\s*([A-Za-z]+)\s*, \s*([A-Za-z]+)\)")
        self.token_pattern = re.compile(
            r'#\d+\(\s*[A-Za-z]+\s*,\s*[A-Za-z]+\s*\)'
            r'|"[^"]*"'
            r'|\b(?:AND|OR|NOT)\b'
            r'|[A-Za-z0-9]+'
        )
        #
        self.forward_cached_tf_idf = defaultdict(dict)
        self.docs_norm = {}

    # calculate term_tf_idf
    def term_tf_idf(self, term, doc_id) -> float:
        if doc_id in self.forward_cached_tf_idf and term in self.forward_cached_tf_idf[doc_id]:
            term_weight = self.forward_cached_tf_idf[doc_id][term]
            return term_weight
        token_info= self.inverted_index.get(term, {})
        postings = token_info.get("postings", {})
        term_count = len(postings.get(doc_id, []))
        if not term_count:
            return 0.0
        term_weight =(1 + math.log10(term_count)) * self.idf(term)
        self.forward_cached_tf_idf[doc_id][term] = term_weight
        return term_weight
    # calculate vsm
    def calculate_query_norm(self, vsm_dict: dict[str, float]) -> float:
        query_norm = 0.0
        for token, term_weight in vsm_dict.items():
            query_norm += term_weight ** 2
        return math.sqrt(query_norm)

    # calculate doc_norms
    def get_doc_norms(self, doc_id):
        if doc_id in self.docs_norm:
            return self.docs_norm[doc_id]
        
        squared_norms = defaultdict(float)
        for token, token_info in self.inverted_index.items():
            if token == "number of docs":
                continue
            postings = token_info.get("postings", {})
            for docid, positons in postings.items():
                weight = self.term_tf_idf(token, docid)
                squared_norms[docid] += weight * weight

        for docid, squared_sum in squared_norms.items():
            self.docs_norm[docid] = math.sqrt(squared_sum)
        return self.docs_norm.get(doc_id, 0.0)

    # calculate idf of term
    def idf(self, term):
        token_info = self.inverted_index.get(term, {})
        if not len(token_info):
            raise ValueError("Term doesn't exist in any document")
        idf = math.log10(self.inverted_index["number of docs"] / token_info["df"])
        return idf

    def normal_single_term_index(self, term) -> dict[str, list[int]]:
        token = stemmer.stem(term.lower().strip())
        token_info = self.inverted_index.get(token)
        if token_info is None:
            return {}
        return token_info["postings"]

    def positions_compare(self, pos1, pos2, distance, exact):
        if exact:
            return abs(pos1 - pos2) == distance
        return abs(pos1 - pos2) <= distance
    # match the position
    def positions_match(self, w1, w2, distance, exact) -> set:
        indexes_1 = self.normal_single_term_index(w1)
        indexes_2 = self.normal_single_term_index(w2)
        common_doc_id = set(indexes_1.keys()) & set(indexes_2.keys())
        common_ret = set()
        for common_index in common_doc_id:
            file_1 = indexes_1[common_index]
            file_2 = indexes_2[common_index]
            i = j = 0
            while i < len(file_1) and j < len(file_2):
                pos_1 = int(file_1[i])
                pos_2 = int(file_2[j])
                distance = int(distance)
                if self.positions_compare(pos_1, pos_2, distance, exact):
                    common_ret.add(common_index)
                    break
                if int(file_1[i]) <= int(file_2[j]):
                    i += 1
                else:
                    j += 1
        return common_ret

    def normal_query_parse(self, query) -> set:
        terms = self.token_pattern.findall(query)
        pre_operator = None
        is_complement = False
        index_set_lists = []

        for t in terms:
            if t == 'NOT':
                is_complement = True
            elif t in ('AND', 'OR'):
                pre_operator = t
            else:
                ret_set = self.single_term_process(t)
                if is_complement:
                    ret_set = self.all_docs_set - ret_set
                    is_complement = False
                if not pre_operator or not index_set_lists:
                    index_set_lists.append(ret_set)
                elif pre_operator == 'AND':

                    index_set_lists[-1] &= ret_set
                elif pre_operator == 'OR':
                    index_set_lists.append(ret_set)

        final_sets = set()
        for s in index_set_lists:
            final_sets |= s
        return final_sets


    # process normal_query_parse
    def normal_query_batch_process(self, query_list) -> defaultdict[str,list]:
        result_dict = defaultdict(list)
        for q in query_list:
            ret = self.normal_query_parse(q)
            print(f"Query {q} successfully processed")
            result_dict[q] = sorted(ret, key = int)
        return result_dict

    # process batch of text query
    def text_query_batch_search(self, query_list, mode:str = "ltn.ltn") -> dict[str,dict[str, int]]:
        all_results = {}
        print(f"text query search mode: {mode}")
        for query in query_list:
            number, query_text = query.split(" ", maxsplit=1)
            search_results = self.text_search(query_text, mode)
            all_results[number] = search_results
        return all_results

    # return the final result of text query
    def text_search(self, text_query: str, mode):
        # process the term_list
        term_list = [
            stemmer.stem(word.lower())
            for word in text_query.split()
            if word.lower() not in stopwords
        ]
        return self.text_find_relevant_docs(term_list, mode)

    def calculate_query_vsm(self, term_list) -> defaultdict[str,float]:
        # calculate tf-idf
        query_vsm = defaultdict(float)

        # calculate term frequency in query
        for term in term_list:
            if term in self.inverted_index:
                query_vsm[term] = query_vsm.get(term, 0) + 1

        for key, value in query_vsm.items():
            query_vsm[key] = (1 + math.log10(value)) * self.idf(key)

        return query_vsm

    def calculate_score(self, term_list):
        query_vsm = self.calculate_query_vsm(term_list)

        # find doc have the same words
        common_doc = set()
        for key in query_vsm:
            if key in self.inverted_index:
                common_doc.update(self.inverted_index[key]["postings"].keys())
        docs = {}

        for doc_id in common_doc:
            score = 0
            for key in query_vsm.keys():
                score += self.term_tf_idf(key, doc_id) * query_vsm[key]
            docs[doc_id] = score
        docs = dict(sorted(docs.items(), key=lambda item: item[1], reverse=True))
        return docs
    def calculate_cosine_similarity(self, term_list)-> defaultdict[str,float]:
        query_vsm = self.calculate_query_vsm(term_list)
        query_norm = self.calculate_query_norm(query_vsm)
        if query_norm == 0:
            return {}
        # find doc have the same words
        common_doc = set()
        for key in query_vsm:
            if key in self.inverted_index:
                common_doc.update(self.inverted_index[key]["postings"].keys())
        docs = {}

        for doc_id in common_doc:
            doc_norm = self.get_doc_norms(doc_id)
            if doc_norm == 0.0 :
                continue
            score = 0
            for key in query_vsm.keys():
                score += self.term_tf_idf(key, doc_id) * query_vsm[key]
            docs[doc_id] = score /(doc_norm * query_norm)
        docs = dict(sorted(docs.items(), key=lambda item: item[1], reverse=True))
        return docs
    # find text search relevant_docs
    def text_find_relevant_docs(self, term_list, mode) -> dict:

        if mode == "ltn.ltn":
            return self.calculate_score(term_list)

        elif mode == "ltc.ltc":
            return self.calculate_cosine_similarity(term_list)
        return {}

    def phrase_match(self, w1, w2, distance) -> set:
        return self.positions_match(w1, w2, distance, True)

    def proximity_match(self, w1, w2, distance) -> set:
        return self.positions_match(w1, w2, distance, False)

    def single_term_process(self, term) -> set:
        term_match = self.term_pattern.fullmatch(term)
        if term_match:
            term = term_match.group(1)
            return set(self.normal_single_term_index(term).keys())

        phrase_match = self.phrase_pattern.fullmatch(term)
        if phrase_match:
            w1 = phrase_match.group(1)
            w2 = phrase_match.group(2)
            term_distance = 1

            return self.phrase_match(w1, w2, term_distance)

        proximity_match = self.proximity_pattern.fullmatch(term)
        if proximity_match:
            term_distance = proximity_match.group(1)
            w1 = proximity_match.group(2)
            w2 = proximity_match.group(3)
            return self.proximity_match(w1, w2, term_distance)

        return set()



if __name__ == '__main__':
    index_file = "./collections/trec.sample.xml"
    index_save_file = "./index.txt"
    text_query = "./text_query.txt"
    text_save_file = "./text_query_result.txt"
    normal_query = "./normal_query.txt"
    normal_query_save_file = "./normal_query_result.txt"
    doc_dict = read_xml_file(index_file)
    I = Index(doc_dict)

    inverted_index = I.build_inverted_index()
    save_inverted_index(inverted_index, index_save_file)

    inverted_index = load_inverted_index(index_save_file)
    se = SearchEngine(inverted_index)
    text_query_list = read_text_query(text_query)
    print(text_query_list)
    text_results = se.text_query_batch_search(text_query_list,"ltc.ltc")
    with open(text_save_file, "w", encoding="utf-8") as f:
        for n, result in text_results.items():
            for key, value in result.items():
                f.write(f"{n},{key},{value}\n")
            f.write("\n")

    normal_query_list = read_normal_query(normal_query)
    print(text_query_list)
    normal_query_results = se.normal_query_batch_process(normal_query_list)
    with open(normal_query_save_file, "w", encoding="utf-8") as f:
        for n, result in normal_query_results.items():
            f.write(f"{n},{result}\n")

