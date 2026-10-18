from pathlib import  Path
from nltk.stem import PorterStemmer
from collections import defaultdict
from index import stop_words
import re
import math

stemmer = PorterStemmer()
index_dir = Path(r"./output")
index_path = index_dir / "trec.sample.index.txt"

def read_index(index_path):
    indexes = defaultdict(lambda: defaultdict(list))
    current_term = None
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if  line.endswith(":"):
                current_term = line[:-1]
            else:
                docno, positions = line.split(":", 1)
                indexes[current_term][docno] = [
                    int(pos.strip()) for pos in positions.strip().split(",")
                ]
    return indexes

# search_module.py
def read_query(query_path) -> list:
    queries = []
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n, q = line.strip().split(":", 1)
            queries.append(q.strip())
    return queries




class SearchQuery:
    def __init__(self, indexes, stemmer):
        self.indexes = indexes
        self.stemmer = stemmer
        self.all_docs_set = {str(i) for i in range(1, 3948)}
        self.term_pattern = re.compile(r"^([a-zA-Z0-9]+)$")
        self.phrase_pattern = re.compile(r'"([A-Za-z]+)\s+([A-Za-z]+)"')
        self.proximity_pattern = re.compile(r"#(\d+)\(\s*([A-Za-z]+)\s*, \s*([A-Za-z]+)\)")
        self.token_pattern = re.compile(
            r'#\d+\(\s*[A-Za-z]+\s*,\s*[A-Za-z]+\s*\)'
            r'|"[^"]*"'
            r'|\b(?:AND|OR|NOT)\b'
            r'|[A-Za-z0-9]+',
            re.IGNORECASE,
        )

    def tokenization(self, term):
        return self.stemmer.stem(term.lower().strip())

    def single_term_index(self, term) -> dict[str, list[int]]:
        token =  self.tokenization(term)
        return self.indexes[token]

    def positions_compare(self, pos1, pos2, distance, exact):
        if exact:
            return abs(pos1 - pos2) == distance
        return abs(pos1 - pos2) <= distance

    def positions_match(self, w1, w2, distance, exact) -> set:
        indexes_1 = self.single_term_index(w1)
        indexes_2 = self.single_term_index(w2)
        common_docn = set(indexes_1.keys()) & set(indexes_2.keys())
        common_ret = set()
        for common_index in common_docn:
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
                if file_1[i] <= file_2[j]:
                    i += 1
                else:
                    j += 1
        return common_ret
    def phrase_match(self, w1, w2, distance) -> set:
        return self.positions_match(w1, w2, distance, True)
    def proximity_match(self, w1, w2, distance) -> set:
        return self.positions_match(w1, w2, distance, False)
    def single_term_process(self, term) -> set:
        term_match = self.term_pattern.fullmatch(term)
        term_distance = None
        if term_match:
            term = term_match.group(1)
            return set(self.single_term_index(term).keys())

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

    def query_parse(self, query) -> set:
        terms = self.token_pattern.findall(query)
        print(terms)
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
                    if ret_set >= index_set_lists[-1] :
                        print("empty")
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

    def batch_query_process(self, query_list) -> defaultdict[str,list]:
        result_dict = defaultdict(list)
        for q in query_list:
            ret = self.query_parse(q)
            print(f"Query {q} successfully processed")
            result_dict[q] = sorted(ret, key = int)
        return result_dict
    def save_query_output(self, output_path, result_dict):
        output_file = Path(output_path)
        with output_file.open("w", encoding="utf-8") as f:
            for query, result in result_dict.items():
                f.write(f"{query}:\n")
                f.write(", ".join(map(str, result)))
                f.write("\n\n")


class RankedSearch:
    def __init__(self, index, stemmer, document_number):
        self.index = index
        self.stemmer = stemmer
        self.document_number = document_number

    def term_tf_idf(self, term, docn) -> float:
        postings = self.index.get(term, {})
        term_count = len(postings.get(docn,[]))
        if not term_count:
            return 0.0
        term_weight =(1 + math.log10(term_count)) * self.idf(term)
        return term_weight

    def idf(self, term):
        doc_collect = len(self.index.get(term, {}))
        if not doc_collect:
            raise ValueError("Term doesn't exist in any document")
        idf = math.log10(self.document_number / doc_collect)
        return idf

    def find_relevant_docs(self, term_list) -> dict:
        # calculate tf-idf
        query_vsm = {}
        # calculate term frequency
        for term in term_list:
            query_vsm[term] = query_vsm.get(term, 0) + 1

        for key, value in query_vsm.items():
            query_vsm[key] = (1 + math.log10(value)) * self.idf(key)
        common_doc = set()
        for key, value in query_vsm.items():
            common_doc |= set(self.index[key].keys())
        docs = {}

        for docn in common_doc:
            score = 0
            for key in query_vsm.keys():
                score += self.term_tf_idf(key, docn) * query_vsm[key]
            docs[docn] = score
        docs = dict(sorted(docs.items(), key=lambda item: item[1], reverse=True))
        return docs

    def query_parse(self, query) -> list:
        # remove stop word and tokenize
        term_list = [
            self.stemmer.stem(word.lower())
            for word in query.split()
            if word.lower() not in stop_words
        ]
        return term_list

    def search(self, query):
        term_list = self.query_parse(query)
        return self.find_relevant_docs(term_list)

    def batch_query_search(self, query_list) -> dict[str,dict[str, int]]:
        all_results = {}
        for query in query_list:
            number, query_text = query.split(maxsplit=1)
            search_results = self.search(query_text)
            all_results[number] = search_results
        return all_results

if __name__ == "__main__":
    with open("./queries_retrieved.txt", "r", encoding="utf-8") as f:
        queries = f.readlines()
        query_list = list(map(str.strip, queries))
    indexes = read_index(index_path)
    document_number = 3947
    rk = RankedSearch(indexes, stemmer, document_number)
    all_results = rk.batch_query_search(query_list)
    output_path = "./tfidf.results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for n, result in all_results.items():
            for key, value in result.items():
                f.write(f"{n},{key},{value}\n")
            f.write("\n")

