import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "or",
    "our",
    "the",
    "their",
    "to",
    "used",
    "using",
    "we",
    "what",
    "which",
    "with",
}


QUERY_EXPANSIONS = {
    "33 grade": "ordinary portland cement 33 grade opc chemical physical",
    "43 grade": "ordinary portland cement 43 grade opc",
    "53 grade": "ordinary portland cement 53 grade opc",
    "aggregate": "coarse fine aggregate aggregates natural sources concrete",
    "calcined clay": "portland pozzolana cement part 2 calcined clay based",
    "fly ash": "portland pozzolana cement part 1 fly ash based",
    "slag cement": "portland slag cement",
    "white portland": "white portland cement architectural decorative",
    "rapid hardening": "rapid hardening portland cement",
    "hydrophobic": "hydrophobic portland cement",
    "sulphate resisting": "sulphate resisting portland cement",
    "supersulphated": "supersulphated cement marine aggressive sulphate",
    "masonry cement": "masonry cement mortars masonry not structural concrete",
    "masonry mortars": "sand masonry mortars",
    "precast concrete pipes": "precast concrete pipes with without reinforcement water mains sewers culverts irrigation",
    "water mains": "pipes pressure non pressure water mains",
    "hollow and solid lightweight": "concrete masonry units part 2 hollow solid lightweight concrete blocks",
    "lightweight concrete masonry blocks": "concrete masonry units part 2 hollow solid lightweight concrete blocks",
    "concrete masonry blocks": "concrete masonry units hollow solid blocks",
    "corrugated and semi-corrugated": "corrugated semi corrugated asbestos cement sheets roofing cladding",
    "roofing and cladding": "sheets roofing cladding",
}


MATERIAL_TERMS = {
    "aggregate",
    "aggregates",
    "asbestos",
    "bitumen",
    "cement",
    "concrete",
    "corrugated",
    "gypsum",
    "lime",
    "lightweight",
    "masonry",
    "pozzolana",
    "precast",
    "slag",
    "steel",
    "supersulphated",
    "timber",
    "white",
}


def tokenize(text):
    return [
        token
        for token in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
        if token not in STOPWORDS
    ]


def normalize_standard(standard):
    standard = re.sub(r"\s+", " ", str(standard).strip()).upper()
    standard = re.sub(r"\(\s*PART\s*(\d+)\s*\)", r"(Part \1)", standard, flags=re.I)
    standard = re.sub(r"\s*:\s*", ": ", standard)
    standard = re.sub(r"^IS\s*", "IS ", standard)
    return standard


class BISRetriever:
    def __init__(self, data_path=None):
        root = Path(__file__).resolve().parents[1]
        self.data_path = Path(data_path) if data_path else root / "data" / "processed_data.json"
        self.documents = json.loads(self.data_path.read_text(encoding="utf-8"))
        self._prepare()

    def _prepare(self):
        self.doc_tokens = []
        self.doc_term_counts = []
        document_frequency = Counter()

        for doc in self.documents:
            standard = normalize_standard(doc.get("standard", ""))
            title = doc.get("title") or self._title_from_text(doc.get("text", ""))
            doc["standard"] = standard
            doc["title"] = title

            weighted_text = " ".join(
                [
                    (standard + " ") * 8,
                    (title + " ") * 12,
                    doc.get("text", ""),
                ]
            )
            tokens = tokenize(weighted_text)
            counts = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_term_counts.append(counts)
            document_frequency.update(set(tokens))

        self.num_docs = len(self.documents)
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(self.num_docs, 1)
        self.idf = {
            term: math.log(1 + (self.num_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _title_from_text(self, text):
        text = re.sub(r"\s+", " ", text)
        match = re.search(
            r"IS\s+\d+(?:\s*\(\s*PART\s*\d+\s*\))?\s*:\s*\d{4}\s+(.+)",
            text,
            flags=re.I,
        )
        if not match:
            return ""
        title = re.split(
            r"\s+(?:\([^)]*Revision\)|1\.\s*Scope|Scope\s*[—-]|1\.\s|2\.\s|TABLE\s+|Note\s*[—-])",
            match.group(1),
            maxsplit=1,
            flags=re.I,
        )[0]
        return re.sub(r"\s+", " ", title).strip(" -—:")[:250]

    def _expanded_query(self, query):
        lowered = query.lower()
        additions = [value for key, value in QUERY_EXPANSIONS.items() if key in lowered]
        return " ".join([query, *additions])

    def _explicit_standard_matches(self, query):
        matches = re.findall(
            r"\bIS\s*[:\-]?\s*(\d{2,5})(?:\s*\(?\s*Part\s*(\d+)\s*\)?)?\s*[:\-]?\s*(\d{4})?",
            query,
            flags=re.I,
        )
        explicit = set()
        for number, part, year in matches:
            wanted = f"is{number}"
            if part:
                wanted += f"(part{part})"
            for index, doc in enumerate(self.documents):
                standard_key = doc["standard"].lower().replace(" ", "")
                if wanted in standard_key and (not year or year in standard_key):
                    explicit.add(index)
        return explicit

    def retrieve(self, query, top_k=5):
        query_tokens = tokenize(self._expanded_query(query))
        query_counts = Counter(query_tokens)
        explicit_matches = self._explicit_standard_matches(query)

        scored = []
        for index, doc in enumerate(self.documents):
            score = self._bm25_score(query_counts, index)
            title_tokens = set(tokenize(f"{doc['standard']} {doc.get('title', '')}"))
            score += 2.8 * sum(
                1 for token in set(query_tokens) if len(token) > 2 and token in title_tokens
            )
            for token in MATERIAL_TERMS.intersection(query_tokens):
                score += 5.0 if token in title_tokens else -1.5
            if index in explicit_matches:
                score += 100.0
            scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "standard": self.documents[index]["standard"],
                "title": self.documents[index].get("title", ""),
                "score": score,
            }
            for score, index in scored[:top_k]
        ]

    def _bm25_score(self, query_counts, doc_index):
        counts = self.doc_term_counts[doc_index]
        doc_len = len(self.doc_tokens[doc_index])
        score = 0.0
        k1 = 1.4
        b = 0.70

        for token, query_count in query_counts.items():
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += (
                self.idf.get(token, 0.0)
                * frequency
                * (k1 + 1)
                / denominator
                * (1 + 0.10 * (query_count - 1))
            )

        return score


@lru_cache(maxsize=1)
def get_retriever():
    return BISRetriever()
