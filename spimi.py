"""
spimi.py
--------
SPIMI (Single-Pass In-Memory Indexing) implementation as an alternative
to the BSBI (Blocked Sort-Based Indexing) approach in bsbi.py.

Key differences between SPIMI and BSBI
---------------------------------------
BSBI:
  - Collects ALL <termID, docID> pairs from a block first, then sorts them.
  - Requires a global term-to-ID mapping maintained across blocks.
  - Sort step is O(T log T) where T = total pairs in block.

SPIMI:
  - Processes tokens ONE BY ONE and builds the inverted index on the fly.
  - Maintains a per-block in-memory dictionary (term_str -> postings list).
  - When memory is full, writes the current in-memory index to disk (a "spill").
  - Does NOT need a global term-ID mapping during indexing — term strings
    are used directly in the in-memory dict and only converted to IDs at merge.
  - More memory-efficient for large collections because it avoids storing
    all pairs before sorting.
  - Final merge is an N-way merge of sorted spill files (same as BSBI merge).

Memory management
-----------------
We simulate memory limits using a configurable max_tokens_per_block parameter.
When the total number of token occurrences accumulated exceeds this threshold,
the current in-memory index is sorted, written to disk, and the memory is freed.

Usage
-----
    from spimi import SPIMIIndex
    from compression import VBEPostings

    index = SPIMIIndex(
        data_dir='collection',
        output_dir='index',
        postings_encoding=VBEPostings,
        index_name='main_index',
        max_tokens_per_block=200_000   # tune based on available RAM
    )
    index.index()

After indexing, the SPIMIIndex is fully compatible with BSBIIndex for retrieval:
    results = index.retrieve_bm25("query terms", k=10)
"""

import os
import pickle
import heapq
import math
import contextlib

from index import InvertedIndexReader, InvertedIndexWriter
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from util import IdMap, sorted_merge_posts_and_tfs
from patricia_tree import PatriciaIdMap

try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False
    def tqdm(x, **kwargs):
        return x


class SPIMIIndex:
    """
    SPIMI-based Inverted Index builder.

    Unlike BSBIIndex which collects all (termID, docID) pairs per block and
    then sorts, SPIMI builds the in-memory postings dictionary incrementally.
    When memory fills up, it spills the current dictionary to disk and starts
    a fresh one.  The spill files are then merged exactly like BSBI.

    Attributes
    ----------
    term_id_map : PatriciaIdMap
        Patricia Tree-based term dictionary. Supports O(k) lookup AND
        prefix search — richer than a plain Python dict.
    doc_id_map : IdMap
        Document path -> integer ID mapping.
    data_dir : str
    output_dir : str
    postings_encoding : class
        Compression class (VBEPostings, EliasGammaPostings, etc.)
    index_name : str
    max_tokens_per_block : int
        How many token occurrences to accumulate before spilling to disk.
        Lower = less RAM, more spill files. Higher = more RAM, fewer spills.
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", max_tokens_per_block=200_000):
        self.term_id_map = PatriciaIdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.max_tokens_per_block = max_tokens_per_block

        self.spill_files = []   


    def save(self):
        """Save term_id_map and doc_id_map to output_dir."""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load term_id_map and doc_id_map from output_dir."""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def _spimi_invert(self, token_stream):
        """
        Core SPIMI invert function (Algorithm 4.2 from IIR textbook).

        Processes a stream of (token_str, doc_id) pairs, building an in-memory
        inverted index.  Returns when the block memory limit is reached OR the
        stream is exhausted.

        Parameters
        ----------
        token_stream : iterator of (str, int)
            Yields (token_string, doc_id) pairs one at a time.

        Returns
        -------
        dict : {term_str -> {doc_id -> tf}}
            In-memory inverted index for this block.
        int  : number of tokens processed in this call.
        bool : True if the stream was fully consumed, False if we hit the limit.
        """
        index = {}         
        token_count = 0

        for token, doc_id in token_stream:
            if token not in index:
                index[token] = {}
            if doc_id not in index[token]:
                index[token][doc_id] = 0
            index[token][doc_id] += 1
            token_count += 1

            if token_count >= self.max_tokens_per_block:
                return index, token_count, False   

        return index, token_count, True   

    def _write_spill(self, in_memory_index, spill_name):
        """
        Sort the in-memory index by term string and write it as an
        intermediate inverted index file (a "spill").

        We sort by term STRING (not termID), because SPIMI doesn't require
        a global term-ID mapping during indexing.  TermIDs are assigned
        lazily via self.term_id_map during this write step.

        Parameters
        ----------
        in_memory_index : dict {str -> {int -> int}}
        spill_name : str
        """
        with InvertedIndexWriter(spill_name, self.postings_encoding,
                                 directory=self.output_dir) as writer:
            for term_str in sorted(in_memory_index.keys()):
                # Assign term ID now (via PatriciaIdMap)
                term_id = self.term_id_map[term_str]
                doc_tf_dict = in_memory_index[term_str]
                sorted_docs = sorted(doc_tf_dict.keys())
                tf_list = [doc_tf_dict[d] for d in sorted_docs]
                writer.append(term_id, sorted_docs, tf_list)

        self.spill_files.append(spill_name)

    def _token_generator(self):
        """
        Generator that yields (token_str, doc_id) pairs by walking the
        entire data_dir collection.

        Yields
        ------
        (str, int) : (token, doc_id)
        """
        blocks = sorted(next(os.walk(self.data_dir))[1])
        for block_dir in blocks:
            block_path = os.path.join(self.data_dir, block_dir)
            try:
                filenames = next(os.walk(block_path))[2]
            except StopIteration:
                continue
            for filename in filenames:
                doc_path = "./" + self.data_dir + "/" + block_dir + "/" + filename
                doc_id = self.doc_id_map[doc_path]
                try:
                    with open(doc_path, "r", encoding="utf8",
                              errors="surrogateescape") as f:
                        for token in f.read().split():
                            yield token, doc_id
                except (IOError, OSError):
                    continue

    def merge(self, indices, merged_index):
        """
        N-way external merge sort of all spill files into one final index.
        Identical to BSBIIndex.merge().
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def index(self):
        """
        Full SPIMI indexing pipeline:

        1. Create a single-pass token generator over the entire collection.
        2. Repeatedly call _spimi_invert() until the stream is exhausted,
           writing a spill file each time memory fills up.
        3. Merge all spill files into one final inverted index.
        4. Save term_id_map and doc_id_map to disk.

        This approach is more memory-efficient than BSBI because:
        - We never sort a huge list of (termID, docID) pairs.
        - Each in-memory block is a dict, which is insert-optimal.
        - Spill files are sorted by term string (not termID), and merge
          uses a heap for O(N log K) total cost.
        """
        print("SPIMI Indexing started...")
        print(f"  Using PatriciaIdMap as term dictionary")
        print(f"  Max tokens per spill block: {self.max_tokens_per_block:,}")

        token_stream = self._token_generator()
        spill_count = 0
        total_tokens = 0
        exhausted = False

        while not exhausted:
            in_memory_index, tokens_processed, exhausted = self._spimi_invert(token_stream)

            if not in_memory_index:
                break

            total_tokens += tokens_processed
            spill_name = f"spimi_spill_{spill_count:04d}"
            print(f"  Writing spill {spill_count} "
                  f"({len(in_memory_index):,} unique terms, "
                  f"{tokens_processed:,} tokens)...")
            self._write_spill(in_memory_index, spill_name)
            spill_count += 1
            in_memory_index = None   

        print(f"  Total: {spill_count} spill file(s), {total_tokens:,} tokens indexed")
        print(f"  Vocabulary size: {len(self.term_id_map):,} unique terms")

        print("  Merging spill files...")
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(spill_name, self.postings_encoding,
                                            directory=self.output_dir)
                    )
                    for spill_name in self.spill_files
                ]
                if indices:
                    self.merge(indices, merged_index)

        self.save()
        print("SPIMI Indexing complete!")


    def retrieve_tfidf(self, query, k=10):
        """TF-IDF retrieval. See BSBIIndex.retrieve_tfidf for full docs."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    for doc_id, tf in zip(postings, tf_list):
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))
            docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """BM25 retrieval. See BSBIIndex.retrieve_bm25 for full docs."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for doc_id, tf in zip(postings, tf_list):
                        dl = merged_index.doc_length.get(doc_id, avgdl)
                        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm
            docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def prefix_search(self, prefix):
        """
        Return all indexed terms that start with the given prefix.
        Only possible because we use PatriciaIdMap as the term dictionary!

        Parameters
        ----------
        prefix : str

        Returns
        -------
        List[str]
            Sorted list of matching term strings.
        """
        if len(self.term_id_map) == 0:
            self.load()
        return self.term_id_map.starts_with(prefix)

if __name__ == "__main__":
    import sys

    encoding = VBEPostings

    spimi = SPIMIIndex(
        data_dir='collection',
        output_dir='index',
        postings_encoding=encoding,
        index_name='main_index',
        max_tokens_per_block=100_000
    )
    spimi.index()

    print("\n--- Sample retrieval (BM25) ---")
    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
    ]
    for q in queries:
        print(f"Query: {q}")
        for score, doc in spimi.retrieve_bm25(q, k=5):
            print(f"  {score:.3f}  {doc}")
        print()

    print("--- Prefix search demo (PatriciaTree) ---")
    for prefix in ["alky", "psycho", "lip"]:
        matches = spimi.prefix_search(prefix)
        print(f"  '{prefix}' -> {matches[:8]}")