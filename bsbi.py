import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, EliasGammaPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Parameters
        ----------
        indices: List[InvertedIndexReader]
        merged_index: InvertedIndexWriter
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time) dan TF-IDF scoring.
        Mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))   jika tf(t, D) > 0
                = 0                     jika sebaliknya
        w(t, Q) = IDF = log(N / df(t))
        Score   = sum over query terms of w(t, Q) * w(t, D)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah top dokumen yang dikembalikan

        Returns
        -------
        List[(float, str)]
            List of (score, doc_path) terurut menurun berdasarkan skor
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time) dan BM25 scoring.
        Mengembalikan top-K retrieval results.

        Formula BM25:
            BM25(t, D) = IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

        dimana:
            IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   [Robertson IDF, selalu positif]
            N      = jumlah total dokumen di koleksi
            df(t)  = document frequency dari term t
            tf(t,D)= term frequency dari t di dokumen D
            |D|    = panjang dokumen D (jumlah token)
            avgdl  = rata-rata panjang dokumen di koleksi
            k1     = parameter saturasi TF (default 1.2)
            b      = parameter normalisasi panjang dokumen (default 0.75)

        Pre-komputasi yang dibutuhkan (sudah tersimpan di index):
            - doc_length: panjang setiap dokumen
            - avgdl dihitung dari doc_length saat retrieval

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah top dokumen yang dikembalikan
        k1: float
            Parameter saturasi TF BM25 (biasanya 1.2 - 2.0)
        b: float
            Parameter normalisasi panjang dokumen (0 = tanpa normalisasi, 1 = normalisasi penuh)

        Returns
        -------
        List[(float, str)]
            List of (score, doc_path) terurut menurun berdasarkan skor
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            # Pre-komputasi average document length
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    # Robertson IDF: selalu positif karena + 1
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]
                        dl = merged_index.doc_length.get(doc_id, 0)
                        # Hitung skor BM25 untuk term ini di dokumen ini
                        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        if doc_id not in scores:
                            scores[doc_id] = 0.0
                        scores[doc_id] += idf * tf_norm

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan algoritma WAND (Weak AND) Top-K.

        WAND adalah algoritma yang memungkinkan kita untuk SKIP dokumen yang
        tidak mungkin masuk ke top-K TANPA menghitung skor BM25 penuhnya.
        Ini menghemat komputasi secara signifikan untuk koleksi besar.

        Cara kerja WAND:
        ----------------
        1. Untuk setiap term query, hitung upper_bound BM25 score-nya:
           UB(t) = IDF(t) * (max_tf(t) * (k1 + 1)) / (max_tf(t) + k1 * (1 - b + b * min_dl / avgdl))
           Kita gunakan pendekatan sederhana: UB(t) = IDF(t) * (k1 + 1)
           (ketika tf sangat besar, saturation factor -> (k1+1), dan dl/avgdl ~ 1 - b + b = 1)

        2. Maintain sebuah min-heap berisi top-K dokumen sejauh ini (threshold θ).

        3. Untuk setiap kandidat dokumen (dari pointer di postings list):
           a. Urutkan term berdasarkan current doc_id di pointer masing-masing.
           b. Cari "pivot" term: term pertama dimana cumulative upper bound >= θ.
           c. Jika doc_id di pivot == doc_id di term[0], FULL EVAL dokumen tsb.
           d. Jika tidak, advance pointer term[0] ke doc_id pivot.
           e. Jika cumulative UB < θ bahkan untuk semua term, STOP.

        Implementasi ini menggunakan pendekatan "sorted postings dengan cursor"
        yang merupakan inti dari WAND.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k: int
            Jumlah top dokumen yang dikembalikan (Top-K)
        k1: float
            Parameter BM25 (default 1.2)
        b: float
            Parameter BM25 (default 0.75)

        Returns
        -------
        List[(float, str)]
            List of (score, doc_path) terurut menurun berdasarkan skor
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_words = query.split()
        # Deduplikasi query terms
        seen = set()
        unique_words = []
        for w in query_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        terms = [self.term_id_map[word] for word in unique_words]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            # Filter term yang ada di index
            valid_terms = [t for t in terms if t in merged_index.postings_dict]
            if not valid_terms:
                return []

            # Load semua postings list ke memori (untuk WAND kita perlu random access)
            postings_data = {}
            for term in valid_terms:
                pl, tfl = merged_index.get_postings_list(term)
                postings_data[term] = (pl, tfl)

            # Hitung IDF dan upper bound untuk setiap term
            idf = {}
            upper_bound = {}
            for term in valid_terms:
                df = merged_index.postings_dict[term][1]
                idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
                # Upper bound BM25 contribution:
                # Ketika tf -> infinity, tf_norm -> (k1 + 1)
                # Ketika dl = avgdl, normalisasi = 1
                # Jadi UB(t) ≈ IDF(t) * (k1 + 1)
                # Kita bisa perbaiki dengan max_tf dari index:
                max_tf = merged_index.term_max_tf.get(term, 1)
                # Hitung UB pakai min dl (dokumen terpendek) untuk upper bound yang ketat
                min_dl = min(merged_index.doc_length.values()) if merged_index.doc_length else avgdl
                tf_norm_ub = (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b + b * min_dl / avgdl))
                upper_bound[term] = idf[term] * tf_norm_ub

            # Inisialisasi cursor (pointer) untuk setiap term
            # cursor[term] = index ke dalam postings list
            cursors = {term: 0 for term in valid_terms}
            postings_lists = {term: postings_data[term][0] for term in valid_terms}
            tf_lists = {term: postings_data[term][1] for term in valid_terms}

            def current_doc(term):
                """Kembalikan doc_id saat ini untuk sebuah term, atau inf jika exhausted."""
                idx = cursors[term]
                pl = postings_lists[term]
                return pl[idx] if idx < len(pl) else float('inf')

            def advance_to(term, target_doc_id):
                """Advance cursor term ke doc_id >= target_doc_id."""
                pl = postings_lists[term]
                idx = cursors[term]
                while idx < len(pl) and pl[idx] < target_doc_id:
                    idx += 1
                cursors[term] = idx

            # Min-heap untuk top-K: berisi (score, doc_id)
            top_k_heap = []  # min-heap
            threshold = 0.0  # θ = skor minimum yang perlu dilewati untuk masuk top-K

            evaluated_docs = set()

            # WAND main loop
            max_iterations = sum(len(postings_lists[t]) for t in valid_terms) * 2
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Urutkan term berdasarkan current doc_id (ascending)
                active_terms = [t for t in valid_terms if cursors[t] < len(postings_lists[t])]
                if not active_terms:
                    break

                active_terms.sort(key=lambda t: current_doc(t))

                # Hitung cumulative upper bound dari kiri
                # Cari pivot: term pertama dimana cumsum UB >= threshold
                pivot_idx = None
                cumsum = 0.0
                for i, term in enumerate(active_terms):
                    cumsum += upper_bound[term]
                    if cumsum >= threshold or len(top_k_heap) < k:
                        pivot_idx = i
                        break

                if pivot_idx is None:
                    # Cumulative UB bahkan untuk semua term < threshold: STOP
                    break

                pivot_term = active_terms[pivot_idx]
                pivot_doc = current_doc(pivot_term)

                if pivot_doc == float('inf'):
                    break

                # Cek apakah semua term sebelum pivot sudah di doc_id yang sama
                if current_doc(active_terms[0]) == pivot_doc:
                    # FULL EVALUATION: hitung BM25 penuh untuk pivot_doc
                    if pivot_doc not in evaluated_docs:
                        evaluated_docs.add(pivot_doc)
                        score = 0.0
                        dl = merged_index.doc_length.get(pivot_doc, avgdl)
                        for term in valid_terms:
                            idx = cursors[term]
                            pl = postings_lists[term]
                            if idx < len(pl) and pl[idx] == pivot_doc:
                                tf = tf_lists[term][idx]
                                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                                score += idf[term] * tf_norm

                        # Update top-K heap
                        if len(top_k_heap) < k:
                            heapq.heappush(top_k_heap, (score, pivot_doc))
                            if len(top_k_heap) == k:
                                threshold = top_k_heap[0][0]
                        elif score > top_k_heap[0][0]:
                            heapq.heapreplace(top_k_heap, (score, pivot_doc))
                            threshold = top_k_heap[0][0]

                    # Advance SEMUA term yang berada di pivot_doc
                    for term in valid_terms:
                        if current_doc(term) == pivot_doc:
                            cursors[term] += 1
                else:
                    # Advance semua term sebelum pivot ke pivot_doc
                    for i in range(pivot_idx):
                        advance_to(active_terms[i], pivot_doc)

            # Konversi heap ke sorted list
            results = []
            while top_k_heap:
                score, doc_id = heapq.heappop(top_k_heap)
                results.append((score, self.doc_id_map[doc_id]))
            results.sort(key=lambda x: x[0], reverse=True)
            return results

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.index()  # memulai indexing!