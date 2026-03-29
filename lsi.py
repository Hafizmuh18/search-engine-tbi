"""
lsi.py
------
Latent Semantic Indexing (LSI) with efficient Truncated SVD.

Theory
------
LSI (also called LSA - Latent Semantic Analysis) is a technique that
represents documents and queries in a low-dimensional "latent semantic space"
by applying Singular Value Decomposition (SVD) to the Term-Document Matrix.

The key idea: words that appear in similar contexts (documents) will be
represented by similar vectors, even if they never co-occur. This allows
LSI to handle synonymy (different words, same meaning) and polysemy
(same word, multiple meanings) better than exact-match keyword search.

Pipeline
--------
1. Build a Term-Document Matrix (TDM) from the index:
      M[i][j] = TF-IDF weight of term i in document j
   Shape: (|V| x |D|), where |V| = vocab size, |D| = num docs.

2. Apply Truncated SVD (keep top-k singular values):
      M ≈ U_k * Σ_k * V_k^T
   where:
      U_k  : (|V| x k)  — term vectors in latent space
      Σ_k  : (k x k)    — diagonal matrix of top-k singular values
      V_k  : (|D| x k)  — document vectors in latent space

3. Project each document to the k-dimensional latent space:
      doc_vector[j] = V_k[j] * Σ_k     (shape: k)

4. At query time, project the query into the same latent space:
      q_vec = (U_k^T * q_tfidf) * Σ_k^{-1} ... simplified to:
      q_vec = U_k^T * q_tfidf            (folding-in)

5. Rank documents by cosine similarity between q_vec and doc_vectors.

Efficient SVD for large Term-Document Matrices
----------------------------------------------
A full TDM for a real collection is very sparse (most entries are 0).
We use scipy.sparse for memory-efficient storage and
sklearn.utils.extmath.randomized_svd (randomized SVD) for fast computation
of the top-k singular values without materializing the full dense matrix.

This approach scales to millions of documents and large vocabularies, which
is much more practical than a full dense SVD (which would require O(|V|*|D|)
memory and O(min(|V|,|D|)^2 * max(|V|,|D|)) time).

Usage
-----
    from lsi import LSIIndex
    from compression import VBEPostings

    lsi = LSIIndex(data_dir='collection', output_dir='index',
                   postings_encoding=VBEPostings, n_components=100)
    lsi.build()   # builds TDM + SVD (can take a few minutes)
    lsi.save()

    results = lsi.retrieve(query="protein synthesis in cell", k=10)
    for score, doc in results:
        print(f"{score:.4f}  {doc}")
"""

import os
import math
import pickle
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from index import InvertedIndexReader
from compression import VBEPostings


class LSIIndex:
    """
    Latent Semantic Indexing over a pre-built inverted index.

    Parameters
    ----------
    data_dir : str
    output_dir : str
        Directory containing the inverted index files.
    postings_encoding : class
        Compression class used when the index was built.
    index_name : str
        Name of the main index file (without extension).
    n_components : int
        Number of latent dimensions k (rank of truncated SVD).
        Typical values: 50-300. Higher = more semantic detail but slower.
    lsi_save_path : str
        Path to save/load the LSI model (pickle file).
    """

    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", n_components=100,
                 lsi_save_path=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.postings_encoding = postings_encoding
        self.index_name = index_name
        self.n_components = n_components
        self.lsi_save_path = lsi_save_path or os.path.join(output_dir, 'lsi_model.pkl')

        self.U_k = None          
        self.S_k = None          
        self.Vt_k = None         
        self.doc_vectors = None  
        self.term_id_map = None
        self.doc_id_map = None
        self.idf = {}            


    def build(self):
        """
        Build the LSI model from the existing inverted index.

        Steps:
        1. Load term/doc ID maps.
        2. Construct sparse TF-IDF Term-Document Matrix.
        3. Apply randomized truncated SVD.
        4. Precompute document vectors in latent space.
        """
        print(f"Building LSI index (k={self.n_components})...")

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

        num_terms = len(self.term_id_map)
        num_docs  = len(self.doc_id_map)
        print(f"  Vocabulary: {num_terms:,} terms, {num_docs:,} documents")

        print("  Building sparse TF-IDF matrix...")
        rows = []
        cols = []
        data = []

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            N = len(idx.doc_length)
            avgdl = sum(idx.doc_length.values()) / N if N > 0 else 1.0

            for term_id, postings, tf_list in idx:
                df = len(postings)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                self.idf[term_id] = idf

                for doc_id, tf in zip(postings, tf_list):
                    if tf > 0:
                        tfidf = idf * (1 + math.log(tf))
                        rows.append(term_id)
                        cols.append(doc_id)
                        data.append(tfidf)

        M = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_terms, num_docs),
            dtype=np.float32
        )
        print(f"  Sparse matrix: {M.shape}, {M.nnz:,} non-zeros "
              f"({100 * M.nnz / (num_terms * num_docs):.4f}% density)")

        k = min(self.n_components, min(num_terms, num_docs) - 1)
        print(f"  Running randomized SVD (k={k})...")
        U, S, Vt = randomized_svd(M, n_components=k, random_state=42, n_iter=4)

        self.U_k  = U.astype(np.float32)  
        self.S_k  = S.astype(np.float32)  
        self.Vt_k = Vt.astype(np.float32) 
        self.doc_vectors = (Vt.T * S).astype(np.float32) 

        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = self.doc_vectors / norms

        print(f"  doc_vectors shape: {self.doc_vectors.shape}")
        print("LSI build complete!")


    def _project_query(self, query_str):
        """
        Project a query string into the k-dimensional latent space.

        Query folding-in formula:
            q_latent = (U_k^T * q_tfidf) normalized to unit length

        where q_tfidf[t] = idf(t) * (1 + log tf(t,q)) for each query term.

        This is an approximation — the query is folded into the existing
        latent space rather than recomputing the SVD.

        Parameters
        ----------
        query_str : str

        Returns
        -------
        np.ndarray of shape (k,) or None if no query terms found in vocab
        """
        num_terms = len(self.term_id_map)
        q_tfidf = np.zeros(num_terms, dtype=np.float32)

        tokens = query_str.split()
        tf_query = {}
        for t in tokens:
            tf_query[t] = tf_query.get(t, 0) + 1

        for token, tf in tf_query.items():
            term_id_val = None
            if hasattr(self.term_id_map, '_lookup'):
                term_id_val = self.term_id_map._lookup(token)
            elif hasattr(self.term_id_map, 'str_to_id'):
                term_id_val = self.term_id_map.str_to_id.get(token)
            else:
                try:
                    term_id_val = self.term_id_map[token]
                except Exception:
                    pass

            if term_id_val is not None and term_id_val in self.idf:
                q_tfidf[term_id_val] = self.idf[term_id_val] * (1 + math.log(tf))

        if q_tfidf.sum() == 0:
            return None

        q_latent = self.U_k.T @ q_tfidf 

        # L2-normalize
        norm = np.linalg.norm(q_latent)
        if norm == 0:
            return None
        return q_latent / norm

    def retrieve(self, query, k=10):
        """
        Retrieve top-K documents using LSI (cosine similarity in latent space).

        Parameters
        ----------
        query : str
            Space-separated query tokens.
        k : int

        Returns
        -------
        List[(float, str)]
            List of (cosine_similarity_score, doc_path), sorted descending.
        """
        if self.doc_vectors is None:
            self.load()

        q_latent = self._project_query(query)
        if q_latent is None:
            return []

        similarities = self.doc_vectors @ q_latent  

        if k >= len(similarities):
            top_k_indices = np.argsort(similarities)[::-1]
        else:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        results = []
        for doc_id in top_k_indices:
            score = float(similarities[doc_id])
            if score > 0:
                results.append((score, self.doc_id_map[int(doc_id)]))

        return results[:k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save the LSI model to disk."""
        model = {
            'U_k': self.U_k,
            'S_k': self.S_k,
            'Vt_k': self.Vt_k,
            'doc_vectors': self.doc_vectors,
            'idf': self.idf,
            'n_components': self.n_components,
        }
        with open(self.lsi_save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"LSI model saved to {self.lsi_save_path}")

    def load(self):
        """Load the LSI model and ID maps from disk."""
        with open(self.lsi_save_path, 'rb') as f:
            model = pickle.load(f)
        self.U_k         = model['U_k']
        self.S_k         = model['S_k']
        self.Vt_k        = model['Vt_k']
        self.doc_vectors = model['doc_vectors']
        self.idf         = model['idf']
        self.n_components = model['n_components']

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def explained_variance_ratio(self):
        """
        Return the proportion of variance explained by each latent dimension.
        Useful for choosing n_components.
        """
        if self.S_k is None:
            return None
        total_var = (self.S_k ** 2).sum()
        return (self.S_k ** 2) / total_var

    def most_related_terms(self, term_str, top_n=10):
        """
        Find terms most semantically related to a given term in latent space.
        Demonstrates LSI's ability to capture synonymy.

        Parameters
        ----------
        term_str : str
        top_n : int

        Returns
        -------
        List[(float, str)]
        """
        if self.U_k is None:
            self.load()

        term_id_val = None
        if hasattr(self.term_id_map, '_lookup'):
            term_id_val = self.term_id_map._lookup(term_str)
        elif hasattr(self.term_id_map, 'str_to_id'):
            term_id_val = self.term_id_map.str_to_id.get(term_str)

        if term_id_val is None:
            return []

        t_vec = self.U_k[term_id_val] * self.S_k
        t_norm = np.linalg.norm(t_vec)
        if t_norm == 0:
            return []
        t_vec = t_vec / t_norm


        all_vecs = self.U_k * self.S_k[np.newaxis, :]  
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_vecs_normed = all_vecs / norms

        sims = all_vecs_normed @ t_vec   
        top_ids = np.argsort(sims)[::-1][1:top_n + 1] 

        return [(float(sims[i]), self.term_id_map[int(i)]) for i in top_ids]


if __name__ == "__main__":
    lsi = LSIIndex(
        data_dir='collection',
        output_dir='index',
        postings_encoding=VBEPostings,
        n_components=100
    )

    lsi_path = os.path.join('index', 'lsi_model.pkl')
    if os.path.exists(lsi_path):
        print("Loading existing LSI model...")
        lsi.load()
    else:
        lsi.build()
        lsi.save()

    evr = lsi.explained_variance_ratio()
    cumvar = np.cumsum(evr)
    print(f"\nTop-10 singular values explain {100*cumvar[9]:.1f}% of variance")
    print(f"All {lsi.n_components} components explain {100*cumvar[-1]:.1f}% of variance")

    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        for score, doc in lsi.retrieve(query, k=5):
            print(f"  {score:.4f}  {doc}")

    print("\n--- Semantic Relations (LSI) ---")
    for term in ["protein", "cell", "blood"]:
        related = lsi.most_related_terms(term, top_n=5)
        terms_str = ", ".join(f"{t}({s:.2f})" for s, t in related)
        print(f"  '{term}' related to: {terms_str}")