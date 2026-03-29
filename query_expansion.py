"""
query_expansion.py
------------------
Query Expansion techniques to improve retrieval recall.

Two approaches are implemented:

1. Pseudo-Relevance Feedback (PRF) via Rocchio Algorithm
   -------------------------------------------------------
   Assumes the top-K documents from an initial retrieval are relevant,
   then expands the query with the most discriminative terms from those docs.

   Rocchio formula:
       q_new = α * q_orig + β * (1/|R|) * Σ d_i - γ * (1/|NR|) * Σ d_j
   where R = top-K relevant docs, NR = bottom-K non-relevant docs.

   In practice (blind PRF), we skip the γ term (no negative feedback):
       q_new = α * q_orig + β * (1/|R|) * Σ d_i

   The expansion terms are the top-M terms with highest weight in q_new
   that are NOT in the original query.

2. Co-occurrence based expansion
   --------------------------------
   Finds terms that frequently co-occur with the query terms within the
   same document window. Terms that appear together with query terms
   more than chance would predict are good expansion candidates.

   Implemented via Pointwise Mutual Information (PMI):
       PMI(t1, t2) = log(P(t1,t2) / (P(t1) * P(t2)))
                   = log(df(t1,t2) * N / (df(t1) * df(t2)))

These techniques are used in PyTerrier's query rewriting pipeline and
similar frameworks.

Usage
-----
    from query_expansion import PseudoRelevanceFeedback, CooccurrenceExpander

    # Rocchio PRF
    prf = PseudoRelevanceFeedback(bsbi_instance, top_k=10, n_expand=5)
    expanded_query = prf.expand("protein synthesis")
    # e.g., "protein synthesis amino acid ribosome mrna"

    # Retrieve with expanded query
    results = bsbi_instance.retrieve_bm25(expanded_query, k=10)
"""

import math
import heapq
from index import InvertedIndexReader


class PseudoRelevanceFeedback:
    """
    Pseudo-Relevance Feedback (PRF) using the Rocchio Algorithm.

    Assumes the top-K retrieved documents are relevant, and expands
    the query with the most informative terms from those documents.

    Parameters
    ----------
    index_instance : BSBIIndex or SPIMIIndex
        A fully initialized index instance (already indexed).
    top_k : int
        Number of top documents to use as pseudo-relevant feedback.
    n_expand : int
        Number of new terms to add to the query.
    alpha : float
        Weight for original query (Rocchio alpha). Default 1.0.
    beta : float
        Weight for pseudo-relevant documents (Rocchio beta). Default 0.8.
    """

    def __init__(self, index_instance, top_k=10, n_expand=5, alpha=1.0, beta=0.8):
        self.index = index_instance
        self.top_k = top_k
        self.n_expand = n_expand
        self.alpha = alpha
        self.beta = beta

    def expand(self, query, retrieval_method='bm25'):
        """
        Expand a query using PRF.

        Parameters
        ----------
        query : str
            Original query string.
        retrieval_method : str
            'bm25' or 'tfidf'

        Returns
        -------
        str
            Expanded query string (original + expansion terms).
        List[str]
            List of expansion terms only (for debugging/display).
        """
        if retrieval_method == 'bm25':
            initial_results = self.index.retrieve_bm25(query, k=self.top_k)
        else:
            initial_results = self.index.retrieve_tfidf(query, k=self.top_k)

        if not initial_results:
            return query, []

        path_to_id = {}
        for doc_id, path in enumerate(self.index.doc_id_map.id_to_str):
            path_to_id[path] = doc_id

        relevant_doc_ids = set()
        for score, doc_path in initial_results:
            if doc_path in path_to_id:
                relevant_doc_ids.add(path_to_id[doc_path])

        original_terms = set(query.split())
        original_term_ids = set()
        for t in original_terms:
            if hasattr(self.index.term_id_map, '_lookup'):
                tid = self.index.term_id_map._lookup(t)
            elif hasattr(self.index.term_id_map, 'str_to_id'):
                tid = self.index.term_id_map.str_to_id.get(t)
            else:
                tid = None
            if tid is not None:
                original_term_ids.add(tid)

        term_feedback_score = {}
        num_relevant = len(relevant_doc_ids)

        with InvertedIndexReader(self.index.index_name, self.index.postings_encoding,
                                 directory=self.index.output_dir) as idx:
            N = len(idx.doc_length)
            for term_id, postings, tf_list in idx:
                if term_id in original_term_ids:
                    continue  

                df = len(postings)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                feedback_sum = 0.0
                for doc_id, tf in zip(postings, tf_list):
                    if doc_id in relevant_doc_ids and tf > 0:
                        feedback_sum += idf * (1 + math.log(tf))

                if feedback_sum > 0:
                    term_feedback_score[term_id] = self.beta * feedback_sum / num_relevant

        if not term_feedback_score:
            return query, []

        top_expansion_ids = heapq.nlargest(
            self.n_expand,
            term_feedback_score.keys(),
            key=lambda tid: term_feedback_score[tid]
        )

        expansion_terms = []
        for tid in top_expansion_ids:
            term_str = self.index.term_id_map[tid]
            expansion_terms.append(term_str)

        expanded_query = query + " " + " ".join(expansion_terms)
        return expanded_query, expansion_terms


class CooccurrenceExpander:
    """
    Query expansion using term co-occurrence statistics (PMI-based).

    Finds terms that frequently appear in the same documents as the
    query terms. Uses Pointwise Mutual Information (PMI) to measure
    the strength of co-occurrence beyond chance.

    PMI(t_query, t_candidate) = log( df(t_q AND t_c) * N / (df(t_q) * df(t_c)) )

    Parameters
    ----------
    index_instance : BSBIIndex or SPIMIIndex
    n_expand : int
        Number of expansion terms to return per query.
    min_df : int
        Minimum document frequency for a candidate term to be considered.
    """

    def __init__(self, index_instance, n_expand=5, min_df=2):
        self.index = index_instance
        self.n_expand = n_expand
        self.min_df = min_df
        self._doc_sets = {}    
        self._dfs = {}         
        self._N = 0

    def _build_cache(self):
        """Build in-memory cache of doc-sets for all terms."""
        if self._doc_sets:
            return
        with InvertedIndexReader(self.index.index_name, self.index.postings_encoding,
                                 directory=self.index.output_dir) as idx:
            self._N = len(idx.doc_length)
            for term_id, postings, tf_list in idx:
                self._doc_sets[term_id] = set(postings)
                self._dfs[term_id] = len(postings)

    def expand(self, query):
        """
        Expand a query using co-occurrence PMI.

        Parameters
        ----------
        query : str

        Returns
        -------
        str
            Expanded query.
        List[str]
            Expansion terms only.
        """
        self._build_cache()

        query_term_ids = []
        original_terms = set(query.split())
        for t in original_terms:
            if hasattr(self.index.term_id_map, '_lookup'):
                tid = self.index.term_id_map._lookup(t)
            elif hasattr(self.index.term_id_map, 'str_to_id'):
                tid = self.index.term_id_map.str_to_id.get(t)
            else:
                tid = None
            if tid is not None and tid in self._doc_sets:
                query_term_ids.append(tid)

        if not query_term_ids:
            return query, []

        query_doc_set = set()
        for tid in query_term_ids:
            query_doc_set |= self._doc_sets[tid]

        df_query_union = len(query_doc_set)
        if df_query_union == 0:
            return query, []

        pmi_scores = {}
        for cand_tid, cand_docs in self._doc_sets.items():
            if cand_tid in set(query_term_ids):
                continue  
            if self._dfs.get(cand_tid, 0) < self.min_df:
                continue  

            cooc = len(query_doc_set & cand_docs)
            if cooc == 0:
                continue

            df_cand = self._dfs[cand_tid]
            N = self._N

            pmi = math.log(cooc * N / (df_query_union * df_cand + 1e-9) + 1)
            pmi_scores[cand_tid] = pmi

        if not pmi_scores:
            return query, []

        top_ids = heapq.nlargest(self.n_expand, pmi_scores.keys(),
                                 key=lambda t: pmi_scores[t])

        expansion_terms = [self.index.term_id_map[tid] for tid in top_ids]
        expanded_query = query + " " + " ".join(expansion_terms)
        return expanded_query, expansion_terms


class QueryExpansionPipeline:
    """
    A complete query expansion pipeline that chains multiple expansion
    strategies and retrieval methods.

    This is inspired by PyTerrier's pipeline design where transformers
    are composed with the >> operator.

    Example
    -------
        pipeline = QueryExpansionPipeline(bsbi_instance)
        results = pipeline.run("protein synthesis", method='prf+bm25', k=10)
        results = pipeline.run("protein synthesis", method='cooc+bm25', k=10)
        results = pipeline.run("protein synthesis", method='prf+lsi', k=10, lsi=lsi_instance)
    """

    def __init__(self, index_instance, top_k_feedback=10, n_expand=5):
        self.index = index_instance
        self.prf = PseudoRelevanceFeedback(index_instance,
                                           top_k=top_k_feedback,
                                           n_expand=n_expand)
        self.cooc = CooccurrenceExpander(index_instance, n_expand=n_expand)

    def run(self, query, method='prf+bm25', k=10, lsi=None):
        """
        Run retrieval with optional query expansion.

        Parameters
        ----------
        query : str
        method : str
            One of: 'bm25', 'tfidf', 'prf+bm25', 'prf+tfidf',
                    'cooc+bm25', 'prf+lsi', 'lsi'
        k : int
        lsi : LSIIndex or None

        Returns
        -------
        List[(float, str)]
        dict with keys 'results', 'expanded_query', 'expansion_terms'
        """
        expanded_query = query
        expansion_terms = []

        if method.startswith('prf'):
            base_method = 'bm25' if 'bm25' in method else 'tfidf'
            expanded_query, expansion_terms = self.prf.expand(query, base_method)
        elif method.startswith('cooc'):
            expanded_query, expansion_terms = self.cooc.expand(query)

        if method.endswith('lsi') and lsi is not None:
            results = lsi.retrieve(expanded_query, k=k)
        elif 'tfidf' in method:
            results = self.index.retrieve_tfidf(expanded_query, k=k)
        else:
            results = self.index.retrieve_bm25(expanded_query, k=k)

        return {
            'results': results,
            'expanded_query': expanded_query,
            'expansion_terms': expansion_terms,
        }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from bsbi import BSBIIndex
    from compression import VBEPostings

    print("Loading index...")
    engine = BSBIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index')
    engine.load()

    pipeline = QueryExpansionPipeline(engine, top_k_feedback=10, n_expand=5)

    queries = [
        "alkylated with radioactive iodoacetate",
        "lipid metabolism toxemia pregnancy",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Original query: '{query}'")

        out = pipeline.run(query, method='prf+bm25', k=5)
        print(f"\n[PRF + BM25]")
        print(f"  Expanded: '{out['expanded_query']}'")
        print(f"  New terms: {out['expansion_terms']}")
        for score, doc in out['results']:
            print(f"    {score:.3f}  {doc}")

        out2 = pipeline.run(query, method='cooc+bm25', k=5)
        print(f"\n[Co-occurrence + BM25]")
        print(f"  Expanded: '{out2['expanded_query']}'")
        print(f"  New terms: {out2['expansion_terms']}")
        for score, doc in out2['results']:
            print(f"    {score:.3f}  {doc}")