"""
ranked_fusion.py
----------------
Ensemble Retrieval via Rank Aggregation.

Motivation
----------
No single retrieval method is universally best:
- BM25 is strong on keyword matching but misses synonyms
- LSI captures semantic similarity but loses exact-match precision
- WAND is fast but may miss some documents
- PRF improves recall but can drift from the original topic

Rank fusion combines ranked lists from multiple retrievers into a single
ranked list that is more robust than any individual method.

Three fusion strategies are implemented:

1. Reciprocal Rank Fusion (RRF)  [Cormack et al., 2009]
   -------------------------------------------------------
   The simplest and most robust fusion method. Each document's score is:
       RRF(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))
   where k=60 is a smoothing constant and rank_r(d) is d's rank in retriever r.
   Key insight: RRF is insensitive to score magnitude differences between
   systems — only ranks matter. Works well even when combining very
   different scoring regimes (BM25 scores in 0-20 range, cosine in -1 to 1).

2. Condorcet Fusion
   ------------------
   A voting-based method from social choice theory. Document A beats B
   if A is ranked higher than B in more retrievers than B beats A.
   Final ranking = total wins across all pairwise comparisons.
   More robust to outlier retrievers than score-based fusion.

3. CombSUM / CombMNZ (normalized score fusion)
   -----------------------------------------------
   Normalize each retriever's scores to [0, 1], then:
   CombSUM(d) = Σ scores_normalized(d)
   CombMNZ(d) = CombSUM(d) × (number of retrievers that returned d)
   CombMNZ rewards documents found by multiple retrievers.

Usage
-----
    from ranked_fusion import RankFusion

    fusion = RankFusion(engine, lsi=lsi_instance)

    # Fuse BM25 + LSI + PRF with RRF
    results = fusion.retrieve("protein synthesis", k=10,
                              methods=['bm25', 'lsi', 'prf+bm25'],
                              strategy='rrf')

    # Compare all fusion strategies at once
    fusion.compare("protein synthesis", k=10)
"""

import math
from collections import defaultdict


class RankFusion:
    """
    Combines multiple retrieval methods into a single ranked list.

    Parameters
    ----------
    index_instance : BSBIIndex or SPIMIIndex
        A fully loaded index instance.
    lsi : LSIIndex or None
        Optional LSI model for semantic retrieval.
    rrf_k : int
        Smoothing constant for RRF (default 60, as in the original paper).
    """

    AVAILABLE_METHODS = ['tfidf', 'bm25', 'wand', 'lsi', 'prf+bm25', 'cooc+bm25']

    def __init__(self, index_instance, lsi=None, rrf_k=60):
        self.index = index_instance
        self.lsi = lsi
        self.rrf_k = rrf_k
        self._prf_pipeline = None

    def _get_pipeline(self):
        if self._prf_pipeline is None:
            from query_expansion import QueryExpansionPipeline
            self._prf_pipeline = QueryExpansionPipeline(
                self.index, top_k_feedback=10, n_expand=5
            )
        return self._prf_pipeline

    def _run_single(self, query: str, method: str, k: int) -> list:
        """
        Run a single retrieval method and return List[(score, doc_path)].

        Parameters
        ----------
        query : str
        method : str
        k : int

        Returns
        -------
        List[(float, str)]
        """
        if method == 'tfidf':
            return self.index.retrieve_tfidf(query, k=k)
        elif method == 'bm25':
            return self.index.retrieve_bm25(query, k=k)
        elif method == 'wand':
            return self.index.retrieve_bm25_wand(query, k=k)
        elif method == 'lsi':
            if self.lsi is None:
                return []
            return self.lsi.retrieve(query, k=k)
        elif method in ('prf+bm25', 'cooc+bm25', 'prf+lsi'):
            pipeline = self._get_pipeline()
            out = pipeline.run(query, method=method, k=k, lsi=self.lsi)
            return out['results']
        else:
            raise ValueError(f"Unknown method: {method}")

    # ─────────────────────────────────────────────
    # Fusion Strategy 1: Reciprocal Rank Fusion
    # ─────────────────────────────────────────────

    def rrf_fusion(self, ranked_lists: list, k: int) -> list:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        RRF_score(d) = Σ_r 1 / (rrf_k + rank_r(d))

        Only documents that appear in at least one list are considered.
        Documents not appearing in a list get rank = len(list) + 1 (soft penalty).

        Parameters
        ----------
        ranked_lists : List[List[(float, str)]]
            Each inner list is a ranked result list from one retriever.
        k : int
            Number of final results to return.

        Returns
        -------
        List[(float, str)]
            Sorted by RRF score descending.
        """
        rrf_scores = defaultdict(float)

        for ranked_list in ranked_lists:
            rank_of = {doc: rank + 1 for rank, (score, doc) in enumerate(ranked_list)}
            max_rank = len(ranked_list) + 1

            all_docs = set(doc for _, doc in ranked_list)
            for doc in all_docs:
                r = rank_of.get(doc, max_rank)
                rrf_scores[doc] += 1.0 / (self.rrf_k + r)

        results = [(score, doc) for doc, score in rrf_scores.items()]
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    # ─────────────────────────────────────────────
    # Fusion Strategy 2: Condorcet Fusion
    # ─────────────────────────────────────────────

    def condorcet_fusion(self, ranked_lists: list, k: int) -> list:
        """
        Condorcet Fusion (voting-based rank aggregation).

        For each pair of documents (A, B), we count across all retrievers
        how many times A is ranked higher than B. Document A's final score
        is the total number of pairwise wins against all other documents.

        This is O(D^2 * R) where D = unique docs, R = number of retrievers.
        For large result sets, approximate with a subset.

        Parameters
        ----------
        ranked_lists : List[List[(float, str)]]
        k : int

        Returns
        -------
        List[(float, str)]
        """
        # Collect all unique documents
        all_docs = list({doc for ranked_list in ranked_lists for _, doc in ranked_list})

        if len(all_docs) > 500:
            # For efficiency, only compare top-200 from each list
            candidate_set = set()
            for ranked_list in ranked_lists:
                for _, doc in ranked_list[:200]:
                    candidate_set.add(doc)
            all_docs = list(candidate_set)

        # Build rank lookup per retriever
        rank_lookups = []
        for ranked_list in ranked_lists:
            rank_of = {doc: rank + 1 for rank, (score, doc) in enumerate(ranked_list)}
            rank_lookups.append(rank_of)

        # Count pairwise wins
        wins = defaultdict(int)
        for i, doc_a in enumerate(all_docs):
            for doc_b in all_docs[i + 1:]:
                a_wins = 0
                b_wins = 0
                for rank_of in rank_lookups:
                    ra = rank_of.get(doc_a, len(rank_of) + 1)
                    rb = rank_of.get(doc_b, len(rank_of) + 1)
                    if ra < rb:
                        a_wins += 1
                    elif rb < ra:
                        b_wins += 1
                wins[doc_a] += a_wins
                wins[doc_b] += b_wins

        results = [(wins[doc], doc) for doc in all_docs if wins[doc] > 0]
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    # ─────────────────────────────────────────────
    # Fusion Strategy 3: CombSUM / CombMNZ
    # ─────────────────────────────────────────────

    def combmnz_fusion(self, ranked_lists: list, k: int,
                       use_mnz: bool = True) -> list:
        """
        CombSUM / CombMNZ score fusion with min-max normalization.

        1. Normalize each retriever's scores to [0, 1]:
              score_norm = (s - min) / (max - min)
        2. CombSUM(d) = Σ score_norm(d)
        3. CombMNZ(d) = CombSUM(d) * num_retrievers_returning_d
           (MNZ = "multiply by non-zero count")

        Parameters
        ----------
        ranked_lists : List[List[(float, str)]]
        k : int
        use_mnz : bool
            If True, use CombMNZ; if False, use CombSUM.

        Returns
        -------
        List[(float, str)]
        """
        combined_scores = defaultdict(float)
        retriever_counts = defaultdict(int)

        for ranked_list in ranked_lists:
            if not ranked_list:
                continue

            scores = [s for s, _ in ranked_list]
            min_s = min(scores)
            max_s = max(scores)
            denom = max_s - min_s if max_s != min_s else 1.0

            for score, doc in ranked_list:
                norm_score = (score - min_s) / denom
                combined_scores[doc] += norm_score
                retriever_counts[doc] += 1

        if use_mnz:
            final_scores = {
                doc: score * retriever_counts[doc]
                for doc, score in combined_scores.items()
            }
        else:
            final_scores = dict(combined_scores)

        results = [(score, doc) for doc, score in final_scores.items()]
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    # ─────────────────────────────────────────────
    # Unified retrieve interface
    # ─────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10,
                 methods: list = None,
                 strategy: str = 'rrf') -> list:
        """
        Run multiple retrievers and fuse their results.

        Parameters
        ----------
        query : str
        k : int
        methods : List[str] or None
            Which methods to fuse. Defaults to ['bm25', 'lsi'] if LSI available,
            else ['tfidf', 'bm25', 'wand'].
        strategy : str
            'rrf'      → Reciprocal Rank Fusion (recommended)
            'condorcet' → Condorcet voting fusion
            'combsum'  → CombSUM normalized score fusion
            'combmnz'  → CombMNZ normalized score fusion (default over combsum)

        Returns
        -------
        List[(float, str)]
        """
        if methods is None:
            if self.lsi is not None:
                methods = ['bm25', 'lsi', 'prf+bm25']
            else:
                methods = ['tfidf', 'bm25', 'wand']

        ranked_lists = []
        for method in methods:
            try:
                results = self._run_single(query, method, k=max(k * 3, 100))
                ranked_lists.append(results)
            except Exception as e:
                print(f"  Warning: {method} failed: {e}")
                ranked_lists.append([])

        if strategy == 'rrf':
            return self.rrf_fusion(ranked_lists, k=k)
        elif strategy == 'condorcet':
            return self.condorcet_fusion(ranked_lists, k=k)
        elif strategy == 'combsum':
            return self.combmnz_fusion(ranked_lists, k=k, use_mnz=False)
        elif strategy == 'combmnz':
            return self.combmnz_fusion(ranked_lists, k=k, use_mnz=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Choose from: rrf, condorcet, combsum, combmnz")

    # ─────────────────────────────────────────────
    # Analysis tools
    # ─────────────────────────────────────────────

    def compare(self, query: str, k: int = 10,
                methods: list = None) -> dict:
        """
        Run all fusion strategies and individual methods, show comparison.

        Returns
        -------
        dict : {method_name: List[(score, doc)]}
        """
        if methods is None:
            methods = ['bm25', 'wand']
            if self.lsi is not None:
                methods.append('lsi')

        print(f"\nQuery: '{query}'")
        print(f"Individual retrievers + all fusion strategies (k={k})\n")

        all_results = {}

        for method in methods:
            res = self._run_single(query, method, k=k)
            all_results[method] = res

        for strategy in ['rrf', 'condorcet', 'combmnz']:
            label = f'fusion:{strategy}'
            res = self.retrieve(query, k=k, methods=methods, strategy=strategy)
            all_results[label] = res

        for name, results in all_results.items():
            print(f"  [{name}]")
            for rank, (score, doc) in enumerate(results[:5], 1):
                import re
                m = re.search(r'/([^/]+)\.txt$', doc)
                short = m.group(1) if m else doc
                print(f"    {rank}. {score:8.4f}  {short}")
            print()

        return all_results

    def rank_overlap_analysis(self, query: str, k: int = 10,
                              methods: list = None) -> dict:
        """
        Analyze how much the top-K results overlap between methods.
        High overlap = methods agree; low overlap = diverse perspectives.

        Returns
        -------
        dict : {(method_a, method_b): jaccard_similarity}
        """
        if methods is None:
            methods = ['bm25', 'wand']
            if self.lsi is not None:
                methods.append('lsi')

        top_k_sets = {}
        for method in methods:
            res = self._run_single(query, method, k=k)
            top_k_sets[method] = {doc for _, doc in res}

        overlaps = {}
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                s1 = top_k_sets[m1]
                s2 = top_k_sets[m2]
                if not s1 and not s2:
                    j = 1.0
                elif not s1 or not s2:
                    j = 0.0
                else:
                    j = len(s1 & s2) / len(s1 | s2)
                overlaps[(m1, m2)] = j

        print(f"\nTop-{k} Jaccard overlap between methods (query: '{query[:40]}'):")
        for (m1, m2), j in sorted(overlaps.items(), key=lambda x: -x[1]):
            bar = '█' * int(j * 20)
            print(f"  {m1:<15} vs {m2:<15} : {j:.3f} {bar}")

        return overlaps

def eval_fusion(fusion_instance, qrels, query_file='queries.txt', k=1000):
    """
    Evaluate all fusion strategies as part of the main evaluation suite.

    Parameters
    ----------
    fusion_instance : RankFusion
    qrels : dict
    query_file : str
    k : int

    Returns
    -------
    dict : {strategy: {metric: score}}
    """
    import re
    from evaluation import dcg, ndcg, average_precision, rbp

    strategies = {
        'RRF(bm25+wand)':     lambda q, k: fusion_instance.retrieve(
            q, k=k, methods=['bm25', 'wand'], strategy='rrf'),
        'CombMNZ(bm25+wand)': lambda q, k: fusion_instance.retrieve(
            q, k=k, methods=['bm25', 'wand'], strategy='combmnz'),
        'Condorcet(bm25+wand)': lambda q, k: fusion_instance.retrieve(
            q, k=k, methods=['bm25', 'wand'], strategy='condorcet'),
    }

    if fusion_instance.lsi is not None:
        strategies['RRF(bm25+lsi+prf)'] = lambda q, k: fusion_instance.retrieve(
            q, k=k, methods=['bm25', 'lsi', 'prf+bm25'], strategy='rrf')

    from evaluation import eval_retrieval
    all_scores = {}
    for name, fn in strategies.items():
        scores = eval_retrieval(fn, name, qrels, query_file=query_file, k=k)
        all_scores[name] = scores

    return all_scores


if __name__ == '__main__':
    from bsbi import BSBIIndex
    from compression import VBEPostings

    print("Loading index...")
    engine = BSBIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index')
    engine.load()

    fusion = RankFusion(engine)

    queries = [
        "alkylated with radioactive iodoacetate",
        "lipid metabolism toxemia pregnancy",
    ]

    for query in queries:
        fusion.compare(query, k=5, methods=['tfidf', 'bm25', 'wand'])
        fusion.rank_overlap_analysis(query, k=10, methods=['tfidf', 'bm25', 'wand'])