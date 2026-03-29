"""
index_inspector.py
------------------
Deep analytics and health inspection of the inverted index.

This module provides statistical insights into the index structure:

1. Vocabulary Statistics
   - Total terms, document frequencies, term lengths
   - Hapax legomena ratio (terms appearing in exactly 1 document)
   - High-frequency vs rare term distribution

2. Zipf's Law Analysis
   - Verify that the term frequency distribution follows Zipf's law
   - Compute best-fit Zipfian exponent α via linear regression on log-log plot
   - A well-formed English corpus should have α ≈ 1.0 (Zipf's law)

3. Document Length Distribution
   - Mean, median, std dev, min, max document lengths
   - Percentile breakdown
   - Identifies unusually short/long documents (potential data quality issues)

4. Index Compression Report
   - Compare theoretical uncompressed size vs actual on-disk size
   - Per-codec compression estimate

5. Term Co-occurrence Density
   - Estimate average posting list length
   - Identify terms with pathologically long postings (stop word candidates)

6. Query Difficulty Predictor
   - Given a query, estimate retrieval difficulty:
     * High IDF variance → easy (discriminative terms)
     * Low IDF variance → hard (all terms are common/rare)
     * Very short postings → possibly no results

Usage
-----
    python index_inspector.py
    python index_inspector.py --output-dir index --top-n 20
    python index_inspector.py --query "protein synthesis"
"""

import os
import math
import pickle
import struct
from collections import Counter
import numpy as np

from index import InvertedIndexReader
from compression import VBEPostings


class IndexInspector:
    """
    Comprehensive analytics engine for an inverted index.

    Parameters
    ----------
    output_dir : str
        Directory containing the built index files.
    postings_encoding : class
        The compression codec used when building the index.
    index_name : str
    """

    def __init__(self, output_dir='index', postings_encoding=VBEPostings,
                 index_name='main_index'):
        self.output_dir = output_dir
        self.postings_encoding = postings_encoding
        self.index_name = index_name

        self._term_id_map = None
        self._doc_id_map = None
        self._stats = None   

    def _load_maps(self):
        if self._term_id_map is not None:
            return
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self._term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self._doc_id_map = pickle.load(f)

    def _collect_stats(self):
        """Single pass over the full index to collect all statistics."""
        if self._stats is not None:
            return self._stats

        self._load_maps()

        dfs = {}          
        max_tfs = {}      
        sum_tfs = {}       
        term_lengths = []  

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            doc_lengths = dict(idx.doc_length)
            N = len(doc_lengths)

            for term_id, postings, tf_list in idx:
                df = len(postings)
                dfs[term_id] = df
                max_tfs[term_id] = max(tf_list)
                sum_tfs[term_id] = sum(tf_list)

                try:
                    term_str = self._term_id_map[term_id]
                    term_lengths.append(len(term_str))
                except Exception:
                    term_lengths.append(0)

        self._stats = {
            'N': N,
            'V': len(dfs),
            'dfs': dfs,
            'max_tfs': max_tfs,
            'sum_tfs': sum_tfs,
            'term_lengths': term_lengths,
            'doc_lengths': doc_lengths,
        }
        return self._stats

    def vocabulary_report(self, top_n: int = 20):
        """Print vocabulary statistics and top/bottom terms by DF."""
        st = self._collect_stats()
        N, V = st['N'], st['V']
        dfs = st['dfs']
        sum_tfs = st['sum_tfs']
        term_lengths = st['term_lengths']

        hapax = sum(1 for df in dfs.values() if df == 1)

        df_vals = sorted(dfs.values(), reverse=True)
        p50 = df_vals[len(df_vals) // 2]
        p90 = df_vals[len(df_vals) // 10]
        p99 = df_vals[len(df_vals) // 100] if len(df_vals) >= 100 else df_vals[0]

        avg_term_len = sum(term_lengths) / len(term_lengths) if term_lengths else 0

        print("\n" + "=" * 60)
        print("  Vocabulary Report")
        print("=" * 60)
        print(f"  Documents (N)          : {N:>10,}")
        print(f"  Vocabulary (V)         : {V:>10,}")
        print(f"  Hapax legomena         : {hapax:>10,}  ({100*hapax/V:.1f}% of vocab)")
        print(f"  Avg term length        : {avg_term_len:>10.2f} chars")
        print(f"  DF percentiles  p50={p50}  p90={p90}  p99={p99}")
        print(f"  Total token occurrences: {sum(sum_tfs.values()):>10,}")

        print(f"\n  Top-{top_n} terms by document frequency:")
        top_terms = sorted(dfs.items(), key=lambda x: -x[1])[:top_n]
        for tid, df in top_terms:
            try:
                term = self._term_id_map[tid]
            except Exception:
                term = f"<id:{tid}>"
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            print(f"    {term:<20}  df={df:>6}  idf={idf:.3f}")

        rare_2 = sum(1 for df in dfs.values() if df <= 2)
        print(f"\n  Terms with df <= 2: {rare_2:,}  ({100*rare_2/V:.1f}%)")
        print(f"  → Removing these would reduce vocab by {100*rare_2/V:.1f}%")
        print("=" * 60)


    def zipf_analysis(self):
        """
        Verify whether the collection follows Zipf's law.

        Zipf's law states: frequency(rank r) ∝ 1 / r^α
        Taking log: log(freq) = -α * log(rank) + C

        For English text, α ≈ 1.0.
        We fit this via least-squares regression on the log-log plot.
        """
        st = self._collect_stats()
        sum_tfs = st['sum_tfs']

        freqs = sorted(sum_tfs.values(), reverse=True)
        ranks = list(range(1, len(freqs) + 1))

        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)

        A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
        alpha_neg, C = np.linalg.lstsq(A, log_freqs, rcond=None)[0]
        alpha = -alpha_neg

        predicted = -alpha * log_ranks + C
        ss_res = np.sum((log_freqs - predicted) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print("\n" + "=" * 60)
        print("  Zipf's Law Analysis")
        print("=" * 60)
        print(f"  Fitted Zipfian exponent α = {alpha:.4f}")
        print(f"  R² (goodness of fit)      = {r_squared:.4f}")

        if abs(alpha - 1.0) < 0.1:
            verdict = "✓ Excellent fit to Zipf's law (α ≈ 1.0, typical of natural language)"
        elif abs(alpha - 1.0) < 0.3:
            verdict = "~ Good fit (α close to 1.0)"
        else:
            verdict = f"! Deviation from Zipf (α = {alpha:.2f}). Check for stopword removal or domain-specific corpus."

        print(f"  Verdict: {verdict}")

        # Show top-10 rank vs expected Zipf frequency
        print(f"\n  {'Rank':>5}  {'Actual freq':>12}  {'Zipf pred':>12}  {'Term':<15}")
        C_val = math.exp(C)
        for r, f in zip(ranks[:10], freqs[:10]):
            expected = C_val / (r ** alpha)
            try:
                term_id = sorted(sum_tfs, key=lambda x: -sum_tfs[x])[r - 1]
                term = self._term_id_map[term_id]
            except Exception:
                term = "?"
            print(f"  {r:>5}  {f:>12,}  {expected:>12.0f}  {term:<15}")
        print("=" * 60)

        return {'alpha': alpha, 'r_squared': r_squared}


    def document_length_report(self):
        """Analyze the distribution of document lengths."""
        st = self._collect_stats()
        doc_lengths = list(st['doc_lengths'].values())

        if not doc_lengths:
            print("No document length data available.")
            return

        arr = np.array(doc_lengths, dtype=np.float64)
        percentiles = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])

        print("\n" + "=" * 60)
        print("  Document Length Distribution")
        print("=" * 60)
        print(f"  N docs    : {len(arr):,}")
        print(f"  Mean      : {arr.mean():.1f} tokens")
        print(f"  Std dev   : {arr.std():.1f}")
        print(f"  Min       : {arr.min():.0f}")
        print(f"  Max       : {arr.max():.0f}")
        print(f"  p10       : {percentiles[0]:.0f}")
        print(f"  p25       : {percentiles[1]:.0f}")
        print(f"  Median    : {percentiles[2]:.0f}")
        print(f"  p75       : {percentiles[3]:.0f}")
        print(f"  p90       : {percentiles[4]:.0f}")
        print(f"  p95       : {percentiles[5]:.0f}")
        print(f"  p99       : {percentiles[6]:.0f}")

        # ASCII histogram
        print("\n  Length distribution histogram:")
        bins = np.linspace(arr.min(), arr.max(), 11)
        counts, _ = np.histogram(arr, bins=bins)
        max_count = max(counts)
        bar_width = 40
        for i, (cnt, left, right) in enumerate(zip(counts, bins[:-1], bins[1:])):
            bar = '█' * int(bar_width * cnt / max_count)
            print(f"  {int(left):>5}–{int(right):<5}  {bar:<40}  {cnt:>4}")

        # Identify outliers (very short / very long)
        very_short = sum(1 for l in doc_lengths if l < percentiles[0] * 0.5)
        very_long  = sum(1 for l in doc_lengths if l > percentiles[6] * 2)
        if very_short > 0:
            print(f"\n  ⚠ {very_short} docs with < {percentiles[0]*0.5:.0f} tokens (potential data quality issue)")
        if very_long > 0:
            print(f"  ⚠ {very_long} docs with > {percentiles[6]*2:.0f} tokens (unusually long)")
        print("=" * 60)


    def compression_report(self):
        """Report index file sizes and theoretical compression estimates."""
        index_file = os.path.join(self.output_dir, f'{self.index_name}.index')
        dict_file  = os.path.join(self.output_dir, f'{self.index_name}.dict')

        print("\n" + "=" * 60)
        print("  Index Compression Report")
        print("=" * 60)

        st = self._collect_stats()
        V = st['V']
        N = st['N']
        total_postings = sum(st['dfs'].values())

        # Theoretical uncompressed size (4 bytes per int)
        theoretical_bytes = total_postings * 2 * 4   # postings + TF
        print(f"  Vocabulary size V        : {V:>10,}")
        print(f"  Collection size N        : {N:>10,}")
        print(f"  Total posting entries    : {total_postings:>10,}")
        print(f"  Theoretical size (4B/int): {theoretical_bytes/1024:>10.1f} KB")

        if os.path.exists(index_file):
            actual_kb = os.path.getsize(index_file) / 1024
            ratio = theoretical_bytes / (actual_kb * 1024)
            print(f"  Actual index file        : {actual_kb:>10.1f} KB")
            print(f"  Compression ratio        : {ratio:>10.2f}x")
        else:
            print(f"  Index file not found at {index_file}")

        if os.path.exists(dict_file):
            dict_kb = os.path.getsize(dict_file) / 1024
            print(f"  Dictionary file          : {dict_kb:>10.1f} KB")

        avg_pl = total_postings / V if V > 0 else 0
        print(f"  Avg postings list length : {avg_pl:>10.1f}")

        long_threshold = N * 0.5
        long_terms = [(tid, df) for tid, df in st['dfs'].items() if df > long_threshold]
        if long_terms:
            print(f"\n  ⚠ {len(long_terms)} terms appear in > 50% of docs:")
            for tid, df in sorted(long_terms, key=lambda x: -x[1])[:10]:
                try:
                    term = self._term_id_map[tid]
                except Exception:
                    term = f"<id:{tid}>"
                print(f"    '{term}'  df={df}  ({100*df/N:.0f}% of docs)")
        print("=" * 60)


    def predict_query_difficulty(self, query: str) -> dict:
        """
        Estimate how "difficult" a query is for this collection.

        Difficulty signals:
        - Average IDF of query terms: low IDF → common terms → hard
        - IDF variance: high variance → mixed terms → uncertain
        - Min DF: if any term has very low DF, few results expected
        - Query length: longer queries are generally harder (more conjunction)
        - Specificity score: composite difficulty estimate [0=easy, 1=hard]

        Parameters
        ----------
        query : str

        Returns
        -------
        dict with keys: idfs, avg_idf, idf_variance, min_df, max_df,
                        expected_results, specificity, difficulty_label
        """
        self._load_maps()
        st = self._collect_stats()
        N = st['N']
        dfs = st['dfs']

        tokens = query.split()
        term_data = []

        for token in tokens:
            if hasattr(self._term_id_map, '_lookup'):
                tid = self._term_id_map._lookup(token)
            elif hasattr(self._term_id_map, 'str_to_id'):
                tid = self._term_id_map.str_to_id.get(token)
            else:
                tid = None

            if tid is not None and tid in dfs:
                df = dfs[tid]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                term_data.append({'term': token, 'df': df, 'idf': idf})
            else:
                term_data.append({'term': token, 'df': 0, 'idf': 10.0})

        if not term_data:
            return {'difficulty_label': 'UNKNOWN', 'specificity': 0.5}

        idfs = [d['idf'] for d in term_data]
        dfs_q = [d['df'] for d in term_data]

        avg_idf = sum(idfs) / len(idfs)
        idf_var = np.var(idfs) if len(idfs) > 1 else 0.0
        min_df  = min(dfs_q)
        max_df  = max(dfs_q)

        prob_match = 1.0
        for df in dfs_q:
            prob_match *= (df / N) if df > 0 else 0.001
        expected_results = max(1, int(prob_match * N))

        max_possible_idf = math.log(N + 1)
        specificity = avg_idf / max_possible_idf if max_possible_idf > 0 else 0

        if specificity > 0.7 and min_df > 0:
            label = 'EASY  (specific terms, many good candidates)'
        elif specificity > 0.4:
            label = 'MEDIUM'
        elif min_df == 0:
            label = 'HARD  (at least one term not in vocabulary)'
        else:
            label = 'HARD  (very common terms, low discrimination)'

        print(f"\n  Query Difficulty Analysis: '{query}'")
        print(f"  {'Term':<20}  {'DF':>7}  {'IDF':>7}")
        for d in term_data:
            oov = " [OOV]" if d['df'] == 0 else ""
            print(f"  {d['term']:<20}  {d['df']:>7,}  {d['idf']:>7.3f}{oov}")
        print(f"  Average IDF         : {avg_idf:.3f}")
        print(f"  IDF variance        : {idf_var:.3f}")
        print(f"  Expected results    : ~{expected_results:,}")
        print(f"  Specificity score   : {specificity:.3f} / 1.0")
        print(f"  Difficulty          : {label}")

        return {
            'term_data': term_data,
            'avg_idf': avg_idf,
            'idf_variance': idf_var,
            'min_df': min_df,
            'max_df': max_df,
            'expected_results': expected_results,
            'specificity': specificity,
            'difficulty_label': label,
        }

    def full_report(self, top_n: int = 15, sample_queries: list = None):
        """Run all analyses and print the complete index health report."""
        print("\n" + "█" * 60)
        print("  INDEX HEALTH REPORT")
        print("█" * 60)

        self.vocabulary_report(top_n=top_n)
        self.zipf_analysis()
        self.document_length_report()
        self.compression_report()

        if sample_queries:
            print("\n" + "=" * 60)
            print("  Query Difficulty Predictions")
            print("=" * 60)
            for q in sample_queries:
                self.predict_query_difficulty(q)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Index Inspector')
    parser.add_argument('--output-dir', default='index')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Top N terms to show in vocabulary report')
    parser.add_argument('--query', type=str, default=None,
                        help='Predict difficulty for this query')
    args = parser.parse_args()

    inspector = IndexInspector(output_dir=args.output_dir)

    sample_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
        "the of and",
        "xylophones jazz",
    ]

    if args.query:
        inspector.predict_query_difficulty(args.query)
    else:
        inspector.full_report(
            top_n=args.top_n,
            sample_queries=sample_queries
        )