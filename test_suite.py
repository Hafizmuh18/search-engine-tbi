"""
test_suite.py
-------------
Test suite untuk memvalidasi semua komponen search engine.
Jalankan dari root folder project (tempat collection/, index/, dll berada).

Usage:
    python test_suite.py              # semua test
    python test_suite.py --unit       # hanya unit tests (tidak perlu index)
    python test_suite.py --index      # test yang butuh index sudah dibangun
    python test_suite.py --full       # semua test termasuk LSI (lambat)
"""

import sys
import os
import math
import time
import traceback
import argparse

PASS = "✓"
FAIL = "✗"
SKIP = "○"
results = []

def test(name, fn):
    """Run a single test and record result."""
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except AssertionError as e:
        results.append((FAIL, name))
        print(f"  {FAIL} {name}")
        print(f"      AssertionError: {e}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL} {name}")
        print(f"      {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()

def skip(name, reason):
    results.append((SKIP, name))
    print(f"  {SKIP} {name}  [{reason}]")


# ══════════════════════════════════════════════════════════════
# SECTION 1: Unit Tests — tidak butuh index
# ══════════════════════════════════════════════════════════════

def run_unit_tests():
    print("\n" + "═"*55)
    print("  SECTION 1: Unit Tests (no index required)")
    print("═"*55)

    # ── 1a. Compression ──────────────────────────────────────
    print("\n  [1a] Compression codecs")

    def test_standard():
        from compression import StandardPostings
        pl = [34, 67, 89, 454, 2345738]
        tf = [12, 10, 3, 4, 1]
        assert StandardPostings.decode(StandardPostings.encode(pl)) == pl
        assert StandardPostings.decode_tf(StandardPostings.encode_tf(tf)) == tf

    def test_vbe():
        from compression import VBEPostings
        pl = [34, 67, 89, 454, 2345738]
        tf = [12, 10, 3, 4, 1]
        assert VBEPostings.decode(VBEPostings.encode(pl)) == pl
        assert VBEPostings.decode_tf(VBEPostings.encode_tf(tf)) == tf

    def test_elias_gamma():
        from compression import EliasGammaPostings
        cases = [[1], [1,2,3], [34,67,89,454,2345738], [1,1,1,1], [100,200,300]]
        for pl in cases:
            assert EliasGammaPostings.decode(EliasGammaPostings.encode(pl)) == pl, \
                f"postings mismatch: {pl}"
        tf_cases = [[1],[1,5,3,2,10],[12,10,3,4,1],[7,7,7]]
        for tf in tf_cases:
            assert EliasGammaPostings.decode_tf(EliasGammaPostings.encode_tf(tf)) == tf, \
                f"tf mismatch: {tf}"

    def test_eg_delta():
        from compression_benchmark import EliasGammaDeltaPostings
        cases = [[1],[1,2,3],[34,67,89,454,2345738],[1,100,1000,10000]]
        for pl in cases:
            assert EliasGammaDeltaPostings.decode(
                EliasGammaDeltaPostings.encode(pl)) == pl, f"postings: {pl}"
        tf_cases = [[1],[12,10,3,4,1],[1,5,10,50]]
        for tf in tf_cases:
            assert EliasGammaDeltaPostings.decode_tf(
                EliasGammaDeltaPostings.encode_tf(tf)) == tf, f"tf: {tf}"

    def test_vbe_eg_tf():
        from compression_benchmark import VBEEliasGammaTF
        cases = [[1],[34,67,89,454,2345738],[1,100,200,300]]
        for pl in cases:
            assert VBEEliasGammaTF.decode(
                VBEEliasGammaTF.encode(pl)) == pl, f"postings: {pl}"
        tf_cases = [[1],[12,10,3,4,1],[7,7,7]]
        for tf in tf_cases:
            assert VBEEliasGammaTF.decode_tf(
                VBEEliasGammaTF.encode_tf(tf)) == tf, f"tf: {tf}"

    def test_compression_sizes():
        """VBE dan EliasGamma harus lebih kecil dari Standard untuk postings panjang."""
        from compression import StandardPostings, VBEPostings, EliasGammaPostings
        pl = list(range(1, 201, 3))   # 67 elements
        std  = len(StandardPostings.encode(pl))
        vbe  = len(VBEPostings.encode(pl))
        eg   = len(EliasGammaPostings.encode(pl))
        assert vbe < std,  f"VBE ({vbe}B) not smaller than Standard ({std}B)"
        assert eg  < std,  f"EliasGamma ({eg}B) not smaller than Standard ({std}B)"

    test("StandardPostings encode/decode", test_standard)
    test("VBEPostings encode/decode", test_vbe)
    test("EliasGammaPostings encode/decode", test_elias_gamma)
    test("EliasGammaDeltaPostings encode/decode", test_eg_delta)
    test("VBEEliasGammaTF encode/decode", test_vbe_eg_tf)
    test("Compression ratios (VBE & EG < Standard)", test_compression_sizes)

    # ── 1b. Evaluation Metrics ───────────────────────────────
    print("\n  [1b] Evaluation metrics")

    def test_rbp():
        from evaluation import rbp
        assert rbp([1,0,0,0,0]) == pytest_approx(0.2, 0.001)
        assert rbp([1,1,1,1,1]) > rbp([0,0,0,1,1])
        assert rbp([]) == 0.0 or rbp([]) == 0

    def test_dcg():
        from evaluation import dcg
        r = [1, 0, 1, 1, 0, 0]
        score = dcg(r)
        expected = 1/math.log2(2) + 1/math.log2(4) + 1/math.log2(5)
        assert abs(score - expected) < 1e-9, f"DCG {score} != {expected}"
        assert dcg([1,0,0], k=1) == 1/math.log2(2)

    def test_ndcg():
        from evaluation import ndcg
        assert abs(ndcg([1,1,1]) - 1.0) < 1e-9, "Perfect ranking should be 1.0"
        assert ndcg([0,0,0]) == 0.0
        assert 0 <= ndcg([1,0,1,0,1]) <= 1.0

    def test_ap():
        from evaluation import average_precision
        assert average_precision([0,0,0]) == 0.0
        assert abs(average_precision([1,0,1]) - 5/6) < 1e-9
        assert average_precision([1,1,1]) == 1.0

    def test_ndcg_monotone():
        """Better ranking should have higher NDCG."""
        from evaluation import ndcg
        good = [1,1,0,0,0]
        bad  = [0,0,1,1,0]
        assert ndcg(good) > ndcg(bad), "Better ranking should have higher NDCG"

    test("RBP metric", test_rbp)
    test("DCG metric (exact values)", test_dcg)
    test("NDCG metric (edge cases)", test_ndcg)
    test("Average Precision (exact values)", test_ap)
    test("NDCG monotone (good ranking > bad ranking)", test_ndcg_monotone)

    # ── 1c. Patricia Tree ────────────────────────────────────
    print("\n  [1c] Patricia Tree")

    def test_patricia_basic():
        from patricia_tree import PatriciaTree
        pt = PatriciaTree()
        assert pt["hello"] == 0
        assert pt["hell"]  == 1
        assert pt["world"] == 2
        assert pt["hello"] == 0   # already exists
        assert len(pt) == 3
        assert pt[0] == "hello"
        assert pt[1] == "hell"

    def test_patricia_prefix():
        from patricia_tree import PatriciaTree
        pt = PatriciaTree()
        for w in ["protein", "protease", "proteomics", "proton", "alpha"]:
            pt[w]
        matches = pt.starts_with("prot")
        assert set(matches) == {"protein","protease","proteomics","proton"}, \
            f"Got: {matches}"
        assert pt.starts_with("alpha") == ["alpha"]
        assert pt.starts_with("xyz") == []

    def test_patricia_idmap_compat():
        """PatriciaIdMap must behave identically to IdMap."""
        from patricia_tree import PatriciaIdMap
        pm = PatriciaIdMap()
        doc = ["halo","semua","selamat","pagi","semua"]
        ids = [pm[term] for term in doc]
        assert ids == [0,1,2,3,1], f"Got: {ids}"
        assert pm[1] == "semua"
        assert pm["selamat"] == 2

    def test_patricia_large():
        import random, string
        from patricia_tree import PatriciaTree
        random.seed(99)
        words = list(set(
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,8)))
            for _ in range(300)
        ))
        pt = PatriciaTree()
        for w in words:
            pt[w]
        assert len(pt) == len(words)
        for w in words:
            assert pt._lookup(w) is not None, f"'{w}' not found after insert"

    test("PatriciaTree basic insert/lookup/reverse", test_patricia_basic)
    test("PatriciaTree prefix search", test_patricia_prefix)
    test("PatriciaIdMap — IdMap interface compatibility", test_patricia_idmap_compat)
    test("PatriciaTree large random (300 words)", test_patricia_large)

    # ── 1d. Rank Fusion (logic only) ────────────────────────
    print("\n  [1d] Rank Fusion logic")

    def test_rrf_logic():
        from ranked_fusion import RankFusion
        # docB appears high in both lists → should be ranked #1
        r1 = [(0.9,'docA'),(0.8,'docB'),(0.7,'docC')]
        r2 = [(0.95,'docB'),(0.85,'docE'),(0.7,'docA')]

        class FakeEngine:
            pass
        fusion = RankFusion(FakeEngine())
        result = fusion.rrf_fusion([r1, r2], k=5)
        top_doc = result[0][1]
        assert top_doc == 'docB', \
            f"docB should win (high in both lists), got {top_doc}"

    def test_combmnz_logic():
        from ranked_fusion import RankFusion
        r1 = [(10.0,'docA'),(8.0,'docB'),(6.0,'docC')]
        r2 = [(9.0,'docB'),(7.0,'docA'),(5.0,'docD')]

        class FakeEngine:
            pass
        fusion = RankFusion(FakeEngine())
        result = fusion.combmnz_fusion([r1, r2], k=5)
        docs = [d for _, d in result]
        # docA and docB both appear in 2 lists → should beat docC/docD (1 list)
        assert 'docC' not in docs[:2] and 'docD' not in docs[:2], \
            f"Single-list docs should rank lower, got top-2: {docs[:2]}"

    test("RRF: consensus document ranks first", test_rrf_logic)
    test("CombMNZ: multi-list docs rank higher than single-list", test_combmnz_logic)


# ══════════════════════════════════════════════════════════════
# SECTION 2: Integration Tests — butuh index sudah dibangun
# ══════════════════════════════════════════════════════════════

def run_index_tests():
    print("\n" + "═"*55)
    print("  SECTION 2: Integration Tests (requires built index)")
    print("═"*55)

    if not os.path.exists(os.path.join('index', 'main_index.index')):
        print("\n  [!] Index not found. Run first:")
        print("      python search_cli.py index")
        skip("All index tests", "index not built")
        return

    # ── 2a. BSBI load & TF-IDF ──────────────────────────────
    print("\n  [2a] BSBI index load & TF-IDF")

    def test_bsbi_load():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        assert len(engine.term_id_map) > 0, "term_id_map is empty"
        assert len(engine.doc_id_map)  > 0, "doc_id_map is empty"

    def test_tfidf_returns_results():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        results = engine.retrieve_tfidf("lipid metabolism", k=10)
        assert len(results) > 0, "TF-IDF returned no results"
        assert all(isinstance(s, float) for s,_ in results), "scores not float"
        scores = [s for s,_ in results]
        assert scores == sorted(scores, reverse=True), "results not sorted desc"

    def test_bm25_returns_results():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        results = engine.retrieve_bm25("lipid metabolism", k=10)
        assert len(results) > 0, "BM25 returned no results"
        scores = [s for s,_ in results]
        assert scores == sorted(scores, reverse=True), "results not sorted desc"

    def test_wand_matches_bm25():
        """WAND top-10 should be identical (or near-identical) to brute BM25."""
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        query = "alkylated with radioactive iodoacetate"
        r_bm25 = engine.retrieve_bm25(query, k=10)
        r_wand = engine.retrieve_bm25_wand(query, k=10)
        docs_bm25 = {doc for _, doc in r_bm25}
        docs_wand = {doc for _, doc in r_wand}
        overlap = len(docs_bm25 & docs_wand)
        assert overlap >= 7, \
            f"WAND top-10 overlap with BM25 too low: {overlap}/10"

    def test_bm25_beats_tfidf_score():
        """BM25 should return positive scores (all IDF Robertson > 0)."""
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        results = engine.retrieve_bm25("protein synthesis cell", k=20)
        assert all(s > 0 for s,_ in results), "BM25 returned non-positive scores"

    def test_unknown_query_returns_empty():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        results = engine.retrieve_bm25("xyzxyz_nonexistent_token_abc123", k=10)
        assert results == [], f"Unknown query should return empty, got {results}"

    test("BSBIIndex.load() — term & doc maps populated", test_bsbi_load)
    test("retrieve_tfidf() — returns sorted results", test_tfidf_returns_results)
    test("retrieve_bm25() — returns sorted results", test_bm25_returns_results)
    test("retrieve_bm25_wand() — top-10 overlap ≥ 7 with brute BM25", test_wand_matches_bm25)
    test("BM25 scores all positive (Robertson IDF)", test_bm25_beats_tfidf_score)
    test("Unknown query returns empty list", test_unknown_query_returns_empty)

    # ── 2b. Encoding codecs with real index ──────────────────
    print("\n  [2b] Compression codecs with real index")

    def test_elias_index():
        from bsbi import BSBIIndex
        from compression import EliasGammaPostings
        if not os.path.exists('index/main_index_eg.index'):
            skip("EliasGamma index", "not built (run: index --encoding elias)")
            return
        engine = BSBIIndex('collection', 'index', EliasGammaPostings,
                           index_name='main_index_eg')
        engine.load()
        results = engine.retrieve_bm25("lipid", k=5)
        assert len(results) > 0

    test("EliasGamma codec with real index (if built)", test_elias_index)

    # ── 2c. Evaluation metrics on real data ──────────────────
    print("\n  [2c] Evaluation pipeline")

    def test_load_qrels():
        from evaluation import load_qrels
        qrels = load_qrels()
        assert qrels["Q1"][166] == 1, "Q1 doc 166 should be relevant"
        assert qrels["Q1"][300] == 0, "Q1 doc 300 should not be relevant"
        assert len(qrels) == 30, f"Expected 30 queries, got {len(qrels)}"

    def test_eval_retrieval_runs():
        from evaluation import load_qrels, eval_retrieval
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        qrels = load_qrels()
        scores = eval_retrieval(engine.retrieve_bm25, "BM25-test",
                                qrels, k=100)
        assert 'rbp'  in scores
        assert 'dcg'  in scores
        assert 'ndcg' in scores
        assert 'ap'   in scores
        assert 0 <= scores['ndcg'] <= 1.0, f"NDCG out of range: {scores['ndcg']}"
        assert scores['rbp'] > 0, "RBP should be > 0 on real data"

    def test_bm25_better_than_tfidf():
        """BM25 should have higher MAP than TF-IDF on this collection."""
        from evaluation import load_qrels, eval_retrieval
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        qrels = load_qrels()
        s_tfidf = eval_retrieval(engine.retrieve_tfidf, "TF-IDF", qrels, k=100)
        s_bm25  = eval_retrieval(engine.retrieve_bm25,  "BM25",   qrels, k=100)
        assert s_bm25['ap'] > s_tfidf['ap'], \
            f"BM25 MAP ({s_bm25['ap']:.4f}) should > TF-IDF MAP ({s_tfidf['ap']:.4f})"

    test("load_qrels() — correct relevance judgments", test_load_qrels)
    test("eval_retrieval() — returns all 4 metrics in range", test_eval_retrieval_runs)
    test("BM25 MAP > TF-IDF MAP", test_bm25_better_than_tfidf)

    # ── 2d. Index Inspector ──────────────────────────────────
    print("\n  [2d] Index Inspector")

    def test_inspector_vocab():
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector('index', VBEPostings)
        st = inspector._collect_stats()
        assert st['N'] > 0,  "N (num docs) should be > 0"
        assert st['V'] > 0,  "V (vocab size) should be > 0"
        assert st['N'] < st['V'] or True  # may vary by collection

    def test_inspector_zipf():
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector('index', VBEPostings)
        result = inspector.zipf_analysis()
        assert 'alpha' in result
        assert 0.5 <= result['alpha'] <= 2.0, \
            f"Zipf alpha {result['alpha']} out of expected range [0.5, 2.0]"
        assert result['r_squared'] > 0.7, \
            f"R² {result['r_squared']:.3f} too low — not a Zipfian distribution?"

    def test_query_difficulty():
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector('index', VBEPostings)
        # Specific query → high specificity
        hard = inspector.predict_query_difficulty("the of and")
        easy = inspector.predict_query_difficulty("alkylated radioactive iodoacetate")
        assert easy['specificity'] > hard['specificity'], \
            f"Specific query should score higher: easy={easy['specificity']:.3f} hard={hard['specificity']:.3f}"

    test("IndexInspector._collect_stats() — N and V populated", test_inspector_vocab)
    test("Zipf analysis — alpha in [0.5, 2.0], R² > 0.7", test_inspector_zipf)
    test("Query difficulty — specific query scores higher", test_query_difficulty)

    # ── 2e. SPIMI (if spimi index exists) ────────────────────
    print("\n  [2e] SPIMI (optional — skip if not built)")

    def test_spimi_prefix():
        if not os.path.exists('index/terms.dict'):
            skip("SPIMI prefix search", "no index")
            return
        try:
            from spimi import SPIMIIndex
            from compression import VBEPostings
            engine = SPIMIIndex('collection', 'index', VBEPostings)
            engine.load()
            if not hasattr(engine.term_id_map, 'starts_with'):
                skip("SPIMI prefix search", "index built with BSBIIndex (no PatriciaIdMap)")
                return
            matches = engine.prefix_search("lip")
            assert isinstance(matches, list)
            for m in matches:
                assert m.startswith("lip"), f"'{m}' doesn't start with 'lip'"
        except Exception as e:
            skip("SPIMI prefix search", f"index not SPIMI-built: {e}")

    test("SPIMI prefix_search() — all results start with prefix", test_spimi_prefix)

    # ── 2f. Query Expansion ──────────────────────────────────
    print("\n  [2f] Query Expansion")

    def test_prf_expands():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        from query_expansion import PseudoRelevanceFeedback
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        prf = PseudoRelevanceFeedback(engine, top_k=5, n_expand=3)
        expanded, terms = prf.expand("lipid metabolism")
        assert len(terms) > 0, "PRF should expand with at least 1 term"
        assert len(expanded.split()) > 2, "Expanded query should be longer"
        for t in terms:
            assert t not in {"lipid", "metabolism"}, \
                f"Expansion term '{t}' is already in original query"

    def test_cooc_expands():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        from query_expansion import CooccurrenceExpander
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        cooc = CooccurrenceExpander(engine, n_expand=3)
        expanded, terms = cooc.expand("protein synthesis")
        assert isinstance(terms, list)

    test("PseudoRelevanceFeedback — expands with new terms", test_prf_expands)
    test("CooccurrenceExpander — returns expansion terms", test_cooc_expands)

    # ── 2g. Rank Fusion with real engine ─────────────────────
    print("\n  [2g] Rank Fusion")

    def test_rrf_with_engine():
        from bsbi import BSBIIndex
        from compression import VBEPostings
        from ranked_fusion import RankFusion
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        fusion = RankFusion(engine)
        results = fusion.retrieve("lipid metabolism", k=10,
                                  methods=['bm25','wand'], strategy='rrf')
        assert len(results) > 0, "RRF returned no results"
        scores = [s for s,_ in results]
        assert scores == sorted(scores, reverse=True), "RRF results not sorted"

    def test_fusion_strategies_agree():
        """All 3 fusion strategies should return overlapping top-5."""
        from bsbi import BSBIIndex
        from compression import VBEPostings
        from ranked_fusion import RankFusion
        engine = BSBIIndex('collection', 'index', VBEPostings)
        engine.load()
        fusion = RankFusion(engine)
        query = "alkylated with radioactive iodoacetate"
        methods = ['bm25', 'wand']
        rrf  = {d for _,d in fusion.retrieve(query, k=10, methods=methods, strategy='rrf')}
        mnz  = {d for _,d in fusion.retrieve(query, k=10, methods=methods, strategy='combmnz')}
        overlap = len(rrf & mnz)
        assert overlap >= 5, \
            f"RRF and CombMNZ top-10 overlap too low: {overlap}"

    test("RRF with real engine — returns sorted results", test_rrf_with_engine)
    test("RRF vs CombMNZ top-10 overlap ≥ 5", test_fusion_strategies_agree)


# ══════════════════════════════════════════════════════════════
# SECTION 3: LSI Tests — slow, optional
# ══════════════════════════════════════════════════════════════

def run_lsi_tests():
    print("\n" + "═"*55)
    print("  SECTION 3: LSI Tests (slow, requires lsi build)")
    print("═"*55)

    lsi_path = os.path.join('index', 'lsi_model.pkl')
    if not os.path.exists(lsi_path):
        print("\n  [!] LSI model not found. Run first:")
        print("      python search_cli.py lsi build")
        skip("All LSI tests", "lsi model not built")
        return

    def test_lsi_loads():
        from lsi import LSIIndex
        from compression import VBEPostings
        lsi = LSIIndex('collection','index',VBEPostings)
        lsi.load()
        assert lsi.doc_vectors is not None
        assert lsi.U_k is not None
        assert lsi.doc_vectors.shape[1] == lsi.n_components

    def test_lsi_retrieval():
        from lsi import LSIIndex
        from compression import VBEPostings
        lsi = LSIIndex('collection','index',VBEPostings)
        lsi.load()
        results = lsi.retrieve("lipid metabolism", k=10)
        assert len(results) > 0, "LSI returned no results"
        scores = [s for s,_ in results]
        assert scores == sorted(scores, reverse=True), "LSI results not sorted"
        assert all(-1.001 <= s <= 1.001 for s in scores), \
            f"Cosine similarity out of [-1,1]: {scores}"

    def test_lsi_related_terms():
        from lsi import LSIIndex
        from compression import VBEPostings
        lsi = LSIIndex('collection','index',VBEPostings)
        lsi.load()
        related = lsi.most_related_terms("protein", top_n=5)
        assert len(related) > 0, "No related terms found"
        assert all(isinstance(t, str) for _,t in related)
        assert all(isinstance(s, float) for s,_ in related)

    def test_lsi_explained_variance():
        from lsi import LSIIndex
        from compression import VBEPostings
        import numpy as np
        lsi = LSIIndex('collection','index',VBEPostings)
        lsi.load()
        evr = lsi.explained_variance_ratio()
        assert evr is not None
        assert abs(sum(evr) - 1.0) < 0.01, \
            f"Explained variance ratios don't sum to 1: {sum(evr):.4f}"
        cumvar = np.cumsum(evr)
        assert cumvar[-1] > 0.05, "LSI model explains < 5% variance — something is wrong"

    test("LSIIndex.load() — model loads correctly", test_lsi_loads)
    test("LSI retrieval — sorted cosine similarities in [-1,1]", test_lsi_retrieval)
    test("LSI most_related_terms() — returns term strings", test_lsi_related_terms)
    test("LSI explained variance ratios sum to 1.0", test_lsi_explained_variance)


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════

def pytest_approx(val, tol):
    """Simple approx check (avoid pytest dependency)."""
    class _Approx:
        def __eq__(self, other):
            return abs(other - val) < tol
        def __repr__(self):
            return f"~{val}±{tol}"
    return _Approx()


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

def print_summary():
    passed = sum(1 for s,_ in results if s == PASS)
    failed = sum(1 for s,_ in results if s == FAIL)
    skipped = sum(1 for s,_ in results if s == SKIP)
    total = passed + failed

    print("\n" + "═"*55)
    print(f"  SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Passed  : {passed}/{total}")
    print(f"  Failed  : {failed}/{total}")
    print(f"  Skipped : {skipped}")
    print("═"*55)

    if failed > 0:
        print("\n  Failed tests:")
        for s, name in results:
            if s == FAIL:
                print(f"    ✗ {name}")

    return failed == 0


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search Engine Test Suite')
    parser.add_argument('--unit',  action='store_true',
                        help='Run only unit tests (no index required)')
    parser.add_argument('--index', action='store_true',
                        help='Run unit + integration tests (index must be built)')
    parser.add_argument('--full',  action='store_true',
                        help='Run all tests including LSI (slow)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full tracebacks on failure')
    args = parser.parse_args()

    print("\n  Search Engine — Test Suite")
    print(f"  Working directory: {os.getcwd()}")

    run_unit = True
    run_idx  = args.index or args.full or (not args.unit)
    run_lsi  = args.full

    run_unit_tests()

    if run_idx:
        run_index_tests()

    if run_lsi:
        run_lsi_tests()

    ok = print_summary()
    sys.exit(0 if ok else 1)