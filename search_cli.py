"""
search_cli.py
-------------
Unified entry point for the entire search engine system.
Integrates: BSBI/SPIMI indexing, all compression codecs, BM25/TF-IDF/WAND/LSI
retrieval, query expansion (PRF/Co-occurrence), rank fusion (RRF/Condorcet/CombMNZ),
index inspection, compression benchmarking, and evaluation (RBP/DCG/NDCG/AP/MAP).

SUBCOMMANDS
  index     Build the inverted index
  search    Run retrieval for one query
  eval      Evaluate retrieval methods with all metrics
  lsi       Build, query, or inspect the LSI model
  inspect   Show index health analytics
  bench     Benchmark compression codecs
  repl      Interactive search REPL

Run  python search_cli.py <subcommand> --help  for per-command options.
"""

import argparse
import os
import re
import sys
import time


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _build_encoding_map():
    from compression import StandardPostings, VBEPostings, EliasGammaPostings
    from compression_benchmark import EliasGammaDeltaPostings, VBEEliasGammaTF
    return {
        'standard':  StandardPostings,
        'vbe':       VBEPostings,
        'elias':     EliasGammaPostings,
        'eg-delta':  EliasGammaDeltaPostings,
        'vbe-eg-tf': VBEEliasGammaTF,
    }

ENCODING_CHOICES = ['standard', 'vbe', 'elias', 'eg-delta', 'vbe-eg-tf']


def _make_engine(index_method, encoding, data_dir='collection',
                 output_dir='index', max_tokens=150_000):
    enc = _build_encoding_map()[encoding]
    if index_method == 'spimi':
        from spimi import SPIMIIndex
        return SPIMIIndex(data_dir=data_dir, output_dir=output_dir,
                          postings_encoding=enc,
                          max_tokens_per_block=max_tokens)
    from bsbi import BSBIIndex
    return BSBIIndex(data_dir=data_dir, output_dir=output_dir,
                     postings_encoding=enc)


def _make_lsi(encoding, output_dir='index', n_components=100):
    from lsi import LSIIndex
    enc = _build_encoding_map()[encoding]
    return LSIIndex(data_dir='collection', output_dir=output_dir,
                    postings_encoding=enc, n_components=n_components)


def _load_lsi_if_exists(encoding, output_dir='index', n_components=100):
    lsi_path = os.path.join(output_dir, 'lsi_model.pkl')
    if not os.path.exists(lsi_path):
        return None
    lsi = _make_lsi(encoding, output_dir, n_components)
    lsi.load()
    return lsi


def _run_query(engine, query, method, k, lsi=None, fusion=None):
    """Dispatch query to correct backend. Returns (results, expanded_query, extra)."""
    extra = {}

    if method in ('rrf', 'condorcet', 'combmnz', 'combsum'):
        if fusion is None:
            from ranked_fusion import RankFusion
            fusion = RankFusion(engine, lsi=lsi)
        base = ['bm25', 'wand']
        if lsi:
            base.append('lsi')
        results = fusion.retrieve(query, k=k, methods=base, strategy=method)
        extra['fusion_methods'] = base
        return results, query, extra

    if method == 'tfidf':
        return engine.retrieve_tfidf(query, k=k), query, extra
    if method == 'bm25':
        return engine.retrieve_bm25(query, k=k), query, extra
    if method == 'wand':
        return engine.retrieve_bm25_wand(query, k=k), query, extra
    if method == 'lsi':
        if lsi is None:
            print("  [!] LSI not loaded. Run: python search_cli.py lsi build")
            return [], query, extra
        return lsi.retrieve(query, k=k), query, extra

    if '+' in method or method in ('prf', 'cooc'):
        from query_expansion import QueryExpansionPipeline
        pipeline = QueryExpansionPipeline(engine, top_k_feedback=10, n_expand=5)
        m = method if '+' in method else method + '+bm25'
        out = pipeline.run(query, method=m, k=k, lsi=lsi)
        extra['expansion_terms'] = out['expansion_terms']
        return out['results'], out['expanded_query'], extra

    print(f"  [!] Unknown method '{method}', using bm25.")
    return engine.retrieve_bm25(query, k=k), query, extra


def _print_results(results, method, k, query, expanded_query, extra, verbose=False):
    if expanded_query != query:
        print(f"  Expanded : {expanded_query}")
        if extra.get('expansion_terms'):
            print(f"  New terms: {extra['expansion_terms']}")
    if extra.get('fusion_methods'):
        print(f"  Fused    : {extra['fusion_methods']}")
    print(f"  Method   : {method}   k={k}")
    print()
    if not results:
        print("  No results found.")
        return
    for i, (score, doc) in enumerate(results, 1):
        m = re.search(r'/([^/]+)\.txt$', doc)
        label = m.group(1) if m else doc
        line = f"  {i:3}.  score={score:9.5f}   {doc}" if verbose else \
               f"  {i:3}.  [{score:7.4f}]  {label}"
        print(line)
    print()


def _compare_all_methods(engine, query, k, lsi):
    methods = ['tfidf', 'bm25', 'wand']
    if lsi:
        methods += ['lsi', 'prf+bm25']
    methods += ['rrf']
    from ranked_fusion import RankFusion
    fusion = RankFusion(engine, lsi=lsi)
    print(f"  Comparing {len(methods)} methods (top-{min(k,5)} shown):\n")
    for method in methods:
        t0 = time.perf_counter()
        results, expanded_q, extra = _run_query(engine, query, method, k, lsi, fusion)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  ── {method:<15}  ({ms:.0f} ms) ──")
        for i, (score, doc) in enumerate(results[:5], 1):
            m = re.search(r'/([^/]+)\.txt$', doc)
            print(f"     {i}. [{score:7.4f}]  {m.group(1) if m else doc}")
        print()


def _run_bench_internal(output_dir='index'):
    from compression_benchmark import (run_benchmark, tf_distribution_analysis,
                                       _generate_realistic_data)
    results = run_benchmark(output_dir=output_dir)
    _, all_tfs = _generate_realistic_data(n_terms=1000)
    tf_distribution_analysis(all_tfs)
    return results


# ─────────────────────────────────────────────────────────────
# index
# ─────────────────────────────────────────────────────────────

def cmd_index(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('tmp', exist_ok=True)
    print(f"\n{'='*55}\n  Indexing\n{'='*55}")
    print(f"  Method   : {args.method}")
    print(f"  Encoding : {args.encoding}")
    if args.method == 'spimi':
        print(f"  Max tokens/block: {args.max_tokens:,}")

    cleaned = sum(1 for f in os.listdir(args.output_dir)
                  if f.endswith('.index') or
                  (f.endswith('.dict') and f != 'lsi_model.pkl'))
    for f in os.listdir(args.output_dir):
        if f.endswith('.index') or (f.endswith('.dict') and f != 'lsi_model.pkl'):
            os.remove(os.path.join(args.output_dir, f))
    if cleaned:
        print(f"  Removed {cleaned} stale file(s).")

    t0 = time.perf_counter()
    engine = _make_engine(args.method, args.encoding,
                          data_dir=args.data_dir,
                          output_dir=args.output_dir,
                          max_tokens=args.max_tokens)
    engine.index()
    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.2f}s")

    idx = os.path.join(args.output_dir, 'main_index.index')
    if os.path.exists(idx):
        print(f"  Index file: {os.path.getsize(idx)/1024:.1f} KB")

    if args.method == 'spimi' and hasattr(engine, 'prefix_search'):
        print("\n  Patricia Tree prefix demo:")
        for pfx in ['alky', 'lip', 'prot']:
            matches = engine.prefix_search(pfx)
            print(f"    '{pfx}' → {matches[:6]}")

    if args.bench_after:
        print()
        _run_bench_internal(args.output_dir)


def _add_index_args(p):
    p.add_argument('--method', default='bsbi', choices=['bsbi', 'spimi'])
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--data-dir', default='collection')
    p.add_argument('--output-dir', default='index')
    p.add_argument('--max-tokens', type=int, default=150_000,
                   help='SPIMI: token pairs per spill (default: 150000)')
    p.add_argument('--bench-after', action='store_true',
                   help='Run codec benchmark after indexing')


# ─────────────────────────────────────────────────────────────
# search
# ─────────────────────────────────────────────────────────────

def cmd_search(args):
    engine = _make_engine(args.index_method, args.encoding,
                          output_dir=args.output_dir)
    engine.load()

    lsi = None
    if args.method in ('lsi', 'prf+lsi', 'rrf', 'condorcet', 'combmnz'):
        lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.lsi_k)
        if lsi is None and args.method == 'lsi':
            print("[!] No LSI model. Run: python search_cli.py lsi build")
            return

    print(f"\n{'='*55}\n  Query: {args.query}\n{'='*55}")

    if args.difficulty:
        from index_inspector import IndexInspector
        inspector = IndexInspector(args.output_dir,
                                   _build_encoding_map()[args.encoding])
        inspector.predict_query_difficulty(args.query)
        print()

    if args.compare:
        _compare_all_methods(engine, args.query, args.k, lsi)
        return

    results, expanded_q, extra = _run_query(
        engine, args.query, args.method, args.k, lsi)
    _print_results(results, args.method, args.k,
                   args.query, expanded_q, extra, verbose=args.verbose)


def _add_search_args(p):
    p.add_argument('--query', required=True)
    p.add_argument('--method', default='bm25',
                   choices=['tfidf','bm25','wand','lsi','prf','cooc',
                            'prf+bm25','cooc+bm25','prf+lsi',
                            'rrf','condorcet','combmnz','combsum'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--index-method', default='bsbi', choices=['bsbi','spimi'])
    p.add_argument('--output-dir', default='index')
    p.add_argument('--lsi-k', type=int, default=100)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--compare', action='store_true',
                   help='Compare results across every available method')
    p.add_argument('--difficulty', action='store_true',
                   help='Show query difficulty prediction before results')


# ─────────────────────────────────────────────────────────────
# eval
# ─────────────────────────────────────────────────────────────

def cmd_eval(args):
    from evaluation import load_qrels, eval_retrieval
    from ranked_fusion import RankFusion

    engine = _make_engine(args.index_method, args.encoding,
                          output_dir=args.output_dir)
    engine.load()

    lsi = None
    if any(m in args.methods
           for m in ('lsi','prf+lsi','rrf','condorcet','combmnz')):
        lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.lsi_k)

    fusion = RankFusion(engine, lsi=lsi)

    def _make_fn(mname):
        def fn(query, k=args.k):
            res, _, _ = _run_query(engine, query, mname, k, lsi, fusion)
            return res
        fn.__name__ = mname
        return fn

    print(f"\n{'='*60}\n  Evaluation   k={args.k}   encoding={args.encoding}")
    print(f"  Methods: {args.methods}\n{'='*60}")

    qrels = load_qrels()
    all_scores = {}
    for method in args.methods:
        scores = eval_retrieval(_make_fn(method), method, qrels, k=args.k)
        all_scores[method] = scores

    print(f"\n{'='*65}")
    print(f"  {'Method':<22} {'RBP':>8} {'DCG':>8} {'NDCG':>8} {'MAP':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m, s in all_scores.items():
        print(f"  {m:<22} {s['rbp']:>8.4f} {s['dcg']:>8.4f} "
              f"{s['ndcg']:>8.4f} {s['ap']:>8.4f}")
    print(f"{'='*65}")

    if len(all_scores) > 1:
        print()
        for metric in ('rbp', 'dcg', 'ndcg', 'ap'):
            best = max(all_scores, key=lambda m: all_scores[m][metric])
            print(f"  Best {metric.upper():<5}: {best} ({all_scores[best][metric]:.4f})")

    if args.overlap and len(args.methods) >= 2:
        q = "lipid metabolism in toxemia and normal pregnancy"
        plain = [m for m in args.methods
                 if m not in ('rrf','condorcet','combmnz','combsum')]
        if len(plain) >= 2:
            print()
            fusion.rank_overlap_analysis(q, k=args.k, methods=plain)


def _add_eval_args(p):
    p.add_argument('--methods', nargs='+',
                   default=['tfidf','bm25','wand','prf+bm25','rrf'],
                   choices=['tfidf','bm25','wand','lsi','prf+bm25','cooc+bm25',
                            'prf+lsi','rrf','condorcet','combmnz','combsum'])
    p.add_argument('--k', type=int, default=1000)
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--index-method', default='bsbi', choices=['bsbi','spimi'])
    p.add_argument('--output-dir', default='index')
    p.add_argument('--lsi-k', type=int, default=100)
    p.add_argument('--overlap', action='store_true',
                   help='Show Jaccard overlap between method result sets')


# ─────────────────────────────────────────────────────────────
# lsi
# ─────────────────────────────────────────────────────────────

def cmd_lsi(args):
    if args.lsi_action == 'build':
        lsi = _make_lsi(args.encoding, args.output_dir, args.n_components)
        t0 = time.perf_counter()
        lsi.build()
        lsi.save()
        elapsed = time.perf_counter() - t0
        import numpy as np
        evr = lsi.explained_variance_ratio()
        cumvar = np.cumsum(evr)
        print(f"\n  LSI model saved ({elapsed:.1f}s)")
        print(f"  Top-10 dims: {100*cumvar[min(9,len(cumvar)-1)]:.1f}% variance")
        print(f"  All {args.n_components} dims: {100*cumvar[-1]:.1f}% variance")

    elif args.lsi_action == 'query':
        if not args.query:
            print("[!] Provide --query TEXT")
            return
        lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.n_components)
        if lsi is None:
            print("[!] Run: python search_cli.py lsi build")
            return
        results = lsi.retrieve(args.query, k=args.k)
        print(f"\n  LSI results for: '{args.query}'")
        for i, (score, doc) in enumerate(results, 1):
            m = re.search(r'/([^/]+)\.txt$', doc)
            print(f"  {i:3}. [{score:.5f}]  {m.group(1) if m else doc}")

    elif args.lsi_action == 'related':
        if not args.term:
            print("[!] Provide --term WORD")
            return
        lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.n_components)
        if lsi is None:
            print("[!] Run: python search_cli.py lsi build")
            return
        related = lsi.most_related_terms(args.term, top_n=args.k)
        print(f"\n  Terms semantically related to '{args.term}':")
        for score, term in related:
            print(f"    {score:.4f}  {term}")

    elif args.lsi_action == 'info':
        lsi_path = os.path.join(args.output_dir, 'lsi_model.pkl')
        if not os.path.exists(lsi_path):
            print("[!] No LSI model found.")
            return
        lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.n_components)
        import numpy as np
        evr = lsi.explained_variance_ratio()
        cumvar = np.cumsum(evr)
        kb = os.path.getsize(lsi_path) / 1024
        print(f"\n  LSI Model Info")
        print(f"  {'─'*38}")
        print(f"  File size        : {kb:.1f} KB")
        print(f"  Components (k)   : {lsi.n_components}")
        print(f"  Doc vectors shape: {lsi.doc_vectors.shape}")
        print(f"  Variance explained: {100*cumvar[-1]:.1f}%")
        print(f"  Top-10 dims      : {100*cumvar[min(9,len(cumvar)-1)]:.1f}%")


def _add_lsi_args(p):
    p.add_argument('lsi_action', choices=['build','query','related','info'])
    p.add_argument('--n-components', type=int, default=100)
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--output-dir', default='index')
    p.add_argument('--query', default=None)
    p.add_argument('--term',  default=None)
    p.add_argument('--k', type=int, default=10)


# ─────────────────────────────────────────────────────────────
# inspect
# ─────────────────────────────────────────────────────────────

def cmd_inspect(args):
    from index_inspector import IndexInspector
    enc = _build_encoding_map()[args.encoding]
    inspector = IndexInspector(args.output_dir, enc)

    if args.difficulty:
        inspector.predict_query_difficulty(args.difficulty)
        return

    section = args.section
    sample_queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]
    if section == 'all':
        inspector.full_report(top_n=args.top_n, sample_queries=sample_queries)
    elif section == 'vocab':
        inspector.vocabulary_report(top_n=args.top_n)
    elif section == 'zipf':
        inspector.zipf_analysis()
    elif section == 'lengths':
        inspector.document_length_report()
    elif section == 'compression':
        inspector.compression_report()


def _add_inspect_args(p):
    p.add_argument('--section', default='all',
                   choices=['all','vocab','zipf','lengths','compression'])
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--output-dir', default='index')
    p.add_argument('--difficulty', type=str, default=None, metavar='QUERY',
                   help='Predict difficulty for a query')


# ─────────────────────────────────────────────────────────────
# bench
# ─────────────────────────────────────────────────────────────

def cmd_bench(args):
    if args.verify:
        from compression_benchmark import EliasGammaDeltaPostings, VBEEliasGammaTF
        from compression import VBEPostings, EliasGammaPostings, StandardPostings
        test_post = [1, 34, 67, 89, 454, 2345738]
        test_tf   = [1, 12, 10, 3, 4, 1]
        print("Verifying all codecs...")
        for name, C in [('Standard', StandardPostings), ('VBE', VBEPostings),
                        ('EliasGamma', EliasGammaPostings),
                        ('EG-Delta', EliasGammaDeltaPostings),
                        ('VBE+EG-TF', VBEEliasGammaTF)]:
            dp = C.decode(C.encode(test_post))
            dt = C.decode_tf(C.encode_tf(test_tf))
            op = '✓' if dp == test_post else f'✗ {dp}'
            ot = '✓' if dt == test_tf   else f'✗ {dt}'
            print(f"  {name:<15}  postings:{op}  tf:{ot}")
        return
    _run_bench_internal(args.output_dir)


def _add_bench_args(p):
    p.add_argument('--output-dir', default='index')
    p.add_argument('--verify', action='store_true',
                   help='Only verify correctness, skip timing')


# ─────────────────────────────────────────────────────────────
# repl
# ─────────────────────────────────────────────────────────────

def cmd_repl(args):
    engine = _make_engine(args.index_method, args.encoding,
                          output_dir=args.output_dir)
    engine.load()
    lsi = _load_lsi_if_exists(args.encoding, args.output_dir, args.lsi_k)

    from ranked_fusion import RankFusion
    from index_inspector import IndexInspector
    fusion    = RankFusion(engine, lsi=lsi)
    enc       = _build_encoding_map()[args.encoding]
    inspector = IndexInspector(args.output_dir, enc)

    method = args.method
    k      = args.k

    _repl_banner(lsi is not None,
                 hasattr(engine, 'prefix_search'))

    while True:
        try:
            raw = input(f"\n[{method}|k={k}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw in (':quit', ':q', 'quit', 'exit'):
            print("Goodbye!")
            break

        if raw in (':help', ':h'):
            _repl_banner(lsi is not None, hasattr(engine, 'prefix_search'))
            continue

        if raw.startswith(':method '):
            method = raw.split(None, 1)[1].strip()
            print(f"  → method: {method}")
            continue

        if raw.startswith(':k '):
            try:
                k = int(raw.split(None, 1)[1].strip())
                print(f"  → k: {k}")
            except ValueError:
                print("  [!] Invalid number.")
            continue

        if raw.startswith(':compare '):
            _compare_all_methods(engine, raw.split(None,1)[1].strip(), k, lsi)
            continue

        if raw.startswith(':difficulty '):
            inspector.predict_query_difficulty(raw.split(None,1)[1].strip())
            continue

        if raw.startswith(':prefix '):
            pfx = raw.split(None,1)[1].strip()
            if hasattr(engine, 'prefix_search'):
                print(f"  '{pfx}' → {engine.prefix_search(pfx)[:20]}")
            else:
                print("  [!] Prefix search requires SPIMI (--index-method spimi).")
            continue

        if raw.startswith(':related '):
            term = raw.split(None,1)[1].strip()
            if lsi:
                related = lsi.most_related_terms(term, top_n=10)
                print(f"  Neighbors of '{term}':")
                for s, t in related:
                    print(f"    {s:.4f}  {t}")
            else:
                print("  [!] LSI not loaded. Run: python search_cli.py lsi build")
            continue

        if raw.startswith(':inspect'):
            parts = raw.split()
            section = parts[1] if len(parts) > 1 else 'all'
            dispatch_inspect = {
                'all':         lambda: inspector.full_report(top_n=20),
                'vocab':       lambda: inspector.vocabulary_report(top_n=20),
                'zipf':        inspector.zipf_analysis,
                'lengths':     inspector.document_length_report,
                'compression': inspector.compression_report,
            }
            fn = dispatch_inspect.get(section)
            if fn:
                fn()
            else:
                print(f"  [!] Unknown section. Options: vocab zipf lengths compression")
            continue

        if raw == ':bench':
            _run_bench_internal(args.output_dir)
            continue

        if raw.startswith(':fusion '):
            q = raw.split(None,1)[1].strip()
            base = ['bm25', 'wand'] + (['lsi'] if lsi else [])
            fusion.compare(q, k=min(k, 5), methods=base)
            continue

        if raw.startswith(':overlap '):
            q = raw.split(None,1)[1].strip()
            base = ['tfidf','bm25','wand'] + (['lsi'] if lsi else [])
            fusion.rank_overlap_analysis(q, k=k, methods=base)
            continue

        # ── Regular query ──────────────────────────────────────
        t0 = time.perf_counter()
        results, expanded_q, extra = _run_query(engine, raw, method, k, lsi, fusion)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  ({ms:.0f} ms)")
        _print_results(results, method, k, raw, expanded_q, extra)


def _repl_banner(has_lsi, has_prefix):
    lsi_tag    = '✓' if has_lsi    else '✗ (run: lsi build)'
    prefix_tag = '✓' if has_prefix else '✗ (use --index-method spimi)'
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║         Search Engine — Interactive REPL             ║
  ╠══════════════════════════════════════════════════════╣
  ║  LSI loaded   : {lsi_tag:<36}║
  ║  Prefix search: {prefix_tag:<36}║
  ╠══════════════════════════════════════════════════════╣
  ║  <query>              search with current method     ║
  ║  :method <n>       bm25|tfidf|wand|lsi|prf+bm25  ║
  ║                       rrf|condorcet|combmnz|...      ║
  ║  :k <n>               change result count            ║
  ║  :compare  <query>    all methods side-by-side       ║
  ║  :difficulty <query>  IDF-based difficulty score     ║
  ║  :prefix <pfx>        Patricia prefix search         ║
  ║  :related <term>      LSI semantic neighbors         ║
  ║  :inspect [section]   vocab|zipf|lengths|compression ║
  ║  :bench               compression codec benchmark    ║
  ║  :fusion <query>      compare RRF/Condorcet/CombMNZ  ║
  ║  :overlap <query>     Jaccard overlap between ranks  ║
  ║  :help / :quit                                       ║
  ╚══════════════════════════════════════════════════════╝""")


def _add_repl_args(p):
    p.add_argument('--method', default='bm25',
                   choices=['tfidf','bm25','wand','lsi','prf+bm25','cooc+bm25',
                            'prf+lsi','rrf','condorcet','combmnz','combsum'])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--encoding', default='vbe', choices=ENCODING_CHOICES)
    p.add_argument('--index-method', default='bsbi', choices=['bsbi','spimi'])
    p.add_argument('--output-dir', default='index')
    p.add_argument('--lsi-k', type=int, default=100)


def main():
    parser = argparse.ArgumentParser(
        prog='search_cli.py',
        description='Search Engine from Scratch — unified CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick reference:
  python search_cli.py index
  python search_cli.py index --method spimi --encoding vbe-eg-tf
  python search_cli.py index --method spimi --encoding eg-delta --bench-after
  python search_cli.py search --query "protein synthesis"
  python search_cli.py search --query "lipid metabolism" --method rrf --compare
  python search_cli.py search --query "cell division" --difficulty --method prf+bm25
  python search_cli.py search --query "alkylated iodoacetate" --method lsi --verbose
  python search_cli.py eval
  python search_cli.py eval --methods bm25 wand rrf condorcet --overlap
  python search_cli.py eval --encoding vbe-eg-tf --methods bm25 wand
  python search_cli.py lsi build --n-components 150
  python search_cli.py lsi query --query "protein synthesis"
  python search_cli.py lsi related --term "protein"
  python search_cli.py lsi info
  python search_cli.py inspect
  python search_cli.py inspect --section zipf
  python search_cli.py inspect --difficulty "the of and"
  python search_cli.py bench
  python search_cli.py bench --verify
  python search_cli.py repl
  python search_cli.py repl --method rrf --k 15
  python search_cli.py repl --index-method spimi --encoding vbe-eg-tf
        """)

    sub = parser.add_subparsers(dest='subcommand', metavar='SUBCOMMAND')
    sub.required = True

    _add_index_args(sub.add_parser('index',
        help='Build the inverted index'))
    _add_search_args(sub.add_parser('search',
        help='Run a single query'))
    _add_eval_args(sub.add_parser('eval',
        help='Evaluate all retrieval methods with RBP/DCG/NDCG/MAP'))
    _add_lsi_args(sub.add_parser('lsi',
        help='Build, query, or inspect the LSI model'))
    _add_inspect_args(sub.add_parser('inspect',
        help='Index health analytics (Zipf, lengths, vocab, compression)'))
    _add_bench_args(sub.add_parser('bench',
        help='Benchmark compression codecs'))
    _add_repl_args(sub.add_parser('repl',
        help='Interactive search REPL'))

    args = parser.parse_args()
    {
        'index':   cmd_index,
        'search':  cmd_search,
        'eval':    cmd_eval,
        'lsi':     cmd_lsi,
        'inspect': cmd_inspect,
        'bench':   cmd_bench,
        'repl':    cmd_repl,
    }[args.subcommand](args)


if __name__ == '__main__':
    main()