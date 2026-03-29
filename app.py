"""
app.py
------
FastAPI web interface for the Search Engine from Scratch.

Run:
    uvicorn app:app --reload --port 8000

Or directly:
    python app.py
"""

import os
import re
import time
import math
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index and optional LSI model at startup."""
    print("Loading search engine...")
    try:
        from bsbi import BSBIIndex
        from compression import VBEPostings

        engine = BSBIIndex(
            data_dir="collection",
            output_dir="index",
            postings_encoding=VBEPostings,
        )
        engine.load()
        _state["engine"] = engine
        print(f"  Index loaded: {len(engine.doc_id_map)} docs, "
              f"{len(engine.term_id_map)} terms")
    except Exception as e:
        print(f"  [!] Could not load index: {e}")
        print("      Run: python search_cli.py index")
        _state["engine"] = None

    try:
        from lsi import LSIIndex
        from compression import VBEPostings
        lsi_path = os.path.join("index", "lsi_model.pkl")
        if os.path.exists(lsi_path):
            lsi = LSIIndex("collection", "index", VBEPostings)
            lsi.load()
            _state["lsi"] = lsi
            print(f"  LSI loaded: k={lsi.n_components} components")
        else:
            _state["lsi"] = None
            print("  LSI not found (run: python search_cli.py lsi build)")
    except Exception as e:
        _state["lsi"] = None
        print(f"  LSI load failed: {e}")

    _state["fusion"] = None

    yield
    print("Shutting down.")

app = FastAPI(
    title="Search Engine from Scratch",
    lifespan=lifespan,
)

def _get_fusion():
    if _state.get("fusion") is None and _state.get("engine"):
        from ranked_fusion import RankFusion
        _state["fusion"] = RankFusion(_state["engine"], lsi=_state.get("lsi"))
    return _state.get("fusion")


def _extract_doc_id(path: str) -> str:
    m = re.search(r"/([^/]+)\.txt$", path)
    return m.group(1) if m else path


def _run(query: str, method: str, k: int):
    """Dispatch query to correct backend. Returns (results, expanded_query, extra)."""
    engine = _state.get("engine")
    lsi    = _state.get("lsi")

    if not engine:
        raise HTTPException(503, "Index not loaded. Run: python search_cli.py index")

    extra = {}
    t0 = time.perf_counter()

    if method in ("rrf", "condorcet", "combmnz", "combsum"):
        fusion = _get_fusion()
        methods = ["bm25", "wand"] + (["lsi"] if lsi else [])
        results = fusion.retrieve(query, k=k, methods=methods, strategy=method)
        extra["fused"] = methods

    elif method == "tfidf":
        results = engine.retrieve_tfidf(query, k=k)

    elif method == "bm25":
        results = engine.retrieve_bm25(query, k=k)

    elif method == "wand":
        results = engine.retrieve_bm25_wand(query, k=k)

    elif method == "lsi":
        if not lsi:
            raise HTTPException(
                503,
                "LSI model not loaded. Run: python search_cli.py lsi build"
            )
        results = lsi.retrieve(query, k=k)

    elif "+" in method or method in ("prf", "cooc"):
        from query_expansion import QueryExpansionPipeline
        pipeline = QueryExpansionPipeline(engine, top_k_feedback=10, n_expand=5)
        m = method if "+" in method else method + "+bm25"
        out = pipeline.run(query, method=m, k=k, lsi=lsi)
        results = out["results"]
        extra["expanded_query"] = out["expanded_query"]
        extra["expansion_terms"] = out["expansion_terms"]

    else:
        results = engine.retrieve_bm25(query, k=k)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    extra["elapsed_ms"] = round(elapsed_ms, 1)
    return results, extra

class SearchRequest(BaseModel):
    query: str
    method: str = "bm25"
    k: int = 10


class DifficultyRequest(BaseModel):
    query: str


class RelatedRequest(BaseModel):
    term: str
    k: int = 10


class PrefixRequest(BaseModel):
    prefix: str

@app.get("/", response_class=HTMLResponse)
async def index_page():
    """Serve the main HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(404, "UI template not found")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def status():
    """Return engine status."""
    engine = _state.get("engine")
    lsi    = _state.get("lsi")
    spimi  = engine is not None and hasattr(engine, "prefix_search")
    return {
        "index_loaded":   engine is not None,
        "lsi_loaded":     lsi is not None,
        "spimi_loaded":   spimi,
        "num_docs":       len(engine.doc_id_map)  if engine else 0,
        "num_terms":      len(engine.term_id_map) if engine else 0,
        "lsi_components": lsi.n_components        if lsi    else None,
    }


@app.post("/api/search")
async def search(req: SearchRequest):
    """Run a search query."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    results, extra = _run(req.query.strip(), req.method, req.k)

    hits = []
    for rank, (score, path) in enumerate(results, 1):
        hits.append({
            "rank":   rank,
            "doc_id": _extract_doc_id(path),
            "path":   path,
            "score":  round(float(score), 5),
        })

    return {
        "query":           req.query,
        "method":          req.method,
        "k":               req.k,
        "num_results":     len(hits),
        "elapsed_ms":      extra.get("elapsed_ms"),
        "expanded_query":  extra.get("expanded_query"),
        "expansion_terms": extra.get("expansion_terms"),
        "fused_methods":   extra.get("fused"),
        "hits":            hits,
    }


@app.post("/api/compare")
async def compare(req: SearchRequest):
    """Run query against all available methods and return all results."""
    engine = _state.get("engine")
    lsi    = _state.get("lsi")
    if not engine:
        raise HTTPException(503, "Index not loaded")

    methods = ["tfidf", "bm25", "wand"]
    if lsi:
        methods += ["lsi", "prf+bm25"]
    methods += ["rrf"]

    fusion = _get_fusion()
    comparison = {}
    for m in methods:
        try:
            results, extra = _run(req.query, m, req.k)
            comparison[m] = {
                "hits": [
                    {"rank": i+1,
                     "doc_id": _extract_doc_id(p),
                     "score": round(float(s), 5)}
                    for i, (s, p) in enumerate(results)
                ],
                "elapsed_ms": extra.get("elapsed_ms"),
                "expanded_query": extra.get("expanded_query"),
            }
        except Exception as e:
            comparison[m] = {"error": str(e)}

    return {"query": req.query, "methods": comparison}


@app.post("/api/difficulty")
async def difficulty(req: DifficultyRequest):
    """Predict query difficulty."""
    engine = _state.get("engine")
    if not engine:
        raise HTTPException(503, "Index not loaded")
    try:
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector("index", VBEPostings)
        result = inspector.predict_query_difficulty(req.query)
        result.pop("term_data", None)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/related")
async def related_terms(req: RelatedRequest):
    """Find semantically related terms via LSI."""
    lsi = _state.get("lsi")
    if not lsi:
        raise HTTPException(503, "LSI model not loaded")
    related = lsi.most_related_terms(req.term, top_n=req.k)
    return {
        "term": req.term,
        "related": [{"score": round(s, 4), "term": t} for s, t in related],
    }


@app.post("/api/prefix")
async def prefix_search(req: PrefixRequest):
    """
    Prefix search. Uses Patricia Tree when SPIMI index is loaded.
    Falls back to linear vocabulary scan for BSBI index.
    """
    engine = _state.get("engine")
    if not engine:
        raise HTTPException(503, "Index not loaded")

    prefix = req.prefix.strip().lower()
    if not prefix:
        raise HTTPException(400, "Prefix cannot be empty")

    if hasattr(engine, "prefix_search"):
        matches = engine.prefix_search(prefix)
        return {"prefix": prefix, "matches": matches[:50], "method": "patricia"}

    try:
        all_terms = engine.term_id_map.id_to_str
        matches = [t for t in all_terms if t.startswith(prefix)]
        matches = sorted(matches)[:50]
        return {"prefix": prefix, "matches": matches, "method": "linear"}
    except Exception as e:
        raise HTTPException(500, f"Prefix search failed: {e}")


@app.get("/api/inspect/zipf")
async def inspect_zipf():
    """Return Zipf's law analysis."""
    try:
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector("index", VBEPostings)
        result = inspector.zipf_analysis()
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/inspect/vocab")
async def inspect_vocab():
    """Return vocabulary statistics."""
    engine = _state.get("engine")
    if not engine:
        raise HTTPException(503, "Index not loaded")
    try:
        from index_inspector import IndexInspector
        from compression import VBEPostings
        inspector = IndexInspector("index", VBEPostings)
        st = inspector._collect_stats()
        dfs = st["dfs"]
        N, V = st["N"], st["V"]
        hapax = sum(1 for d in dfs.values() if d == 1)
        df_vals = sorted(dfs.values(), reverse=True)
        return {
            "num_docs":     N,
            "vocab_size":   V,
            "hapax_ratio":  round(hapax / V, 4) if V else 0,
            "hapax_count":  hapax,
            "df_p50":       df_vals[len(df_vals) // 2] if df_vals else 0,
            "df_p90":       df_vals[len(df_vals) // 10] if len(df_vals) >= 10 else 0,
            "total_tokens": sum(st["sum_tfs"].values()),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)