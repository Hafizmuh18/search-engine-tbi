import re
import math

def rbp(ranking, p=0.8):
    """
    Menghitung search effectiveness metric score dengan
    Rank Biased Precision (RBP).

    Parameters
    ----------
    ranking : List[int]
        Vektor biner seperti [1, 0, 1, 1, 1, 0]
        Gold standard relevansi dari dokumen di rank 1, 2, 3, dst.

    p : float
        Persistence parameter RBP (default 0.8)

    Returns
    -------
    float
        Skor RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking, k=None):
    """
    Menghitung Discounted Cumulative Gain (DCG) dari sebuah ranking.

    Formula:
        DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

    dimana rel_i adalah relevansi dokumen di posisi ke-i (biner: 0 atau 1).

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi biner seperti [1, 0, 1, 1, 0]
        1 = relevan, 0 = tidak relevan

    k : int or None
        Kedalaman cutoff. Jika None, gunakan seluruh ranking.

    Returns
    -------
    float
        Skor DCG@k
    """
    if k is not None:
        ranking = ranking[:k]
    score = 0.0
    for i, rel in enumerate(ranking, start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg(ranking, ideal_ranking=None, k=None):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) dari sebuah ranking.

    NDCG menormalisasi DCG dengan ideal DCG (IDCG), yaitu DCG terbaik yang
    mungkin dicapai jika semua dokumen relevan diletakkan di posisi teratas.

    Formula:
        NDCG@k = DCG@k / IDCG@k

    dimana IDCG@k adalah DCG dari ideal ranking (semua dokumen relevan di atas).

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi biner dari hasil retrieval

    ideal_ranking : List[int] or None
        Ideal ranking. Jika None, akan dibuat otomatis dari ranking
        dengan semua 1 di depan dan 0 di belakang.

    k : int or None
        Kedalaman cutoff.

    Returns
    -------
    float
        Skor NDCG@k (antara 0.0 dan 1.0)
    """
    if k is not None:
        eval_ranking = ranking[:k]
    else:
        eval_ranking = ranking

    dcg_score = dcg(eval_ranking)

    if ideal_ranking is None:
        ideal_ranking = sorted(eval_ranking, reverse=True)
    else:
        if k is not None:
            ideal_ranking = ideal_ranking[:k]

    idcg_score = dcg(ideal_ranking)

    if idcg_score == 0.0:
        return 0.0
    return dcg_score / idcg_score


def average_precision(ranking):
    """
    Menghitung Average Precision (AP) dari sebuah ranking.

    Average Precision adalah rata-rata dari nilai Precision pada setiap
    posisi dimana dokumen yang relevan ditemukan.

    Formula:
        AP = (1 / R) * sum_{k=1}^{n} P(k) * rel(k)

    dimana:
        R     = total jumlah dokumen relevan yang ada (di ranking ini)
        P(k)  = Precision @ rank k = (jumlah dokumen relevan di rank 1..k) / k
        rel(k)= 1 jika dokumen di rank k relevan, 0 jika tidak

    Catatan: Jika tidak ada dokumen relevan di ranking, AP = 0.

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi biner dari hasil retrieval

    Returns
    -------
    float
        Skor Average Precision
    """
    num_relevant = sum(ranking)
    if num_relevant == 0:
        return 0.0

    score = 0.0
    relevant_so_far = 0
    for i, rel in enumerate(ranking, start=1):
        if rel == 1:
            relevant_so_far += 1
            precision_at_k = relevant_so_far / i
            score += precision_at_k

    return score / num_relevant


######## >>>>> Memuat qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """
    Memuat query relevance judgment (qrels) dalam format dictionary of dictionary.
    qrels[query_id][document_id] = 1 (relevan) atau 0 (tidak relevan)

    Parameters
    ----------
    qrel_file : str
    max_q_id  : int
    max_doc_id: int

    Returns
    -------
    dict
    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


def eval_retrieval(retrieval_fn, retrieval_name, qrels, query_file="queries.txt", k=1000):
    """
    Evaluasi sebuah fungsi retrieval dengan semua metrik yang tersedia:
    RBP, DCG, NDCG, dan AP.

    Loop ke semua query, hitung semua skor, lalu hitung mean score.

    Parameters
    ----------
    retrieval_fn : callable
        Fungsi retrieval yang menerima (query, k) dan mengembalikan List[(score, doc_path)]
    retrieval_name : str
        Nama metode retrieval untuk ditampilkan
    qrels : dict
        Query relevance judgments
    query_file : str
        File yang berisi daftar query
    k : int
        Jumlah top dokumen yang di-retrieve per query

    Returns
    -------
    dict
        Dictionary berisi mean score untuk setiap metrik
    """
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []

    with open(query_file) as file:
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            ranking = []
            for (score, doc) in retrieval_fn(query, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(average_precision(ranking))

    mean_rbp = sum(rbp_scores) / len(rbp_scores)
    mean_dcg = sum(dcg_scores) / len(dcg_scores)
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mean_ap = sum(ap_scores) / len(ap_scores)

    print(f"\n{'=' * 50}")
    print(f"Hasil evaluasi {retrieval_name} terhadap {len(rbp_scores)} queries")
    print(f"{'=' * 50}")
    print(f"  Mean RBP  (p=0.8) = {mean_rbp:.4f}")
    print(f"  Mean DCG         = {mean_dcg:.4f}")
    print(f"  Mean NDCG        = {mean_ndcg:.4f}")
    print(f"  Mean AP (MAP)    = {mean_ap:.4f}")
    print(f"{'=' * 50}")

    return {
        'rbp': mean_rbp,
        'dcg': mean_dcg,
        'ndcg': mean_ndcg,
        'ap': mean_ap,
    }


def eval(qrels, query_file="queries.txt", k=1000,
         engine=None, lsi=None):
    """
    Evaluasi semua metode retrieval dengan semua metrik.

    Parameters
    ----------
    qrels : dict
    query_file : str
    k : int
    engine : BSBIIndex / SPIMIIndex or None
        Jika None, default BSBIIndex + VBEPostings dibuat otomatis.
    lsi : LSIIndex or None
        Jika diberikan, LSI retrieval juga dievaluasi.
    """
    if engine is None:
        from bsbi import BSBIIndex
        from compression import VBEPostings
        engine = BSBIIndex(data_dir='collection',
                           postings_encoding=VBEPostings,
                           output_dir='index')
        engine.load()

    eval_retrieval(engine.retrieve_tfidf,    "TF-IDF",         qrels, query_file, k)
    eval_retrieval(engine.retrieve_bm25,     "BM25",           qrels, query_file, k)
    eval_retrieval(engine.retrieve_bm25_wand,"BM25 + WAND",    qrels, query_file, k)

    if lsi is not None:
        eval_retrieval(lsi.retrieve, f"LSI (k={lsi.n_components})",
                       qrels, query_file, k)

    try:
        from query_expansion import QueryExpansionPipeline
        pipeline = QueryExpansionPipeline(engine, top_k_feedback=10, n_expand=5)
        def _prf_bm25(query, k=k):
            return pipeline.run(query, method='prf+bm25', k=k)['results']
        eval_retrieval(_prf_bm25, "PRF + BM25", qrels, query_file, k)
    except Exception:
        pass

    try:
        from ranked_fusion import RankFusion
        fusion = RankFusion(engine, lsi=lsi)
        def _rrf(query, k=k):
            methods = ['bm25', 'wand'] + (['lsi'] if lsi else [])
            return fusion.retrieve(query, k=k, methods=methods, strategy='rrf')
        eval_retrieval(_rrf, "RRF (bm25+wand)", qrels, query_file, k)
    except Exception:
        pass


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    print("=== Unit Test Metrik Evaluasi ===")
    r = [1, 0, 1, 1, 0, 0]
    print(f"RBP  {r} = {rbp(r):.4f}")
    print(f"DCG  {r} = {dcg(r):.4f}")
    print(f"DCG  {r} @3 = {dcg(r, k=3):.4f}")
    print(f"NDCG {r} = {ndcg(r):.4f}")
    print(f"NDCG {r} @3 = {ndcg(r, k=3):.4f}")
    print(f"AP   {r} = {average_precision(r):.4f}")
    assert ndcg([1,1,1]) == 1.0
    assert average_precision([0,0,0]) == 0.0
    print("All metric unit tests passed.\n")

    print("=== Memulai Evaluasi Search Engine ===")
    eval(qrels)