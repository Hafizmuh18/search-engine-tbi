from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("=" * 60)
    print("Query  : ", query)

    print("\n--- TF-IDF ---")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
        print(f"  {doc:50} {score:>.3f}")

    print("\n--- BM25 (k1=1.2, b=0.75) ---")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"  {doc:50} {score:>.3f}")

    print("\n--- BM25 + WAND Top-K ---")
    for (score, doc) in BSBI_instance.retrieve_bm25_wand(query, k=10):
        print(f"  {doc:50} {score:>.3f}")

    print()