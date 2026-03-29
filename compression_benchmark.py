"""
compression_benchmark.py
-------------------------
Two additional hybrid compression codecs and an automated benchmarking
suite that compares all available codecs across multiple dimensions:
index size, encoding time, decoding time, and compression ratio.

New Codecs
----------

1. EliasGammaDeltaPostings
   A refinement of EliasGamma that applies Elias-Delta encoding instead of
   Elias-Gamma for larger integers (gaps > 256). The switch is adaptive:
   - Small gaps  (1–256) : Elias-Gamma   (shorter code for small numbers)
   - Large gaps  (> 256) : Elias-Delta   (more efficient for larger numbers)
   Elias-Delta encodes n as: Elias-Gamma(floor(log2(n))+1) followed by
   the lower bits of n (without the leading 1). This gives shorter codes
   than Elias-Gamma for large values while staying equally efficient for small ones.

2. VBEEliasGammaTF  (inverse hybrid of the naive approach)
   - DocID gaps : VBE      (byte-aligned, fast decode, good for varied gaps)
   - TF values  : Elias-Gamma (bit-level, optimal for small integers like TF)
   TF values in a real collection are typically small (1–5), where Elias-Gamma
   is maximally efficient. DocID gaps can be large (wide spreading), where
   VBE's byte alignment gives faster decode at reasonable compression.

Benchmark Dimensions
--------------------
- Encoded size (bytes) for both postings and TF lists
- Encoding throughput (millions of integers/sec)
- Decoding throughput (millions of integers/sec)
- Compression ratio vs StandardPostings (baseline)
- Real index size on disk (if index exists)
"""

import array
import time
import os
import struct
import math

# ─────────────────────────────────────────────────────────────
# Import existing codecs
# ─────────────────────────────────────────────────────────────
from compression import StandardPostings, VBEPostings, EliasGammaPostings


# ─────────────────────────────────────────────────────────────
# Elias-Delta primitive
# ─────────────────────────────────────────────────────────────

def _elias_delta_encode_number(n: int) -> str:
    """
    Elias-Delta encoding for a positive integer n >= 1.

    Let k = floor(log2(n)), so 2^k <= n < 2^(k+1).
    Code = Elias-Gamma(k+1)  +  binary(n)[1:]   (lower k bits of n, no leading 1)

    Examples:
        n=1  k=0  EG(1)='1'      lower=''        -> '1'
        n=2  k=1  EG(2)='010'    lower='0'       -> '0100'
        n=3  k=1  EG(2)='010'    lower='1'       -> '0101'
        n=4  k=2  EG(3)='011'    lower='00'      -> '01100'
        n=9  k=3  EG(4)='00100'  lower='001'     -> '00100001'
    """
    if n <= 0:
        raise ValueError(f"Elias-Delta requires n >= 1, got {n}")
    k = n.bit_length() - 1  # floor(log2(n))
    # Elias-Gamma encode (k+1)
    eg_part = _elias_gamma_encode_number(k + 1)
    # Lower k bits of n (strip the leading 1)
    lower_bits = format(n, f'0{k+1}b')[1:]   # k bits
    return eg_part + lower_bits


def _elias_gamma_encode_number(n: int) -> str:
    k = n.bit_length() - 1
    return '0' * k + format(n, f'0{k+1}b')


def _elias_delta_decode_number(bitstring: str, pos: int):
    """
    Decode one Elias-Delta number from bitstring at position pos.
    Returns (value, new_pos).
    """
    # First, Elias-Gamma decode to get (k+1)
    k_plus_1, pos = _elias_gamma_decode_number(bitstring, pos)
    k = k_plus_1 - 1
    # Read k more bits (lower bits of n)
    if k == 0:
        return 1, pos
    lower = bitstring[pos:pos + k]
    pos += k
    n = int('1' + lower, 2)   # restore leading 1
    return n, pos


def _elias_gamma_decode_number(bitstring: str, pos: int):
    k = 0
    while pos < len(bitstring) and bitstring[pos] == '0':
        k += 1
        pos += 1
    binary_part = bitstring[pos:pos + k + 1]
    pos += k + 1
    return int(binary_part, 2), pos


def _bits_to_bytes(bitstring: str) -> bytes:
    bit_len = len(bitstring)
    pad = (8 - len(bitstring) % 8) % 8
    bitstring += '0' * pad
    byte_array = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return bit_len.to_bytes(4, 'big') + bytes(byte_array)


def _bytes_to_bits(data: bytes) -> str:
    bit_len = int.from_bytes(data[:4], 'big')
    return ''.join(format(b, '08b') for b in data[4:])[:bit_len]

class EliasGammaDeltaPostings:
    """
    Adaptive bit-level codec that automatically switches between
    Elias-Gamma (small gaps) and Elias-Delta (large gaps) per integer.

    A 1-bit selector prefix is prepended to each encoded integer:
        '0' = Elias-Gamma follows
        '1' = Elias-Delta follows

    Crossover threshold: gaps <= THRESHOLD use Elias-Gamma,
    larger gaps use Elias-Delta (which is shorter for n > 2^(2^k-1)).

    Empirically, Elias-Delta is more compact than Elias-Gamma for n > 16.

    DocID gaps use adaptive switching; TF is always Elias-Gamma (TF < 16 typical).
    """

    THRESHOLD = 16   # gaps <= 16 use EG; larger gaps use ED

    @staticmethod
    def _encode_number_adaptive(n: int) -> str:
        """Encode n with selector bit + appropriate code."""
        if n <= EliasGammaDeltaPostings.THRESHOLD:
            return '0' + _elias_gamma_encode_number(n)
        else:
            return '1' + _elias_delta_encode_number(n)

    @staticmethod
    def _decode_number_adaptive(bitstring: str, pos: int):
        """Decode one adaptive-coded integer."""
        selector = bitstring[pos]
        pos += 1
        if selector == '0':
            return _elias_gamma_decode_number(bitstring, pos)
        else:
            return _elias_delta_decode_number(bitstring, pos)

    @staticmethod
    def _encode_list(lst: list, adaptive: bool = True) -> bytes:
        bits = ''.join(
            EliasGammaDeltaPostings._encode_number_adaptive(n) if adaptive
            else _elias_gamma_encode_number(n)
            for n in lst
        )
        return _bits_to_bytes(bits)

    @staticmethod
    def _decode_list(data: bytes, adaptive: bool = True) -> list:
        if not data:
            return []
        bitstring = _bytes_to_bits(data)
        pos = 0
        result = []
        while pos < len(bitstring):
            if adaptive:
                n, pos = EliasGammaDeltaPostings._decode_number_adaptive(bitstring, pos)
            else:
                n, pos = _elias_gamma_decode_number(bitstring, pos)
            result.append(n)
        return result

    @staticmethod
    def encode(postings_list: list) -> bytes:
        """Gap-encode postings then adaptive bit-level compress."""
        if not postings_list:
            return b''
        gaps = [postings_list[0] + 1]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1] + 1)
        return EliasGammaDeltaPostings._encode_list(gaps, adaptive=True)

    @staticmethod
    def decode(encoded: bytes) -> list:
        if not encoded:
            return []
        gaps_plus1 = EliasGammaDeltaPostings._decode_list(encoded, adaptive=True)
        postings = [gaps_plus1[0] - 1]
        for i in range(1, len(gaps_plus1)):
            postings.append(postings[-1] + gaps_plus1[i] - 1)
        return postings

    @staticmethod
    def encode_tf(tf_list: list) -> bytes:
        """TF always uses pure Elias-Gamma (TF values are typically small)."""
        if not tf_list:
            return b''
        return EliasGammaDeltaPostings._encode_list([tf + 1 for tf in tf_list], adaptive=False)

    @staticmethod
    def decode_tf(encoded: bytes) -> list:
        if not encoded:
            return []
        vals = EliasGammaDeltaPostings._decode_list(encoded, adaptive=False)
        return [v - 1 for v in vals]


class VBEEliasGammaTF:
    """
    Hybrid codec optimized for the asymmetry between DocID gaps and TF values:

    - DocID gaps → VBE (Variable-Byte Encoding)
      Byte-aligned, fast hardware decode, reasonable compression for varied gaps.

    - TF values  → Elias-Gamma (bit-level)
      Optimal for small integers (TF=1,2,3 are most common in practice).
      Elias-Gamma codes for TF:
          TF=1 -> '1'       (1 bit)
          TF=2 -> '010'     (3 bits)
          TF=3 -> '011'     (3 bits)
          TF=4 -> '00100'   (5 bits)
      vs VBE: every value uses at least 1 byte (8 bits).
      For a collection where 80% of TFs are 1, this halves the TF storage.

    This is the inverse of the approach in the reference (which uses EG for
    docIDs and VBE for TF). We argue this is MORE theoretically motivated:
    - DocID gaps have wide distribution (small AND large) → VBE is robust
    - TF values cluster near 1 → Elias-Gamma is optimal
    """

    # ── VBE helpers (same as VBEPostings) ──

    @staticmethod
    def _vb_encode(list_of_numbers: list) -> bytes:
        parts = []
        for n in list_of_numbers:
            buf = []
            while True:
                buf.insert(0, n % 128)
                if n < 128:
                    break
                n //= 128
            buf[-1] += 128
            parts.append(array.array('B', buf).tobytes())
        return b''.join(parts)

    @staticmethod
    def _vb_decode(data: bytes) -> list:
        n = 0
        result = []
        for byte in array.array('B', data):
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                result.append(n)
                n = 0
        return result

    @staticmethod
    def _eg_encode_tf(tf_list: list) -> bytes:
        bits = ''.join(_elias_gamma_encode_number(tf + 1) for tf in tf_list)
        return _bits_to_bytes(bits)

    @staticmethod
    def _eg_decode_tf(data: bytes) -> list:
        if not data:
            return []
        bitstring = _bytes_to_bits(data)
        pos = 0
        result = []
        while pos < len(bitstring):
            n, pos = _elias_gamma_decode_number(bitstring, pos)
            result.append(n - 1)
        return result


    @staticmethod
    def encode(postings_list: list) -> bytes:
        """Gap-encode then VBE compress docID gaps."""
        if not postings_list:
            return b''
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1])
        return VBEEliasGammaTF._vb_encode(gaps)

    @staticmethod
    def decode(encoded: bytes) -> list:
        if not encoded:
            return []
        gaps = VBEEliasGammaTF._vb_decode(encoded)
        postings = [gaps[0]]
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])
        return postings

    @staticmethod
    def encode_tf(tf_list: list) -> bytes:
        """Elias-Gamma compress TF values."""
        return VBEEliasGammaTF._eg_encode_tf(tf_list)

    @staticmethod
    def decode_tf(encoded: bytes) -> list:
        return VBEEliasGammaTF._eg_decode_tf(encoded)


# ─────────────────────────────────────────────────────────────
# Benchmarking suite
# ─────────────────────────────────────────────────────────────

def _generate_realistic_data(n_terms: int = 500, max_doc_id: int = 5000,
                              seed: int = 42):
    """
    Generate realistic postings and TF data that mimics a real IR collection.

    DocID gaps follow a roughly geometric distribution (sparse postings).
    TF values follow a Zipfian distribution (most TFs are 1 or 2).
    """
    import random
    random.seed(seed)

    all_postings = []
    all_tfs = []

    for _ in range(n_terms):
        # Sparse postings: geometric gaps
        df = random.randint(1, 200)
        gaps = [random.randint(1, 50) for _ in range(df)]
        postings = []
        curr = 0
        for g in gaps:
            curr += g
            postings.append(curr)
        if max(postings) > max_doc_id:
            postings = [min(p, max_doc_id) for p in postings]
            postings = sorted(set(postings))

        # Zipfian TF: 60% chance TF=1, 25% TF=2-3, rest higher
        tfs = []
        for _ in postings:
            r = random.random()
            if r < 0.60:
                tfs.append(1)
            elif r < 0.85:
                tfs.append(random.randint(2, 3))
            elif r < 0.95:
                tfs.append(random.randint(4, 10))
            else:
                tfs.append(random.randint(11, 50))

        all_postings.append(postings)
        all_tfs.append(tfs)

    return all_postings, all_tfs


def run_benchmark(output_dir: str = 'index', verbose: bool = True):
    """
    Benchmark all available codecs on both synthetic and (optionally) real index data.

    Parameters
    ----------
    output_dir : str
        Path to the real index directory. If valid index files exist,
        the benchmark also measures real on-disk index sizes.
    verbose : bool
        Print results table.

    Returns
    -------
    dict
        {codec_name: {metric: value}}
    """
    codecs = {
        'StandardPostings':     StandardPostings,
        'VBEPostings':          VBEPostings,
        'EliasGammaPostings':   EliasGammaPostings,
        'EliasGammaDelta':      EliasGammaDeltaPostings,
        'VBEEliasGammaTF':      VBEEliasGammaTF,
    }

    print("Generating realistic synthetic data...")
    all_postings, all_tfs = _generate_realistic_data(n_terms=1000)

    results = {}

    for name, codec in codecs.items():
        # Encoding
        encoded_postings = []
        encoded_tfs = []

        t0 = time.perf_counter()
        for postings, tfs in zip(all_postings, all_tfs):
            encoded_postings.append(codec.encode(postings))
            encoded_tfs.append(codec.encode_tf(tfs))
        encode_time = time.perf_counter() - t0

        total_postings_bytes = sum(len(e) for e in encoded_postings)
        total_tf_bytes = sum(len(e) for e in encoded_tfs)
        total_bytes = total_postings_bytes + total_tf_bytes

        # Decoding
        t0 = time.perf_counter()
        for i, (ep, et) in enumerate(zip(encoded_postings, encoded_tfs)):
            dp = codec.decode(ep)
            dt = codec.decode_tf(et)
        decode_time = time.perf_counter() - t0

        # Verify correctness on first 10 terms
        for i in range(min(10, len(all_postings))):
            assert codec.decode(codec.encode(all_postings[i])) == all_postings[i], \
                f"{name}: postings decode mismatch at term {i}"
            assert codec.decode_tf(codec.encode_tf(all_tfs[i])) == all_tfs[i], \
                f"{name}: TF decode mismatch at term {i}"

        total_ints = sum(len(p) + len(t) for p, t in zip(all_postings, all_tfs))

        results[name] = {
            'total_bytes':          total_bytes,
            'postings_bytes':       total_postings_bytes,
            'tf_bytes':             total_tf_bytes,
            'encode_time_s':        encode_time,
            'decode_time_s':        decode_time,
            'encode_mints':         total_ints / 1e6 / encode_time,
            'decode_mints':         total_ints / 1e6 / decode_time,
        }

    # Compression ratio vs Standard
    std_bytes = results['StandardPostings']['total_bytes']
    for name in results:
        results[name]['compression_ratio'] = std_bytes / results[name]['total_bytes']

    # Real index sizes (if available)
    main_index = os.path.join(output_dir, 'main_index.index')
    if os.path.exists(main_index):
        # We can't easily re-encode the real index per codec without re-indexing,
        # so we report the actual on-disk size for the codec used during indexing.
        real_kb = os.path.getsize(main_index) / 1024
        if verbose:
            print(f"\nReal index file (main_index.index): {real_kb:.1f} KB")

    if verbose:
        _print_benchmark_table(results)

    return results


def _print_benchmark_table(results: dict):
    """Print a formatted comparison table."""
    print()
    header = (f"{'Codec':<26} {'Total(KB)':>10} {'Post(KB)':>9} "
              f"{'TF(KB)':>8} {'Ratio':>7} {'Enc(ms)':>9} {'Dec(ms)':>9}")
    print(header)
    print("-" * len(header))

    # Sort by total bytes ascending
    for name, r in sorted(results.items(), key=lambda x: x[1]['total_bytes']):
        print(
            f"{name:<26} "
            f"{r['total_bytes']/1024:>10.2f} "
            f"{r['postings_bytes']/1024:>9.2f} "
            f"{r['tf_bytes']/1024:>8.2f} "
            f"{r['compression_ratio']:>7.3f}x "
            f"{r['encode_time_s']*1000:>9.1f} "
            f"{r['decode_time_s']*1000:>9.1f} "
        )

    print()
    best_size = min(results, key=lambda x: results[x]['total_bytes'])
    best_enc  = max(results, key=lambda x: results[x]['encode_mints'])
    best_dec  = max(results, key=lambda x: results[x]['decode_mints'])
    print(f"Best compression : {best_size}")
    print(f"Fastest encoding : {best_enc}  ({results[best_enc]['encode_mints']:.1f}M int/s)")
    print(f"Fastest decoding : {best_dec}  ({results[best_dec]['decode_mints']:.1f}M int/s)")


def tf_distribution_analysis(all_tfs):
    """Show how TF distribution affects codec efficiency."""
    print("\n=== TF Distribution Analysis ===")
    flat_tfs = [tf for tfs in all_tfs for tf in tfs]
    total = len(flat_tfs)
    from collections import Counter
    counts = Counter(flat_tfs)
    print(f"Total TF values: {total}")
    print(f"{'TF':>4}  {'Count':>7}  {'%':>6}  {'EG bits':>8}  {'VBE bytes':>10}")
    for tf in sorted(counts.keys())[:15]:
        k = (tf + 1).bit_length() - 1
        eg_bits = 2 * k + 1
        vbe_bytes = len(VBEPostings.vb_encode_number(tf))
        print(f"{tf:>4}  {counts[tf]:>7}  {100*counts[tf]/total:>5.1f}%  "
              f"{eg_bits:>8}  {8*vbe_bytes:>10} bits")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compression codec benchmark')
    parser.add_argument('--output-dir', default='index',
                        help='Index directory (for real index size measurement)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify codec correctness, skip benchmark')
    args = parser.parse_args()

    print("=" * 60)
    print("  Compression Codec Benchmark")
    print("=" * 60)

    if args.verify_only:
        print("\nVerifying all codecs on edge cases...")
        test_cases = [
            [1], [1, 2, 3], [1, 100, 200, 300],
            [34, 67, 89, 454, 2345738],
            list(range(1, 101)),
        ]
        tf_cases = [[1], [1, 2, 3], [1, 5, 10, 50], [12, 10, 3, 4, 1]]
        for Codec in [EliasGammaDeltaPostings, VBEEliasGammaTF]:
            print(f"\n  {Codec.__name__}:")
            for tc in test_cases:
                dec = Codec.decode(Codec.encode(tc))
                status = "✓" if dec == tc else f"✗ expected {tc[:3]}... got {dec[:3]}..."
                print(f"    postings {tc[:4]}... -> {status}")
            for tc in tf_cases:
                dec = Codec.decode_tf(Codec.encode_tf(tc))
                status = "✓" if dec == tc else f"✗ expected {tc} got {dec}"
                print(f"    tf      {tc} -> {status}")
    else:
        results = run_benchmark(output_dir=args.output_dir)

        _, all_tfs = _generate_realistic_data(n_terms=1000)
        tf_distribution_analysis(all_tfs)

        print("\nUsage in bsbi.py:")
        print("  from compression_benchmark import EliasGammaDeltaPostings, VBEEliasGammaTF")
        print("  BSBI_instance = BSBIIndex(..., postings_encoding=VBEEliasGammaTF)")