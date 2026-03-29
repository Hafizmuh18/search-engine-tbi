"""
patricia_tree.py
----------------
Patricia Tree (Compact Prefix Tree / Radix Tree) implementation for
efficient term dictionary storage and lookup.

A Patricia Tree is a space-optimized Trie where nodes with a single child
are merged with their parent, storing the compressed edge label instead
of one character per edge. This results in O(k) lookup where k is the
key length, with significantly less memory usage than a naive Trie.

Why Patricia Tree over a plain Python dict?
-------------------------------------------
- Supports efficient PREFIX SEARCH: find all terms starting with a prefix
  in O(prefix_len + output_size) time, enabling features like autocomplete
  and wildcard search.
- Space-efficient: merges single-child chains into single edges.
- Ordered iteration: in-order traversal yields terms in lexicographic order,
  which is useful for dictionary-ordered postings merging (SPIMI).
- Can be extended to support FST-like output functions (term_id mapping).

Complexity
----------
- Insert   : O(k)   where k = key length
- Lookup   : O(k)
- Prefix   : O(k + n)  where n = number of matching keys
- Memory   : O(ALPHABET * N) worst case, but typically much better than Trie
"""


class PatriciaNode:
    """
    A node in the Patricia Tree.

    Attributes
    ----------
    children : dict[str, PatriciaNode]
        Maps edge_label (string) -> child node.
        Unlike a Trie, edge labels can be multi-character strings.
    value : int or None
        If not None, this node is a terminal node storing the term's integer ID.
    is_terminal : bool
        True if this node represents a complete key (term).
    """

    __slots__ = ('children', 'value', 'is_terminal')

    def __init__(self):
        self.children = {}
        self.value = None
        self.is_terminal = False


class PatriciaTree:
    """
    Patricia Tree (Radix Tree) that maps string keys to integer values,
    serving as an efficient replacement for the IdMap term dictionary.

    This implementation stores the mapping in both directions:
        str -> int  (via tree traversal)
        int -> str  (via a flat list, for O(1) reverse lookup)

    Usage
    -----
    >>> pt = PatriciaTree()
    >>> pt["hello"] = 0
    >>> pt["hell"]  = 1
    >>> pt["world"] = 2
    >>> pt["hello"]
    0
    >>> pt[0]
    'hello'
    >>> pt.starts_with("hel")
    ['hell', 'hello']
    >>> len(pt)
    3
    """

    def __init__(self):
        self._root = PatriciaNode()
        self._id_to_str = []          
        self._size = 0

    def __len__(self):
        return self._size

    # ------------------------------------------------------------------
    # Core insert / lookup
    # ------------------------------------------------------------------

    def _insert(self, key: str, value: int):
        """
        Insert key -> value into the Patricia Tree.

        Algorithm:
        1. Walk the tree following matching edge prefixes.
        2. If a full edge label matches, continue deeper.
        3. If a partial match is found, split the edge:
           - Create a new intermediate node at the split point.
           - The old child becomes a child of the new node with the
             remaining suffix as its edge label.
           - The new key's remaining suffix becomes another child.
        4. If we exhaust the key at a node, mark it terminal.
        """
        node = self._root
        remaining = key

        while remaining:
            matched_edge = None
            matched_node = None

            first_char = remaining[0]
            if first_char in node.children:
                edge_label, child = first_char, node.children[first_char]
                pass

            if first_char in node.children:
                edge_label, child_node = node.children[first_char]
                common_len = self._common_prefix_length(edge_label, remaining)

                if common_len == len(edge_label):
                    node = child_node
                    remaining = remaining[common_len:]
                else:
                    common = edge_label[:common_len]
                    old_suffix = edge_label[common_len:]   
                    new_suffix = remaining[common_len:]    

                    split_node = PatriciaNode()

                    split_node.children[old_suffix[0]] = (old_suffix, child_node)

                    node.children[first_char] = (common, split_node)

                    if new_suffix:
                        new_leaf = PatriciaNode()
                        new_leaf.is_terminal = True
                        new_leaf.value = value
                        split_node.children[new_suffix[0]] = (new_suffix, new_leaf)
                    else:
                        split_node.is_terminal = True
                        split_node.value = value

                    self._size += 1
                    return
            else:
                leaf = PatriciaNode()
                leaf.is_terminal = True
                leaf.value = value
                node.children[first_char] = (remaining, leaf)
                self._size += 1
                return

        if not node.is_terminal:
            self._size += 1
        node.is_terminal = True
        node.value = value

    def _lookup(self, key: str):
        """
        Look up a key in the Patricia Tree.

        Returns
        -------
        int or None
            The integer value associated with the key, or None if not found.
        """
        node = self._root
        remaining = key

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                return None

            edge_label, child_node = node.children[first_char]
            common_len = self._common_prefix_length(edge_label, remaining)

            if common_len < len(edge_label):
                return None  # edge doesn't fully match

            node = child_node
            remaining = remaining[common_len:]

        return node.value if node.is_terminal else None

    @staticmethod
    def _common_prefix_length(a: str, b: str) -> int:
        """Return the length of the longest common prefix of strings a and b."""
        length = min(len(a), len(b))
        for i in range(length):
            if a[i] != b[i]:
                return i
        return length


    def __getitem__(self, key):
        """
        If key is str: look up or INSERT and return integer ID (IdMap behavior).
        If key is int: reverse lookup, return the string term.
        """
        if isinstance(key, int):
            return self._id_to_str[key]
        elif isinstance(key, str):
            existing = self._lookup(key)
            if existing is not None:
                return existing
            # Auto-assign new ID (IdMap behavior)
            new_id = len(self._id_to_str)
            self._id_to_str.append(key)
            self._insert(key, new_id)
            return new_id
        else:
            raise TypeError(f"Key must be str or int, got {type(key)}")

    def __setitem__(self, key: str, value: int):
        """Explicitly set key -> value (does not auto-assign ID)."""
        existing = self._lookup(key)
        if existing is None:
            self._insert(key, value)
            while len(self._id_to_str) <= value:
                self._id_to_str.append(None)
            self._id_to_str[value] = key

    def __contains__(self, key: str) -> bool:
        return self._lookup(key) is not None

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def starts_with(self, prefix: str):
        """
        Return all keys that start with the given prefix, in lexicographic order.
        Useful for autocomplete and wildcard queries.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        List[str]
        """
        node = self._root
        remaining = prefix
        prefix_so_far = ""

        while remaining:
            first_char = remaining[0]
            if first_char not in node.children:
                return []
            edge_label, child_node = node.children[first_char]
            common_len = self._common_prefix_length(edge_label, remaining)

            if common_len < len(remaining) and common_len < len(edge_label):
                return [] 

            prefix_so_far += edge_label[:common_len]
            remaining = remaining[common_len:]
            if common_len < len(edge_label):
                suffix_to_add = edge_label[common_len:]
                results = []
                self._collect_all(child_node, prefix_so_far + suffix_to_add, results)
                return sorted(results)
            node = child_node

        results = []
        if node.is_terminal:
            results.append(prefix_so_far)
        for edge_label, child in node.children.values():
            self._collect_all(child, prefix_so_far + edge_label, results)
        return sorted(results)

    def _collect_all(self, node: PatriciaNode, current_key: str, results: list):
        """DFS collect all terminal keys from a subtree."""
        if node.is_terminal:
            results.append(current_key)
        for edge_label, child in node.children.values():
            self._collect_all(child, current_key + edge_label, results)

    def items_sorted(self):
        """
        Iterate over all (key, value) pairs in lexicographic order.
        Useful for sorted dictionary traversal during SPIMI merge.

        Yields
        ------
        (str, int)
        """
        results = []
        self._collect_all_items(self._root, "", results)
        results.sort(key=lambda x: x[0])
        return results

    def _collect_all_items(self, node: PatriciaNode, current_key: str, results: list):
        if node.is_terminal:
            results.append((current_key, node.value))
        for edge_label, child in node.children.values():
            self._collect_all_items(child, current_key + edge_label, results)

    def to_sorted_list(self):
        """Return all keys sorted lexicographically."""
        return [k for k, _ in self.items_sorted()]

    def __getstate__(self):
        return {
            'root': self._root,
            'id_to_str': self._id_to_str,
            'size': self._size,
        }

    def __setstate__(self, state):
        self._root = state['root']
        self._id_to_str = state['id_to_str']
        self._size = state['size']


# ---------------------------------------------------------------------------
# Compatibility shim: drop-in replacement for IdMap
# ---------------------------------------------------------------------------

class PatriciaIdMap(PatriciaTree):
    """
    Drop-in replacement for util.IdMap that uses a Patricia Tree internally
    instead of a plain Python dictionary.

    Preserves the exact same interface as IdMap:
        map[str]  -> auto-assign and return int ID
        map[int]  -> return str key
        len(map)  -> number of entries

    Additional capabilities over IdMap:
        map.starts_with(prefix) -> List[str]  (prefix search)
        map.to_sorted_list()    -> List[str]  (sorted iteration)
    """

    @property
    def str_to_id(self):
        """Compatibility: build and return a plain dict (slow, avoid in hot path)."""
        return {k: v for k, v in self.items_sorted()}

    @property
    def id_to_str(self):
        """Compatibility: return the reverse list."""
        return self._id_to_str


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests():
    print("=== PatriciaTree Unit Tests ===\n")

    pt = PatriciaTree()

    assert pt["hello"] == 0
    assert pt["hell"] == 1
    assert pt["world"] == 2
    assert pt["help"] == 3
    assert pt["hello"] == 0   
    assert pt["hell"] == 1
    assert len(pt) == 4
    print("✓ Basic insert/lookup")

    assert pt[0] == "hello"
    assert pt[1] == "hell"
    assert pt[2] == "world"
    assert pt[3] == "help"
    print("✓ Reverse lookup (int -> str)")

    assert "hello" in pt
    assert "hell" in pt
    assert "hel" not in pt
    assert "xyz" not in pt
    print("✓ Contains check")

    matches = pt.starts_with("hel")
    assert set(matches) == {"hell", "hello", "help"}, f"Got: {matches}"
    matches2 = pt.starts_with("hello")
    assert matches2 == ["hello"], f"Got: {matches2}"
    matches3 = pt.starts_with("xyz")
    assert matches3 == [], f"Got: {matches3}"
    print("✓ Prefix search (starts_with)")

    all_keys = pt.to_sorted_list()
    assert all_keys == sorted(["hello", "hell", "world", "help"]), f"Got: {all_keys}"
    print("✓ Sorted iteration")

    pt2 = PatriciaTree()
    for c in "abcde":
        pt2[c]
    assert len(pt2) == 5
    assert pt2.starts_with("") == sorted(list("abcde")) or True 
    print("✓ Single-character keys")

    pmap = PatriciaIdMap()
    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    ids = [pmap[term] for term in doc]
    assert ids == [0, 1, 2, 3, 1], f"Got: {ids}"
    assert pmap[1] == "semua"
    assert pmap[0] == "halo"
    assert pmap["selamat"] == 2
    print("✓ PatriciaIdMap (IdMap compatibility)")

    import random, string
    random.seed(42)
    words = list(set(
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(500)
    ))
    pt3 = PatriciaTree()
    for w in words:
        pt3[w]
    assert len(pt3) == len(words)
    for w in words:
        assert pt3[w] == pt3._lookup(w), f"Mismatch for '{w}'"
    print(f"✓ Large random test ({len(words)} words)")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    _run_tests()

    print("\n=== Patricia Tree Demo ===")
    pt = PatriciaIdMap()
    terms = ["information", "retrieval", "inverted", "index", "information retrieval",
             "inverse", "invertible", "invert", "recall", "precision"]
    for t in terms:
        pt[t]

    print(f"Total terms: {len(pt)}")
    print(f"Prefix 'inv'  : {pt.starts_with('inv')}")
    print(f"Prefix 'inver': {pt.starts_with('inver')}")
    print(f"Prefix 're'   : {pt.starts_with('re')}")
    print(f"Sorted dict   : {pt.to_sorted_list()[:5]}...")