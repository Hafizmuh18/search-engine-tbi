import pickle
import os


class InvertedIndex:
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file; dan juga menyediakan
    mekanisme untuk menulis Inverted Index ke file (storage) saat melakukan indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict adalah konsep "Dictionary" yang merupakan bagian dari
        Inverted Index. postings_dict ini diasumsikan dapat dimuat semuanya
        di memori.

        Seperti namanya, "Dictionary" diimplementasikan sebagai python's Dictionary
        yang memetakan term ID (integer) ke 4-tuple:
           1. start_position_in_index_file : (dalam satuan bytes) posisi dimana
              postings yang bersesuaian berada di file (storage). Kita bisa
              menggunakan operasi "seek" untuk mencapainya.
           2. number_of_postings_in_list : berapa banyak docID yang ada pada
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : panjang postings list dalam
              satuan byte.
           4. length_in_bytes_of_tf_list : panjang list of term frequencies dari
              postings list terkait dalam satuan byte

    terms: List[int]
        List of terms IDs, untuk mengingat urutan terms yang dimasukan ke
        dalam Inverted Index.

    doc_length: dict
        key: doc ID (int), value: document length (number of tokens).
        Berguna untuk normalisasi Score terhadap panjang dokumen saat
        menghitung score dengan TF-IDF atau BM25.

    term_max_tf: dict
        key: term ID (int), value: max TF across all documents for that term.
        Digunakan untuk WAND Top-K retrieval agar bisa menghitung upper bound
        BM25 score per term tanpa membuka semua postings list.
    """

    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Nama yang digunakan untuk menyimpan files yang berisi index
        postings_encoding : Lihat di compression.py, kandidatnya adalah StandardPostings,
                        VBEPostings, EliasGammaPostings, dsb.
        directory (str): directory dimana file index berada
        """

        self.index_file_path = os.path.join(directory, index_name + '.index')
        self.metadata_file_path = os.path.join(directory, index_name + '.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Untuk keep track urutan term yang dimasukkan ke index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                                # Berguna untuk normalisasi Score terhadap panjang dokumen
                                # saat menghitung score dengan TF-IDF atau BM25

        self.term_max_tf = {}   # key: term ID (int), value: max TF di antara semua dokumen
                                # Digunakan sebagai upper bound untuk WAND Top-K retrieval

    def __enter__(self):
        """
        Memuat semua metadata ketika memasuki context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator untuk List yang berisi urutan term yang masuk ke
                index saat konstruksi. ---> term_iter
            3. doc_length: dict {doc_id: doc_length}
            4. term_max_tf: dict {term_id: max_tf}  <-- BARU untuk WAND
        """
        self.index_file = open(self.index_file_path, 'rb+')

        with open(self.metadata_file_path, 'rb') as f:
            loaded = pickle.load(f)
            # Support format lama (3 elemen) dan baru (4 elemen)
            if len(loaded) == 4:
                self.postings_dict, self.terms, self.doc_length, self.term_max_tf = loaded
            else:
                self.postings_dict, self.terms, self.doc_length = loaded
                self.term_max_tf = {}
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan metadata ketika keluar context"""
        self.index_file.close()

        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.term_max_tf], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __iter__(self):
        return self

    def reset(self):
        """
        Kembalikan file pointer ke awal, dan kembalikan pointer iterator
        term ke awal
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()

    def __next__(self):
        """
        Ketika instance dari kelas InvertedIndexReader ini digunakan
        sebagai iterator pada sebuah loop scheme, special method __next__(...)
        bertugas untuk mengembalikan pasangan (term, postings_list, tf_list) berikutnya
        pada inverted index.
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Kembalikan sebuah postings list (list of docIDs) beserta list
        of term frequencies terkait untuk sebuah term.

        Method ini langsung loncat ke posisi byte tertentu pada file
        dimana postings list dari term disimpan.
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya menulis secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Menambahkan (append) sebuah term, postings_list, dan juga TF list
        yang terasosiasi ke posisi akhir index file.

        Selain menyimpan metadata standar, method ini juga menyimpan
        max_tf untuk setiap term ke self.term_max_tf, yang dibutuhkan
        oleh algoritma WAND Top-K retrieval.

        Parameters
        ----------
        term:
            term atau termID yang merupakan unique identifier dari sebuah term
        postings_list: List[Int]
            List of docIDs dimana term muncul
        tf_list: List[Int]
            List of term frequencies
        """
        self.terms.append(term)

        # Update doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        # Update term_max_tf: simpan max TF untuk term ini
        # Digunakan sebagai upper bound BM25 score per term pada WAND
        if tf_list:
            self.term_max_tf[term] = max(tf_list)

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list),
                                    len(compressed_postings), len(compressed_tf_list))


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1, 2], "terms salah"
        assert index.doc_length == {2: 2, 3: 38, 4: 25, 5: 56, 8: 3, 10: 30}, "doc_length salah"
        assert index.term_max_tf == {1: 30, 2: 56}, "term_max_tf salah"
        assert index.postings_dict == {1: (0,
                                           5,
                                           len(VBEPostings.encode([2, 3, 4, 8, 10])),
                                           len(VBEPostings.encode_tf([2, 4, 2, 3, 30]))),
                                       2: (len(VBEPostings.encode([2, 3, 4, 8, 10])) + len(VBEPostings.encode_tf([2, 4, 2, 3, 30])),
                                           3,
                                           len(VBEPostings.encode([3, 4, 5])),
                                           len(VBEPostings.encode_tf([34, 23, 56])))}, \
            "postings dictionary salah"

        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3, 4, 5])))) == [3, 4, 5], "terdapat kesalahan"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34, 23, 56])))) == [34, 23, 56], "terdapat kesalahan"

    print("Semua assertion index.py passed!")