import array


class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128)  # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128  # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Implementasi kompresi Elias-Gamma Encoding untuk postings list.

    Elias-Gamma encoding adalah algoritma kompresi bit-level yang bekerja
    sebagai berikut untuk sebuah integer positif n:
      1. Hitung k = floor(log2(n)), yaitu banyaknya bit yang dibutuhkan untuk
         merepresentasikan n dikurangi 1.
      2. Tulis k buah bit '0' sebagai unary prefix.
      3. Tulis representasi biner dari n dalam k+1 bit.

    Contoh:
      n=1  -> k=0, prefix='', biner='1'         -> '1'
      n=2  -> k=1, prefix='0', biner='10'        -> '010'
      n=3  -> k=1, prefix='0', biner='11'        -> '011'
      n=9  -> k=3, prefix='000', biner='1001'    -> '0001001'

    Seperti VBE, kita juga menggunakan gap-based encoding untuk postings list
    (tapi bukan untuk tf_list, karena TF tidak harus monoton naik).

    Bit-bitstring tersebut kemudian di-pack menjadi bytes, dengan padding '0'
    di akhir jika perlu, dan ukuran total bitstring disimpan di 4 byte pertama
    agar decoding tahu sampai mana membaca.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!
    """

    @staticmethod
    def _eg_encode_number(n):
        """
        Encode sebuah integer positif n menggunakan Elias-Gamma encoding.
        Mengembalikan string of bits (contoh: '0001001' untuk n=9).

        Parameters
        ----------
        n : int
            Integer positif yang akan di-encode (n >= 1)

        Returns
        -------
        str
            String of '0' dan '1' yang merepresentasikan Elias-Gamma code dari n
        """
        if n <= 0:
            raise ValueError(f"Elias-Gamma hanya support integer positif (n >= 1), dapat: {n}")

        k = n.bit_length() - 1

        unary = '0' * k

        binary = format(n, f'0{k + 1}b')
        return unary + binary

    @staticmethod
    def _eg_decode_number(bitstring, pos):
        """
        Decode satu Elias-Gamma code dari bitstring mulai dari posisi pos.

        Parameters
        ----------
        bitstring : str
            String of '0' dan '1'
        pos : int
            Posisi awal pembacaan

        Returns
        -------
        (int, int)
            Tuple (nilai integer yang di-decode, posisi setelah pembacaan selesai)
        """

        k = 0
        while pos < len(bitstring) and bitstring[pos] == '0':
            k += 1
            pos += 1

        binary_part = bitstring[pos:pos + k + 1]
        pos += k + 1
        n = int(binary_part, 2)
        return n, pos

    @staticmethod
    def _bits_to_bytes(bitstring):
        """
        Mengubah string of bits menjadi bytes.
        Menyimpan panjang bitstring di 4 byte pertama (big-endian unsigned int)
        agar decoding tahu persis berapa bit yang valid.

        Parameters
        ----------
        bitstring : str
            String of '0' dan '1'

        Returns
        -------
        bytes
        """
        bit_len = len(bitstring)
        pad = (8 - len(bitstring) % 8) % 8
        bitstring += '0' * pad
        byte_array = []
        for i in range(0, len(bitstring), 8):
            byte_array.append(int(bitstring[i:i + 8], 2))
        length_bytes = bit_len.to_bytes(4, byteorder='big')
        return length_bytes + bytes(byte_array)

    @staticmethod
    def _bytes_to_bits(data):
        """
        Mengubah bytes kembali ke string of bits, dengan panjang asli yang
        tersimpan di 4 byte pertama.

        Parameters
        ----------
        data : bytes

        Returns
        -------
        str
            String of '0' dan '1' dengan panjang yang tepat (tanpa padding)
        """
        bit_len = int.from_bytes(data[:4], byteorder='big')
        byte_data = data[4:]
        bitstring = ''.join(format(b, '08b') for b in byte_data)
        return bitstring[:bit_len]

    @staticmethod
    def _encode_list(list_of_numbers):
        """
        Encode list of integers menggunakan Elias-Gamma encoding.

        Parameters
        ----------
        list_of_numbers : List[int]
            List of positive integers

        Returns
        -------
        bytes
        """
        bitstring = ''.join(EliasGammaPostings._eg_encode_number(n) for n in list_of_numbers)
        return EliasGammaPostings._bits_to_bytes(bitstring)

    @staticmethod
    def _decode_list(data, count):
        """
        Decode bytes menjadi list of integers menggunakan Elias-Gamma decoding.

        Parameters
        ----------
        data : bytes
        count : int
            Berapa banyak integer yang harus di-decode

        Returns
        -------
        List[int]
        """
        bitstring = EliasGammaPostings._bytes_to_bits(data)
        pos = 0
        numbers = []
        for _ in range(count):
            n, pos = EliasGammaPostings._eg_decode_number(bitstring, pos)
            numbers.append(n)
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan Elias-Gamma
        Encoding. Postings list diubah dulu ke gap-based representation sebelum
        di-encode, agar nilainya lebih kecil dan kompresi lebih efektif.

        Karena Elias-Gamma hanya support n >= 1, kita simpan (gap + 1) untuk
        memastikan semua nilai yang di-encode adalah positif. Saat decode,
        kita kurangi kembali dengan 1.

        Parameters
        ----------
        postings_list : List[int]
            List of docIDs (harus sudah terurut menaik)

        Returns
        -------
        bytes
        """
        if not postings_list:
            return b''
        gaps = [postings_list[0] + 1]
        for i in range(1, len(postings_list)):
            gap = postings_list[i] - postings_list[i - 1]
            gaps.append(gap + 1)
        return EliasGammaPostings._encode_list(gaps)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list dari stream of bytes. Karena encoding menggunakan
        gap-based dengan offset +1, hasil decode perlu dikurangi 1 dan kemudian
        direkonstruksi ke nilai asli.

        Parameters
        ----------
        encoded_postings_list : bytes

        Returns
        -------
        List[int]
        """
        if not encoded_postings_list:
            return []
        bitstring = EliasGammaPostings._bytes_to_bits(encoded_postings_list)
        pos = 0
        gaps = []
        while pos < len(bitstring):
            if pos >= len(bitstring):
                break
            n, pos = EliasGammaPostings._eg_decode_number(bitstring, pos)
            gaps.append(n - 1)

        postings = [gaps[0]]
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menggunakan Elias-Gamma Encoding.
        TF tidak di-gap-encode karena TF tidak harus monoton naik.
        Ditambahkan offset +1 untuk memastikan semua nilai >= 1 (syarat Elias-Gamma).

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies (semua nilai >= 1)

        Returns
        -------
        bytes
        """
        if not tf_list:
            return b''
        # Tambahkan +1 offset agar tf=0 pun valid (walau seharusnya tf >= 1)
        return EliasGammaPostings._encode_list([tf + 1 for tf in tf_list])

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode list of term frequencies dari stream of bytes.
        Kurangi kembali offset +1 yang ditambahkan saat encode.

        Parameters
        ----------
        encoded_tf_list : bytes

        Returns
        -------
        List[int]
        """
        if not encoded_tf_list:
            return []
        bitstring = EliasGammaPostings._bytes_to_bits(encoded_tf_list)
        pos = 0
        tf_list = []
        while pos < len(bitstring):
            n, pos = EliasGammaPostings._eg_decode_number(bitstring, pos)
            tf_list.append(n - 1)
        return tf_list


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]

    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list    : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding TF tidak sama dengan TF original"
        print()

    # Uji Elias-Gamma dengan edge cases
    print("=== Uji Edge Cases Elias-Gamma ===")
    test_cases = [
        [1],
        [1, 2, 3],
        [1, 1, 1, 1],
        [100, 200, 300, 400, 500],
    ]
    for tc in test_cases:
        enc = EliasGammaPostings.encode(tc)
        dec = EliasGammaPostings.decode(enc)
        status = "OK" if dec == tc else f"GAGAL: expected {tc}, got {dec}"
        print(f"  postings {tc} -> {status}")

    tf_cases = [[1, 5, 3, 2, 10], [1], [7, 7, 7]]
    for tc in tf_cases:
        enc = EliasGammaPostings.encode_tf(tc)
        dec = EliasGammaPostings.decode_tf(enc)
        status = "OK" if dec == tc else f"GAGAL: expected {tc}, got {dec}"
        print(f"  tf     {tc} -> {status}")