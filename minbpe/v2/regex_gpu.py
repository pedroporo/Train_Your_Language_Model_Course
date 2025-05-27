"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from tqdm import tqdm
from .base import Tokenizer, get_stats, merge

from collections import defaultdict, Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import torch


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizerGPU(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        for i in tqdm(range(num_merges), total=num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(
                    self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items()
                       if k in allowed_special}
        else:
            raise ValueError(
                f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
    def train_optimized(self, text, vocab_size, verbose=False, device=None):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 1. Preprocesamiento
        text_chunks = self.compiled_pattern.findall(text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks if chunk]
        # Concatenamos todos los ids en un solo tensor (para vectorizar el conteo)
        ids = [torch.tensor(chunk, dtype=torch.int32) for chunk in ids_list]
        all_ids = torch.cat(ids)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        all_ids = all_ids.to(device)

        # 2. Inicialización
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        next_idx = 256

        for i in tqdm(range(num_merges), total=num_merges, disable=not verbose):
            # 3. Conteo de pares en GPU
            a = all_ids[:-1]
            b = all_ids[1:]
            pairs = torch.stack((a, b), dim=1)  # shape: [N-1, 2]
            # Convertimos los pares a un solo valor único para contar
            pairs_flat = pairs[:, 0] * 256 + pairs[:, 1]
            unique, counts = torch.unique(pairs_flat, return_counts=True)
            if len(unique) == 0:
                break
            # Encontrar el par más frecuente
            max_idx = torch.argmax(counts)
            most_common_flat = unique[max_idx].item()
            pair = (most_common_flat // 256, most_common_flat % 256)

            # 4. Reemplazo del par más frecuente (en CPU por simplicidad)
            # (el reemplazo eficiente en GPU es complejo por las longitudes variables)
            ids_cpu = all_ids.cpu().tolist()
            new_ids = []
            j = 0
            while j < len(ids_cpu) - 1:
                if (ids_cpu[j], ids_cpu[j+1]) == pair:
                    new_ids.append(next_idx)
                    j += 2
                else:
                    new_ids.append(ids_cpu[j])
                    j += 1
            if j == len(ids_cpu) - 1:
                new_ids.append(ids_cpu[-1])
            all_ids = torch.tensor(new_ids, dtype=torch.int32, device=device)

            # 5. Actualización de merges y vocab
            merges[pair] = next_idx
            vocab[next_idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {next_idx} ({vocab[next_idx]}) had {counts[max_idx].item()} occurrences")
            next_idx += 1

        self.merges = merges
        self.vocab = vocab
    
    def _count_pairs_optimized(self, ids):
        """Conteo optimizado de pares consecutivos."""
        stats = defaultdict(int)
        
        for chunk_ids in ids:
            if len(chunk_ids) < 2:
                continue
                
            # Usar numpy para operaciones vectorizadas cuando sea posible
            if isinstance(chunk_ids, np.ndarray):
                pairs = list(zip(chunk_ids[:-1], chunk_ids[1:]))
            else:
                pairs = list(zip(chunk_ids, chunk_ids[1:]))
                
            # Conteo eficiente usando Counter
            chunk_stats = Counter(pairs)
            for pair, count in chunk_stats.items():
                stats[pair] += count
                
        return dict(stats)
    
    def _count_pairs_parallel(self, ids):
        """Versión paralela del conteo de pares para datasets grandes."""
        def count_chunk(chunk_group):
            stats = defaultdict(int)
            for chunk_ids in chunk_group:
                if len(chunk_ids) < 2:
                    continue
                pairs = list(zip(chunk_ids[:-1], chunk_ids[1:]))
                chunk_stats = Counter(pairs)
                for pair, count in chunk_stats.items():
                    stats[pair] += count
            return dict(stats)
        
        # Dividir chunks en grupos para procesamiento paralelo (sin usar numpy)
        import multiprocessing as mp
        num_cores = mp.cpu_count()
        def split_list(lst, n):
            k, m = divmod(len(lst), n)
            return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
        chunk_groups = split_list(ids, num_cores)
        
        # Procesamiento paralelo
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(count_chunk, chunk_groups))
        
        # Combinar resultados
        final_stats = defaultdict(int)
        for result in results:
            for pair, count in result.items():
                final_stats[pair] += count
                
        return dict(final_stats)

    
    def _merge_pairs_optimized(self, ids, pair, idx):
        """Fusión optimizada de pares."""
        new_ids = []
        
        for chunk_ids in ids:
            if len(chunk_ids) < 2:
                new_ids.append(chunk_ids)
                continue
                
            # Conversión a lista si es numpy array para la fusión
            if isinstance(chunk_ids, np.ndarray):
                chunk_list = chunk_ids.tolist()
            else:
                chunk_list = list(chunk_ids)
                
            # Usar la función merge optimizada de base.py
            merged = merge(chunk_list, pair, idx)
            
            # Convertir de vuelta a numpy array para eficiencia de memoria
            new_ids.append(np.array(merged, dtype=np.uint16))
        
        return new_ids
