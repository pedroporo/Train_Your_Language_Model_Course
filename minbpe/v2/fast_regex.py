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


class FastRegexTokenizer(Tokenizer):

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

        # 1. Divide el texto en chunks usando el regex
        text_chunks = self.compiled_pattern.findall(text)
        # 2. Convierte todos los chunks a bytes y los concatena en un array plano
        ids = []
        for chunk in text_chunks:
            if chunk:  # evitar chunks vacíos
                ids.extend(chunk.encode("utf-8"))  # añade los bytes del chunk
        # Ahora ids es una lista plana de ints (0..255)

        # 3. Inicializa merges y vocab
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges, disable=not verbose):
            # 4. Cuenta pares consecutivos
            stats = Counter(zip(ids, ids[1:]))
            if not stats:
                break
            pair = max(stats, key=stats.get)

            # 5. Crea nuevo token y haz el merge
            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 6. Reemplaza todos los pares por el nuevo token
            new_ids = []
            j = 0
            while j < len(ids) - 1:
                if (ids[j], ids[j+1]) == pair:
                    new_ids.append(idx)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            if j == len(ids) - 1:
                new_ids.append(ids[-1])
            ids = new_ids

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        Decodifica una lista de ids a texto. Soporta tokens especiales.
        """
        out = []
        for idx in ids:
            if idx in self.special_tokens_inv:
                out.append(self.special_tokens_inv[idx])
            else:
                out.append(self.vocab[idx].decode("utf-8", errors="replace"))
        return "".join(out)

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

    def encode(self, text, allowed_special=None):
        """
        Tokeniza el texto, respetando los tokens especiales definidos en allowed_special.
        allowed_special: conjunto de strings (por ejemplo: {"<|endoftext|>", "<pad>"})
        """
        if allowed_special is None:
            allowed_special = set()

        # 1. Encuentra los tokens especiales y sus posiciones
        if allowed_special:
            # Creamos un regex para encontrar los tokens especiales
            special_pattern = re.compile(
                "|".join(re.escape(s) for s in sorted(allowed_special, key=lambda x: -len(x)))
            )
            # Dividimos el texto en partes: normales y especiales
            splits = []
            last = 0
            for match in special_pattern.finditer(text):
                if match.start() > last:
                    splits.append((False, text[last:match.start()]))
                splits.append((True, match.group()))
                last = match.end()
            if last < len(text):
                splits.append((False, text[last:]))
        else:
            splits = [(False, text)]

        # 2. Tokeniza cada parte
        ids = []
        for is_special, fragment in splits:
            if is_special:
                # Añade el token especial como un solo id (puedes mapearlo a un id fijo si lo deseas)
                # Aquí simplemente guardamos el string, pero puedes mapearlo a un id especial si lo necesitas
                ids.append(fragment)
            else:
                # Tokenización normal por regex
                text_chunks = self.compiled_pattern.findall(fragment)
                chunk_ids = []
                for chunk in text_chunks:
                    if chunk:
                        chunk_ids.extend(chunk.encode("utf-8"))
                # Aplica los merges como en train
                while len(chunk_ids) >= 2:
                    candidate_pairs = [(p, self.merges[p]) for p in zip(chunk_ids, chunk_ids[1:]) if p in self.merges]
                    if not candidate_pairs:
                        break
                    pair, idx = min(candidate_pairs, key=lambda x: x[1])
                    new_chunk_ids = []
                    j = 0
                    while j < len(chunk_ids) - 1:
                        if (chunk_ids[j], chunk_ids[j+1]) == pair:
                            new_chunk_ids.append(idx)
                            j += 2
                        else:
                            new_chunk_ids.append(chunk_ids[j])
                            j += 1
                    if j == len(chunk_ids) - 1:
                        new_chunk_ids.append(chunk_ids[-1])
                    chunk_ids = new_chunk_ids
                ids.extend(chunk_ids)
        return ids
    def train_optimized(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 1. Convertir texto a bytes (array plano)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)  # lista de enteros 0..255

        # 2. Inicializar vocabulario
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in tqdm(range(num_merges), total=num_merges, disable=not verbose):
            # 3. Contar pares consecutivos
            stats = Counter(zip(ids, ids[1:]))
            if not stats:
                break
            pair = max(stats, key=stats.get)

            # 4. Crear nuevo token y hacer merge
            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 5. Reemplazar todos los pares por el nuevo token (merge vectorizado)
            new_ids = []
            j = 0
            while j < len(ids) - 1:
                if (ids[j], ids[j+1]) == pair:
                    new_ids.append(idx)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            if j == len(ids) - 1:
                new_ids.append(ids[-1])
            ids = new_ids

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

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
