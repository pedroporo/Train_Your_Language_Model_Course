with open("../data/AtlaSetCombined.txt", "r") as f:
    #number_of_characters_to_read = 10_000_000
    #text_sequence = f.read(number_of_characters_to_read)
    text_sequence = f.read()

len(text_sequence)
import sys

sys.path.append("..")
from minbpe.v2 import RegexTokenizer, RegexTokenizerGPU

tokenizer = RegexTokenizerGPU()
tokenizer.train_optimized(text_sequence, vocab_size=35_605,verbose=True)
vocab = tokenizer.vocab
vocab
max_vocab_id = list(tokenizer.vocab.keys())[-1]
tokenizer.special_tokens = {
    "<|startoftext|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endoftext|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4,
    "<|padding|>": max_vocab_id + 5,
    "<|start_turn|>": max_vocab_id + 6,
    "<|end_turn|>": max_vocab_id + 7,
}

tokenizer.save(file_prefix="../output/tokenizer/pedro_nuevo_tokenizer")
