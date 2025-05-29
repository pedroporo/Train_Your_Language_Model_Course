#op = 1
#with open("../data/combined_text.txt", "r") as f:
#    match (op):
#        case 0:
#            number_of_characters_to_read = 10_000_000
#            text_sequence = f.read(number_of_characters_to_read)
#        case 1:
#            text_sequence = f.read()
#        case _:
#            exit()
#
#print(len(text_sequence))
import sys

sys.path.append("..")
from minbpe.v2 import RegexTokenizer,FastRegexTokenizer
from minbpe import RegexTokenizer as rt
tokenizer = RegexTokenizer()
tokenizer.train("../data/private/combined_text.txt", vocab_size=36_384, verbose=True,num_workers=16)
vocab = tokenizer.vocab
print(vocab)
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

tokenizer.save(file_prefix="../output/tokenizer/pedro_nuevo_tokenizer3")
