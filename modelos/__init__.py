from .tokenizers import Tokenizer,BasicTokenizer,FastRegexTokenizer,GPT4Tokenizer,RegexTokenizer
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.2

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "dropout": 0.2,  # Dropout rate
    "qkv_bias": True,  # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "pedro-small (124M)": {"emb_dim": 768, "n_layers":8, "n_heads": 12},
    "pedro-medium (124M)": {"emb_dim": 768, "n_layers":14, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "lilith-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}




def selConfig(model):
    BASE_CONFIG.update(model_configs[model])


def getConfigs():
    return model_configs

selConfig("gpt2-medium (355M)")