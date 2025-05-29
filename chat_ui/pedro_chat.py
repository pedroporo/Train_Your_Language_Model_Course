import torch
import sys

sys.path.append("..")

from minbpe import RegexTokenizer
from transformer.pedro_model_layer import GPTLanguageModel
from transformer import BASE_CONFIG, selConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENS = {
    "start": "<|start_turn|>",
    "end": "<|end_turn|>",
    "separator": "<|separator|>",
    "eos": "<|endoftext|>"
}


def get_vocab_size(tokenizer: RegexTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)


def get_input_tokens(turns: list[dict], tokenizer: RegexTokenizer, device: str) -> torch.Tensor:
    formatted_input = "".join(
        f"{TOKENS['start']}{turn['role']}{TOKENS['separator']}{turn['content']}{TOKENS['end']}"
        for turn in turns
    )
    formatted_input += f"{TOKENS['start']}assistant{TOKENS['separator']}"
    input_tokens = tokenizer.encode(formatted_input, allowed_special="all")
    return torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)


def get_generated_message(
    input_tokens: torch.Tensor,
    model: GPTLanguageModel,
    tokenizer: RegexTokenizer,
    block_size: int
) -> str:
    model.eval()
    model_answer = ""
    while True:
        try:
            output_tokens = model.advanced_generation(
                input_tokens=input_tokens, max_new_tokens=10, temperature=0.5, top_k=50, top_p=None
            )
            last_generated_token = output_tokens[0, -1].item()

            if last_generated_token in {tokenizer.special_tokens["<|endoftext|>"], tokenizer.special_tokens["<|end_turn|>"]}:
                break

            input_tokens = torch.cat(
                (input_tokens, output_tokens[:, -1:]), dim=1)
            model_answer += tokenizer.decode([last_generated_token])

            if input_tokens.size(1) > block_size:
                break
        except Exception:
            continue
    return model_answer.strip()


def get_system_message() -> str:
    return "Te llamas lilith eres una ia que te gustan los videojuegos. Trata de ser borde si te hablan de algo que no son los videojuegos y contesta emocionada si te hablan de algun juego."


def get_model(
    block_size: int,
    device: str,
    vocab_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    ignore_index: int,
) -> GPTLanguageModel:
    return GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        ignore_index=ignore_index,
    ).to(device)


def load_checkpoint(model: GPTLanguageModel, checkpoint_path: str) -> GPTLanguageModel:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model


def get_tokenizer(tokenizer_path: str) -> RegexTokenizer:
    tokenizer = RegexTokenizer()
    tokenizer.load(model_file=tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    tokenizer = get_tokenizer("../output/tokenizer/darija_tokenizer.model")
    vocab_size = get_vocab_size(tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    selConfig("pedro-small (124M)")
    block_size = BASE_CONFIG["context_length"]
    n_embd = BASE_CONFIG["emb_dim"]
    n_head = BASE_CONFIG["n_heads"]
    n_layer = BASE_CONFIG["n_layers"]
    dropout = BASE_CONFIG["dropout"]
    vocab_size = get_vocab_size(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../output/fine_tuning/qa/base/run_1/checkpoint_9.pth"
    model = get_model(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        ignore_index=tokenizer.special_tokens["<|padding|>"],
    )

    #checkpoint_path = "./output/fine_tuning/qa/base/run_2/checkpoint_50.pth"
    model = load_checkpoint(model, checkpoint_path=model_path)

    turns = [{"role": "system", "content": get_system_message()}]
    while True:
        user_message = input("Yo: ")
        if user_message.lower() == "quit":
            print("Goodbye!")
            break

        turns.append({"role": "user", "content": user_message})
        input_tokens = get_input_tokens(turns, tokenizer, device)
        model_answer = get_generated_message(
            input_tokens, model, tokenizer, 1024)
        turns.append({"role": "assistant", "content": model_answer})

        print(f"IA: {model_answer}\n")

