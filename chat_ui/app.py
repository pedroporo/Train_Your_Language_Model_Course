from pathlib import Path
import sys

sys.path.append("..")


import torch
import chainlit


from minbpe import RegexTokenizer
from transformer.pedro_model import GPTLanguageModel
from transformer import BASE_CONFIG, selConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENS = {
    "start": "<|start_turn|>",
    "end": "<|end_turn|>",
    "separator": "<|separator|>",
    "eos": "<|endoftext|>",
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


def get_model_and_tokenizer():
    tokenizer = RegexTokenizer()
    tokenizer.load(model_file="../output/tokenizer/darija_tokenizer.model")

    selConfig("pedro-small (124M)")
    block_size = BASE_CONFIG["context_length"]
    n_embd = BASE_CONFIG["emb_dim"]
    n_head = BASE_CONFIG["n_heads"]
    n_layer = BASE_CONFIG["n_layers"]
    dropout = BASE_CONFIG["dropout"]
    vocab_size = get_vocab_size(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "../output/fine_tuning/qa/base/run_1/checkpoint_9.pth"
    # model_path = Path("..") / "output" / "base_model.pth"
    #if not model_path.exists():
    #    print(
    #        f"No se puede encontrar el archivo {model_path}. Por favor comprueba que sea correcto "
    #        "."
    #    )
    #    sys.exit()

    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        ignore_index=tokenizer.special_tokens["<|padding|>"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return tokenizer, model


def extract_response(response_text, input_text):
    return response_text[len(input_text) :].replace("### Response:", "").strip()


def get_generated_message(
    input_tokens: torch.Tensor,
    model: GPTLanguageModel,
    tokenizer: RegexTokenizer,
    block_size: int,
) -> str:
    model.eval()
    model_answer = ""
    while True:
        try:
            output_tokens = model.advanced_generation(
                input_tokens=input_tokens,
                max_new_tokens=100,
                temperature=0.9,
                top_k=50,
                top_p=None,
            )
            last_generated_token = output_tokens[0, -1].item()

            if last_generated_token in {
                tokenizer.special_tokens["<|endoftext|>"],
                tokenizer.special_tokens["<|end_turn|>"],
            }:
                break

            input_tokens = torch.cat((input_tokens, output_tokens[:, -1:]), dim=1)
            model_answer += tokenizer.decode([last_generated_token])

            if input_tokens.size(1) > block_size:
                break
        except Exception:
            continue
    return model_answer.strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model = get_model_and_tokenizer()


def get_system_message() -> str:
    return "Te llamas lilith eres una ia que te gustan los videojuegos. Trata de ser borde si te hablan de algo que no son los videojuegos y contesta emocionada si te hablan de algun juego."

turns = [{"role": "system", "content": get_system_message()}]
@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)
    print(message.__dict__)
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """
    turns.append({"role": "user", "content": message.content})
    tokens_id = get_input_tokens(turns=turns,tokenizer=tokenizer,device=device)
    model_answer = get_generated_message(input_tokens=tokens_id,model=model,tokenizer=tokenizer,block_size=1024)
    turns.append({"role": "assistant", "content": model_answer})
    # token_ids = generate(  # function uses `with torch.no_grad()` internally already
    #    model=model,
    #    idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content`
    #    max_new_tokens=35,
    #    context_size=model_config["context_length"],
    #    eos_id=50256
    # )
    #
    # text = token_ids_to_text(token_ids, tokenizer)
    # response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{model_answer}",  # This returns the model response to the interface
    ).send()
