import sys
sys.path.append('..')
from minbpe.v2 import RegexTokenizer
import matplotlib.pyplot as plt
import torch
from transformer.pedro_model import GPTLanguageModel
from transformer import BASE_CONFIG, selConfig
import numpy as np
import math
from typing import Tuple, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(3647)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

if True:
    useWords=True
    with open('../output/combined_text4.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    words = sorted(list(set(text.split())))
    print(f"Longitud del texto: {len(text)} caracteres")
    print(f"Número de palabras: {len(words)}")
    print(f"Número de caracteres únicos: {len(chars)}")
    if useWords:
        
        vocab_size = len(words)

        stoi = { word:i for i, word in enumerate(words) }
        itos = { i:word for i, word in enumerate(words) }
        encode = lambda s: [stoi[word] for word in s.split()] # encoder: take a string, output a list of integers
        decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    else:
        vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
else:
    data_path = "../output/encoded_data/encoded_atlaset_p_nuevo4.npy"
    data = np.load(data_path, mmap_mode='r')

    tokenizer = RegexTokenizer()
    tokenizer_path = "../output/tokenizer/pedro_nuevo_tokenizer3.model"
    tokenizer.load(model_file=tokenizer_path)

    def get_vocab_size(tokenizer: RegexTokenizer) -> int:
        vocab = tokenizer.vocab
        special_tokens = tokenizer.special_tokens
        return len(vocab) + len(special_tokens)

    vocab_size = get_vocab_size(tokenizer)



def setup_optimized_training():
    torch.cuda.empty_cache()
    selConfig('pedro-medium (124M)')
    
    # Ajusta batch_size a múltiplo de 2 (número GPUs)
    batch_size = 2  
    block_size = BASE_CONFIG['context_length']
    n_embd = BASE_CONFIG['emb_dim']
    n_head = BASE_CONFIG['n_heads']
    n_layer = BASE_CONFIG['n_layers']
    dropout = BASE_CONFIG['dropout']
    
    # Crear modelo en cuda:0 
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device='cuda:0'
    ).to('cuda:0')
    
    # Envolver en DataParallel para multi-GPU
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    
    return model, batch_size, block_size

model, batch_size, block_size = setup_optimized_training()

@torch.no_grad()
def estimate_loss_optimized(model, val_loader, max_eval_batches=100):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
        if batch_idx >= max_eval_batches:
            break
        
        x_batch, y_batch = x_batch.to('cuda', non_blocking=True), y_batch.to('cuda', non_blocking=True)
        
        logits, loss = model(x_batch, y_batch)
        
        # Asegurar que loss sea un escalar antes de usar .item()
        loss = loss.mean()
        
        val_loss += loss.item()
        num_batches += 1
    
    return {'val': val_loss / num_batches if num_batches > 0 else float('inf')}

learning_rate = 6e-4
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str = "../output/pre_training/run_8/checkpoint.pth"
) -> None:
    checkpoint = {
        'epoch': epoch,
        # Accede al modelo original dentro de DataParallel
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)

class GPTDataset(Dataset):
    def __init__(self, data, block_size, split='train', train_split=0.9):
        self.data = data
        self.block_size = block_size
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = int(train_split * len(data))
        else:
            self.start_idx = int(train_split * len(data))
            self.end_idx = len(data)
            
        self.length = self.end_idx - self.start_idx - block_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        x = torch.tensor(self.data[actual_idx:actual_idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[actual_idx + 1:actual_idx + self.block_size + 1], dtype=torch.long)
        return x, y

def train_optimized():
    torch.set_float32_matmul_precision('high')  # Opcional
    
    gradient_accumulation_steps = 8
    eval_interval = 500
    save_interval = 5000
    learning_rate = 3e-4
    num_workers = 4
    num_epochs=2
    
    train_dataset = GPTDataset(data, block_size, split='train')
    val_dataset = GPTDataset(data, block_size, split='val')
    split_index = int(0.9*len(data))

    total_data_to_process = split_index - block_size
    total_data_to_process_in_batches = total_data_to_process // batch_size

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    #optimizer = torch.optim.AdamW(
    #    model.parameters(), 
    #    lr=learning_rate,
    #    betas=(0.9, 0.95),
    #    weight_decay=1e-2,
    #    eps=1e-8
    #)
    optimizer=model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=1e-6
    )
    
    batches_processed = 0
    step=0
    train_losses, val_losses = [], []
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Training")):
            x_batch, y_batch = x_batch.to('cuda', non_blocking=True), y_batch.to('cuda', non_blocking=True)
            
            logits, loss = model(x_batch, y_batch)
            
            # Aseguramos que loss sea escalar promedio
            loss = loss.mean()
            
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                lr = get_lr(batch_idx)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            if step % eval_interval == 0:
                losses = estimate_loss_optimized(model, val_loader)
                avg_train_loss = epoch_loss / num_batches
                print(
                    f"Batch {batches_processed}: "
                    f"train loss {avg_train_loss:.4f}, "
                    f"val loss {losses['val']:.4f}, "
                    f"norm {norm:.4f}, "
                    f"learning rate {lr:.4e}"
                )
                train_losses.append(avg_train_loss)
                val_losses.append(losses['val'])
                model.train()
            
            if step % save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=step,
                    loss=loss.item(),
                    file_path=f"../output/pre_training/run_8/checkpoint_{step}.pth"
                )
    if batches_processed % gradient_accumulation_steps != 0:
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)

    return train_losses, val_losses

def genGraph(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Evaluation Step")
    plt.ylim(0)
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig('../output/pre_training/run_8/foo.png')

def testIa():
    input_tokens = encode("Hola, soy una")
    input_tokens = torch.tensor(
        input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model.generate(input_tokens=input_tokens, max_new_tokens=50)

    print(decode(output[0].tolist()))

if __name__ == "__main__":
    train_loss, val_loss=train_optimized()
    genGraph(train_loss, val_loss)
    testIa()

