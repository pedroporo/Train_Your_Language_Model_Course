import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
sys.path.append('..')

from minbpe import RegexTokenizer
from transformer.pedro_model import GPTLanguageModel
from transformer import BASE_CONFIG, selConfig


torch.manual_seed(3647)

# Configuración
data_path = "../output/encoded_data/encoded_atlaset.npy"
tokenizer_path = "../output/tokenizer/darija_tokenizer.model"
selConfig('pedro-medium (124M)')
block_size = BASE_CONFIG['context_length']
n_embd = BASE_CONFIG['emb_dim']
n_head = BASE_CONFIG['n_heads']
n_layer = BASE_CONFIG['n_layers']
dropout = BASE_CONFIG['dropout']
batch_size = 1  # Aumenta el batch_size para mayor velocidad si tu GPU lo permite

# Carga de datos y tokenizer
data = np.load(data_path, mmap_mode='r')
split_index = int(0.9 * len(data))
tokenizer = RegexTokenizer()
tokenizer.load(model_file=tokenizer_path)

def get_vocab_size(tokenizer):
    vocab = tokenizer.vocab
    special_tokens = getattr(tokenizer, "special_tokens", {})
    return len(vocab) + len(special_tokens)

vocab_size = get_vocab_size(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
).to(device)

# Dataset personalizado
class NPYDataset(Dataset):
    def __init__(self, data, block_size, split='train'):
        self.data = data
        self.block_size = block_size
        self.split = split
        if split == 'train':
            self.start = 0
            self.end = split_index
        else:
            self.start = split_index
            self.end = len(data)
        self.length = self.end - self.start - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.start + idx
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# DataLoaders
train_dataset = NPYDataset(data, block_size, split='train')
val_dataset = NPYDataset(data, block_size, split='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

@torch.no_grad()
def estimate_loss():
    output = {}
    eval_iters = 100
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for k, (x, y) in enumerate(loader):
            if k >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        output[split] = np.mean(losses)
    model.train()
    return output

def save_checkpoint(model, optimizer, epoch, loss, file_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)

def train():
    torch.set_float32_matmul_precision('high')
    gradient_accumulation_steps = 8
    eval_interval = 1000
    save_interval = 10000
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    train_losses, val_losses = [], []
    optimizer.zero_grad(set_to_none=True)
    batches_processed = 0
    torch.cuda.empty_cache()
    
    for epoch in range(1):  # puedes aumentar epochs si lo necesitas
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for it, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with torch.amp.autocast(enabled=(scaler is not None)):
                logits, loss = model(x_batch, y_batch)
                loss = loss / gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            batches_processed += 1

            if batches_processed % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Evaluación periódica
            if batches_processed % eval_interval == 0:
                losses = estimate_loss()
                print(
                    f"Batch {batches_processed}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

            # Guardado periódico
            if batches_processed % save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=batches_processed,
                    loss=loss.item(),
                    file_path=f"../output/pre_training/run_11/checkpoint_{batches_processed}.pth"
                )

    # Último paso de optimizer si quedan gradientes pendientes
    if batches_processed % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    train()
