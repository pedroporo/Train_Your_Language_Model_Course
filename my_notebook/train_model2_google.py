import sys
sys.path.append('..')
from source_code.minbpe import RegexTokenizer
import torch
from source_code.transformer.pedro_model import GPTLanguageModel
from source_code.transformer import BASE_CONFIG, selConfig
import numpy as np
from typing import Tuple,Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader



torch.manual_seed(3647)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = "/content/drive/MyDrive/Colab_Notebooks/LLM_Course/Output/encoded_data/encoded_atlaset_p_nuevo.npy"
data = np.load(data_path, mmap_mode='r')


tokenizer = RegexTokenizer()
tokenizer_path = "/content/drive/MyDrive/Colab_Notebooks/LLM_Course/Output/tokenizer/pedro_nuevo_tokenizer.model"
#tokenizer_path = "../output/tokenizer/pedro_tokenizer.model"
tokenizer.load(model_file=tokenizer_path)


def get_vocab_size(tokenizer: RegexTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)


vocab_size = get_vocab_size(tokenizer)





def setup_optimized_training():
    # Configuración de memoria
    torch.cuda.empty_cache()
    
    # Configuración del modelo con optimizaciones
    selConfig('gpt2-medium (355M)')
    
    # Parámetros optimizados
    #batch_size = min(32, torch.cuda.get_device_properties(0).total_memory // (1024**3))  # Ajuste dinámico
    batch_size = 32000 # Ajuste estatico
    block_size = BASE_CONFIG['context_length']
    n_embd = BASE_CONFIG['emb_dim']
    n_head = BASE_CONFIG['n_heads']
    n_layer = BASE_CONFIG['n_layers']
    dropout = BASE_CONFIG['dropout']
    # Crear modelo optimizado
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    ).to(device)
    
    return model, batch_size,block_size

model, batch_size,block_size= setup_optimized_training()

@torch.no_grad()
def estimate_loss_optimized(model, val_loader, max_eval_batches=100):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
        if batch_idx >= max_eval_batches:
            break
            
        x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=device):
            _, loss = model(x_batch, y_batch)
        
        val_loss += loss.item()
        num_batches += 1
    
    return {'val': val_loss / num_batches if num_batches > 0 else float('inf')}


def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str = "checkpoint.pth"
) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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
    # Configuración mejorada
    torch.set_float32_matmul_precision('high')
    
    # Parámetros optimizados
    #batch_size = 32000  # Incremento significativo del batch size
    gradient_accumulation_steps = 80  # Reducido para compensar el mayor batch size
    eval_interval = 500  # Evaluación más frecuente
    save_interval = 5000
    learning_rate = 3e-4
    num_workers = 4  # Para carga asíncrona de datos
    
    # Crear datasets optimizados
    train_dataset = GPTDataset(data, block_size, split='train')
    val_dataset = GPTDataset(data, block_size, split='val')
    
    # DataLoaders con optimizaciones
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Acelera transferencias a GPU
        persistent_workers=True  # Mantiene workers activos
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Optimizador con configuración mejorada
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.95),  # Configuración recomendada para transformers
        weight_decay=1e-1,
        eps=1e-8
    )
    
    # Compilación del modelo para PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model_compiled = torch.compile(model, mode='max-autotune')
    else:
        model_compiled = model
    
    # Scheduler de learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=1e-6
    )
    
    # Variables de tracking
    step = 0
    train_losses, val_losses = [], []
    
    model_compiled.train()
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(1):  # Un epoch completo
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Training")):
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            # Forward pass
            with torch.amp.autocast(device_type=device):  # Mixed precision para mejor rendimiento
                logits, loss = model_compiled(x_batch, y_batch)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # Evaluación periódica
            if step % eval_interval == 0:
                losses = estimate_loss_optimized(model_compiled, val_loader)
                avg_train_loss = epoch_loss / num_batches
                print(f"Step {step}: train loss {avg_train_loss:.4f}, val loss {losses['val']:.4f}")
                train_losses.append(avg_train_loss)
                val_losses.append(losses['val'])
                
                model_compiled.train()
            
            # Guardado periódico
            if step % save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=step,
                    loss=loss.item(),
                    file_path=f"/content/drive/MyDrive/Colab_Notebooks/LLM_Course/Output/pre_training/run_1/checkpoint_{step}.pth"
                )
    
    return train_losses, val_losses

if __name__ == "__main__":
    train_optimized()
