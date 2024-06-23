from typing import List

import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)

# constants / hyperparams
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "../data"

BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
EVAL_ITERS = 200
N_EMBEDDINGS = 384
N_ATTENTION_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"


def download_tinyshakespeare() -> str:
    """Returns downloaded data file path"""
    # create dir if needed
    os.makedirs(DATA_DIR, exist_ok=True)

    # download and open the file
    data_file_path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    response = requests.get(url=DATA_URL)
    with open(data_file_path, "wb") as f:
        f.write(response.content)

    return data_file_path


data_file_path = download_tinyshakespeare()

with open(data_file_path, "r", encoding="utf-8") as f:
    tinyshakespeare_text = f.read()


# Create encoder and decoder for vocabulary (character level)
unique_characters = sorted(list(set(tinyshakespeare_text)))
vocab_size = len(unique_characters)

char_to_int_mapping = {ch: i for i, ch in enumerate(unique_characters)}
int_to_char_mapping = {i: ch for i, ch in enumerate(unique_characters)}


def encode(s: str) -> List[int]:
    """Return a list of integers for a given string"""
    return [char_to_int_mapping[c] for c in s]


def decode(l: List[int]) -> str:
    """Return a string given a list of integers"""
    return "".join([int_to_char_mapping[i] for i in l])


# create a long vector of integers from the entire training data
data = torch.tensor(encode(tinyshakespeare_text)).to(device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Generate a small batch of inputs X and targets y"""
    data = train_data if split == "train" else val_data
    # generate random offsets of the (train/val) data in the range [0, len - BLOCK_SIZE]
    random_data_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i : i + BLOCK_SIZE] for i in random_data_indices])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in random_data_indices])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, y = get_batch(split)
            _, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention (decoder style because of tril)"""

    def __init__(self, head_size):
        super().__init__()
        # linear projections that will be applied to all nodes (embeddings)
        self.key = nn.Linear(
            N_EMBEDDINGS, head_size, bias=False
        )  # biases are typically not used
        self.query = nn.Linear(N_EMBEDDINGS, head_size, bias=False)
        self.value = nn.Linear(N_EMBEDDINGS, head_size, bias=False)

        # tril is not a Parameter of the model, so use a buffer
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        w = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) ---> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        # weighted aggregation of values
        v = self.value(x)
        out = w @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBEDDINGS, N_EMBEDDINGS)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # run all heads in parallel in a list
        # then concatenate the outputs over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # projection for residual
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBEDDINGS, 4 * N_EMBEDDINGS),
            nn.ReLU(),
            nn.Linear(
                4 * N_EMBEDDINGS, N_EMBEDDINGS
            ),  # projection for going back to residual pathway
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        head_size = N_EMBEDDINGS // N_ATTENTION_HEADS
        self.sa = MultiHeadAttention(N_ATTENTION_HEADS, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBEDDINGS)
        self.ln2 = nn.LayerNorm(N_EMBEDDINGS)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBEDDINGS)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBEDDINGS)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.lm_head = nn.Linear(N_EMBEDDINGS, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # batch (4), time(8), channel (65 = vocab_size)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)  # B*T
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate max_new_tokens tokens in a bigram fashion (context=1)

        Args:
            idx (List): (B, T) list of indices in the current context
        """
        for _ in range(max_new_tokens):
            # crop/clip idx to the last block_size tokens because of positional embeddings size
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get logits (preds)
            logits, _ = self(idx_cond)  # (B, T, C)
            # enforce bigram = get last timestep only
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.2f}, val loss {losses['val']:.2f}"
        )

    xb, yb = get_batch("train")

    # eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
