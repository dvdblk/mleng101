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

BATCH_SIZE = 4
BLOCK_SIZE = 8
MAX_ITERS = 5000
LEARNING_RATE = 1e-3
EVAL_INTERVAL = 500
EVAL_ITERS = 200
N_EMBEDDINGS = 32
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
    """One head of self-attention"""

    def __init__(self, head_size, n_embed):
        super().__init__()
        # linear projections that will be applied to all nodes (embeddings)
        self.key = nn.Linear(
            n_embed, head_size, bias=False
        )  # biases are typically not used
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # tril is not a Parameter of the model, so use a buffer
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        w = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) ---> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        w = F.softmax(w, dim=-1)

        # weighted aggregation of values
        v = self.value(x)
        out = w @ v
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.sa_head = Head(head_size=n_embed, n_embed=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # batch (4), time(8), channel (65 = vocab_size)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # apply self attention, (B, T, C)
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


model = BigramLanguageModel(vocab_size=vocab_size, n_embed=N_EMBEDDINGS)
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
