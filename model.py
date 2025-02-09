import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    ## for kv-cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "dim must be even"

    ## build theta 
    ## acc to formula theta = 10000^-2(i - 1)/dim
    ## shape (Head_Dim / 2)

    theta_numerator = torch.arange(0, head_dim, 2).float()
    ## shape (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    m = torch.arange(seq_len, device=device)

    freqs = torch.outer(m, theta).float()

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex

def apply_rotary_emb(x: torch.Tensor, freqs_complex: torch.Tensor, device: str): 
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep:int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            ## (B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.w * self._norm(x.float()).type_as(x)
    

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()


        ## num heads for k and v
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        ## num heads for q
        self.n_heads_q = args.n_heads

        ## num times heads of k and v should be repeated to match the head of q
        self.n_rep = self.n_heads_q // self.n_kv_heads

        ## dim of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape # B, 1, dim

        #(B, 1, dim) -> (B, 1, H_Q * head_dim)
        xq = self.wq(x)

        # (B, 1, Dim) -> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, Head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, Head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)


        ## does not change the shape of the vectors
        xq = apply_rotary_emb(xq, freqs_complex, device=x.device)
        xk = apply_rotary_emb(xk, freqs_complex, device=x.device)

        ## replace the entry in cache for this token

        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        ## retrieve k and v till start_pos(token that we pass)
        ## (B, seq_len_KV, H_KV, head_dim)
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]

        ## Repeat the heads of the K and V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        ## (B, 1, H_Q, Head_dim) -> (B, H_Q, 1, head_dim)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1,2)
        values = values.transpose(1, 2)


        # (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_kv) --> (B, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len_kv) @ (B, h_q, seq_len_kv, head_dim) --> (b, h_q, 1, head_dim)
        out = torch.matmul(scores, values)

        # (b, h_q, 1, head_dim)
        out = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(out)
        



        




class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (Batch, seq_len, dim) -> (Batch, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab_size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.n_layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.n_layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len)

        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time"

        h = self.tok_embeddings(tokens)

        # retrive (m, theta) corresponding to pos [stary_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        ## next layers
        for layer in self.n_layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        logits = self.output(h).float()

        return logits

