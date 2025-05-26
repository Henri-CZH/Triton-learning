import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple


class LlamaConfig():
    def __init__(
            self,
            vocab_size=32000,
            hidden_dim=4096,
            num_hidden_layers=1,
            num_heads=32,
            num_kv_heads=8,
            ffn_dim=11008, # FFN layer dim
            rms_norm_eps=1e-6,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.hidden_dim, 2).float() / self.hidden_dim)) # (hidden_dim/2, )
        self.register_buffer("inv_freq", inv_freq)


    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq_len -     
        device -         
        """        
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq) # (seq_len, )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (seq_len, hidden_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, hidden_dim)
        return emb.sin(), emb.cos() # (seq_len, hidden_dim)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q -     (batch_size, seq_len, num_heads, hidden_dim)
    k -     (batch_size, seq_len, num_kv_heads, head_dim)
    sin -   (seq_len, 1, head_dim)
    cos -   (seq_len, 1, head_dim)
    """       
    q_embed = (q * cos) + (rotate_half(q) * sin) # (batch_size, seq_len, num_heads, head_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin) # (batch_size, seq_len, num_kv_heads, head_dim)
    return q_embed, k_embed


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """将输入张量的后一半维度旋转180度"""
    """
    hidden_states - (batch_size, seq_len, hidden_dim)
    """

    x1, x2 = hidden_states.chunk(2, dim=-1) # (batch_size, seq_len, hidden_dim//2)
    return torch.cat((-x2, x1), dim=-1) # (batch_size, seq_len, hidden_dim)


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(config.hidden_dim, config.hidden_dim, bias = False)
        self.wk = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(config.hidden_dim, config.hidden_dim, bias = False)
        self.rotary_emb = RotaryEmbedding(config)
    
    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden_states -     (batch_size, seq_len, hidden_dim)
        attn_mask -         # (1, 1, seq_len, seq_len)
        """

        batch_size, seq_len, _  = hidden_states.shape
        q = self.wq(hidden_states) # (batch_size, seq_len, hidden_dim)
        k = self.wk(hidden_states) # (batch_size, seq_len, num_kv_heads * hidden_dim)
        v = self.wv(hidden_states) # (batch_size, seq_len, num_kv_heads * hidden_dim)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)    # (batch_size, seq_len, num_heads, hidden_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (batch_size, seq_len, num_kv_heads, head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim) # (batch_size, seq_len, num_kv_heads, head_dim)

        # RoPE
        sin, cos = self.rotary_emb(seq_len, hidden_states.device) # [seq_len, head_dim]
        if position_ids is not None:
            sin = sin[position_ids].unsqueeze(1) # [batch_size, seq_len, 1, head_dim]
            cos = cos[position_ids].unsqueeze(1) # [batch_size, seq_len, 1, head_dim]
        else:
            sin = sin.unsqueeze(1).unsqueeze(1)  # [1, seq_len, 1, head_dim]
            cos = cos.unsqueeze(1).unsqueeze(1)  # [1, seq_len, 1, head_dim]     
        
        q, k = apply_rotary_pos_emb(q, k, sin, cos)

        # GQA
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2) # (batch_size, seq_len, num_heads, head_dim)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2) # (batch_size, seq_len, num_heads, head_dim)

        attn = (q @ k.transpose(2, 3)) * self.scale # (batch_size, num_heads, seq_len, seq_len)
        if attn_mask is not None:
            attn = attn + attn_mask
        else:
            attn_mask = torch.full((1, 1, hidden_states.shape[1], hidden_states.shape[1]), float("-float"), device=hidden_states.device) # [1, 1, seq_len, seq_len]
            attn_mask = torch(attn_mask, diagnoal=1)
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        output = (attn @ v)  #(batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2) #(batch_size, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, -1) #(batch_size, seq_len, hidden_dim)

        return self.wo(output) #(batch_size, seq_len, hidden_dim)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_up = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False) # (batch_size, seq_len, hidden_dim)->(batch_size, seq_len, ffn_dim)
        self.w_gate = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False) # (batch_size, seq_len, hidden_dim)->(batch_size, seq_len, ffn_dim)
        self.w_down = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False) # (batch_size, seq_len, ffn_dim)->(batch_size, seq_len, hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        x -     (batch_size, seq_len, hidden_dim)
        """

        return self.w_down(F.silu(self.w_up(hidden_states)) * self.w_gate(hidden_states)) #(batch_size, seq_len, hidden_dim)


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.norm_gamma = nn.Parameter(torch.ones(config.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        x -     (batch_size, seq_len, hidden_dim)
        """
        rms_norm = torch.mean(hidden_states.pow(2), dim=-1, keepdim=True) # (batch_size, seq_len, )
        return hidden_states * torch.rsqrt(rms_norm + self.rms_norm_eps) * self.norm_gamma


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupQueryAttention(config)
        self.ffn = SwiGLU(config)
        self.input_rms_norm = RMSNorm(config)
        self.post_atten_rms_norm = RMSNorm(config)

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        hidden_states -     (batch_size, seq_len, hidden_dim)
        """        
        # GQA
        residual = hidden_states # (batch_size, seq_len, hidden_dim)
        hidden_states = self.input_rms_norm(hidden_states) # (batch_size, seq_len, hidden_dim)
        hidden_states = self.attn(hidden_states, attn_mask, position_ids) # (batch_size, seq_len, hidden_dim)
        hidden_states = residual + hidden_states # (batch_size, seq_len, hidden_dim)

        # FFN
        residual = hidden_states
        hidden_states = self.post_atten_rms_norm(hidden_states) # (batch_size, seq_len, hidden_dim)
        hidden_states = self.ffn(hidden_states) # (batch_size, seq_len, hidden_dim)

        return residual + hidden_states # (batch_size, seq_len, hidden_dim)


class LlamaModel(nn.Module):
    # config_class = LlamaConfig

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.llama_layer = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]) # (batch_size, seq_len, hidden_dim)
        self.rms_norm = RMSNorm(config) # (batch_size, seq_len, hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size) # (batch_size, seq_len, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x - (batch_size, seq_len)
        """

        hidden_states = self.embed_tokens(x) # (batch_size, seq_len)->(batch_size, seq_len, hidden_dim)

        for layer in self.llama_layer:
            hidden_states = layer(hidden_states, attn_mask, position_ids) # (batch_size, seq_len, hidden_dim)
        
        hidden_states = self.rms_norm(hidden_states) # (batch_size, seq_len, hidden_dim)
        return self.output(hidden_states) # (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    config = LlamaConfig()

    model = LlamaModel(config).to("cuda")

    # 模拟输入
    input_ids = torch.randint(0, 32000, (1, 128), device="cuda") # [batch_size, seq_len]
    output = model(input_ids)
    print(f"输出形状: {output.shape}")  # [batch_size, seq_len, vocab_size]
    