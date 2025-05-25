import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import lightning as L


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        embed_dim -     dim of embedding layer
        hidden_dim -    dim of hidden layer in feed-forward network
        num_heads -     number of heads to use in Encoder block
        dropout -       probabilibty of dropout apply in feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim),
                    nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.ffn(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_channels,
            num_heads,
            num_layers,
            patch_size,
            num_patches,
            num_classes,
            dropout=0.0
    ):
        """
        embed_dim -         dim of embedding layer
        hidden_dim -        dim of hidden layer in feed-forward network
        num_channels -      number of channel of input (3 for RGB)
        num_heads -         number of heads to user in multi-head attention
        num_layers -        number of layers to use in transformer
        num_classes -       number of classes to predict
        patch_size -        number of pixels that the patches have
        num_patches -       maximum number of patches that a image have
        dropout -           probability of dropout
        """

        super().__init__()

        self.patch_size = patch_size

        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim) # (num_channels * patch^2)->(embed_dim)

        self.transformer_layer = nn.Sequential(
            *(EncoderBlock(embed_dim,hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        ) # (embed_dim)->(embed_dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, num_classes)) # (embed_dim)->(num_classes)
        
        self.dropout = nn.Dropout(dropout)

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # (1, 1, embed_dim)

        # pos embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim)) # (1, 1, embed_dim)

    def img_to_patch(self, x, patch_size, flatten_channels=True):
        """
        x -                 image shape [B, C, H, W]
        patch_size -        number of pixel of the patch
        flatten_channels -  if true, the patch will be returned in a flattened format as feature vector instead of a image grid
        """

        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size) #(B, C, num_p_H, ps, num_p_W, ps)
        x = x.permute(0, 2, 4, 1, 3, 5) #(B, num_p_H, num_p_W, C, ps, ps)
        x = x.flatten(1, 2) # (B, num_p_H*num_p_W, C, ps, ps)
        if flatten_channels:
            x = x.flatten(2, 4) # # (B, num_p_H*num_p_W, C*ps*ps)
        return x

    def forward(self, x):
        x = self.img_to_patch(x, self.patch_size) # (B, num_p_H*num_p_W, C*ps*ps)
        B, T, _ = x.shape
        x = self.input_layer(x) # (B, num_p_H*num_p_W, embed_dim)

        cls_token = self.cls_token.repeat(B, 1, 1) # (B, 1, embed_dim)

        x = torch.cat([cls_token, x], dim=1) # (B, num_p_H*num_p_W + 1, embed_dim)

        x = x + self.pos_embedding[:, :T + 1] # (B, num_p_H*num_p_W + 1, embed_dim)

        x = self.dropout(x)
        x = x.transpose(0, 1) # (num_p_H*num_p_W + 1, B, embed_dim)
        x = self.transformer_layer(x) # (num_p_H*num_p_W + 1, B, embed_dim)

        cls = x[0]
        out = self.mlp_head(cls) # (1, B, embed_dim)

        return out
    

class ViT(L.LightningModule):
    def __init__(self, model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)





