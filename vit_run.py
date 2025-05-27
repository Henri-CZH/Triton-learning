from torchview import draw_graph
from vit_model import ViT
import torch

if __name__ == "__main__":
        model = ViT(model_kwargs= {
                "embed_dim": 256,
                "hidden_dim": 512,
                "num_channels": 3,
                "num_heads": 8,
                "num_layers": 6,
                "patch_size": 16,
                "num_patches": 196,
                "num_classes": 10,
                "dropout": 0.2,
                }
        ).to("cuda")

        # 模拟输入
        input_ids = torch.normal(0, 1, (1, 3, 224, 224), device="cuda", dtype=torch.float32) # [B, C, H, W]
        output = model(input_ids)
        print(f"输出形状: {output.shape}")  # [B, num_classes]

        model_graph = draw_graph(model, input_ids)
        model_graph.resize_graph(scale=5)
        model_graph.visual_graph.render(format='png',filename="vit_model")