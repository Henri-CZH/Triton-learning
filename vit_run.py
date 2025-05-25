from torchview import draw_graph
from vit_model import ViT

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
)

model_graph = draw_graph(model, input_size=(1, 3, 224, 224))
model_graph.resize_graph(scale=5)
model_graph.visual_graph.render(format='png',filename="vit_model")