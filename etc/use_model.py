import torch
from vit import get_vit

if __name__ == "__main__":
    path = "vit_small_dino_2x_mixed.pt"

    ckpt = torch.load(path)
    model = get_vit(**ckpt["hyperparams"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        y = model(x)
    print()

    print("\nSuccess!\n")
    print(f"Checkpoint: {path}")
    print(f"Hyperparameters: {ckpt['hyperparams']}")
    print(f"Model produces {y.shape[1]} features")
