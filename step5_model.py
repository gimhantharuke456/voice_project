"""
Step 5 — CDAE Model Architecture
Matches the architecture trained in AgentVoiceTraining.ipynb:
  Encoder  : [Conv2d(1,32) -> ReLU -> MaxPool] -> [Conv2d(32,64) -> ReLU -> MaxPool]
  Decoder  : [ConvTranspose2d(64,32)] -> ReLU -> [ConvTranspose2d(32,1)] -> Sigmoid

Input is padded to the nearest multiple of 4 (2 x MaxPool) then cropped back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


class CDAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1),  # 0
            nn.ReLU(),                          # 1
            nn.MaxPool2d(2, 2),                # 2
            nn.Conv2d(32, 64, 3, padding=1),  # 3
            nn.ReLU(),                          # 4
            nn.MaxPool2d(2, 2),                # 5
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 0
            nn.ReLU(),                                  # 1
            nn.ConvTranspose2d(32,  1, 2, stride=2),  # 2
            nn.Sigmoid(),                               # 3
        )

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Pad to nearest multiple of 4 (2 MaxPool layers)
        pad_h = (4 - orig_h % 4) % 4
        pad_w = (4 - orig_w % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.decoder(self.encoder(x))

        # Crop back to original shape
        return x[:, :, :orig_h, :orig_w]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_summary(model, input_shape=(1, 1, 257, 376)):
    print(f"\n{BOLD}Layer shapes (batch=1, input={input_shape[1:]}){RESET}")
    print(f"  {'Layer':<35} {'Output shape'}")
    print(f"  {'-'*52}")

    x = torch.zeros(*input_shape)
    orig_h, orig_w = x.shape[2], x.shape[3]
    pad_h = (4 - orig_h % 4) % 4
    pad_w = (4 - orig_w % 4) % 4
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    for i, layer in enumerate(model.encoder):
        x = layer(x)
        name = f"encoder.{i} ({layer.__class__.__name__})"
        print(f"  {name:<35} {tuple(x.shape)}")

    for i, layer in enumerate(model.decoder):
        x = layer(x)
        name = f"decoder.{i} ({layer.__class__.__name__})"
        print(f"  {name:<35} {tuple(x.shape)}")

    x = x[:, :, :orig_h, :orig_w]
    print(f"  {'crop to original':<35} {tuple(x.shape)}")
    print()
    print(f"  Trainable parameters : {count_parameters(model):,}")
    match = tuple(x.shape) == input_shape
    print(f"  Output matches input : {'YES' if match else 'NO'}")
    return match


if __name__ == "__main__":
    print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 5: CDAE Model ==={RESET}")

    model = CDAE()
    ok    = print_summary(model)

    dummy = torch.randn(4, 1, 257, 376)
    out   = model(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output outside [0,1]"

    if ok:
        print(f"\n{GREEN}Model OK. Matches trained weights.{RESET}\n")
