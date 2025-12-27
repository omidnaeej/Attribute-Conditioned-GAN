import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image
from scipy.linalg import sqrtm
from tqdm.auto import tqdm

try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

class Inception2048(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Do NOT pass aux_logits. Torchvision sets it correctly for weights.
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.net.fc = nn.Identity()
        self.net.eval().to(device)

    @torch.no_grad()
    def forward(self, x01):
        # x01 in [0,1]
        x01 = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net(x01)

def fid_from_feats(real_feats, fake_feats):
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

@torch.no_grad()
def evaluate_model(config, netG, dataloader, device):
    print("--- Evaluating Model (FID via torchvision InceptionV3) ---")
    netG.eval()
    inc = Inception2048(device)

    real_batches = int(config.get("eval", {}).get("fid_real_batches", 10))
    fake_batches = int(config.get("eval", {}).get("fid_fake_batches", 10))
    bs = int(config["data"]["batch_size"])
    zdim = int(config["gan"]["latent_dim"])

    real_feats = []
    fake_feats = []

    # real
    for i, (real, _) in enumerate(tqdm(dataloader, desc="Real feats", leave=True)):
        if i >= real_batches:
            break
        real = real.to(device)
        real01 = ((real + 1) * 0.5).clamp(0, 1)
        feats = inc(real01).detach().cpu().numpy()
        real_feats.append(feats)

    # fake
    for _ in tqdm(range(fake_batches), desc="Fake feats", leave=True):
        z = torch.randn(bs, zdim, 1, 1, device=device)
        fake = netG(z)
        fake01 = ((fake + 1) * 0.5).clamp(0, 1)
        feats = inc(fake01).detach().cpu().numpy()
        fake_feats.append(feats)

    real_feats = np.concatenate(real_feats, axis=0)
    fake_feats = np.concatenate(fake_feats, axis=0)

    fid = fid_from_feats(real_feats, fake_feats)
    print(f"Final FID Score: {fid:.4f}")
    return fid

@torch.no_grad()
def evaluate_lpips(config, netG, dataloader, device, net="alex"):
    if not _HAS_LPIPS:
        print("LPIPS not installed, skipping.")
        return None

    print("--- Evaluating LPIPS ---")
    netG.eval()
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    pairs = int(config.get("eval", {}).get("lpips_pairs", 200))
    zdim = int(config["gan"]["latent_dim"])

    it = iter(dataloader)
    scores = []
    for _ in tqdm(range(pairs), desc="LPIPS pairs", leave=True):
        try:
            real, _ = next(it)
        except StopIteration:
            it = iter(dataloader)
            real, _ = next(it)

        real = real.to(device)[:1]
        z = torch.randn(1, zdim, 1, 1, device=device)
        fake = netG(z)
        scores.append(loss_fn(real, fake).mean().item())

    lp_mean = float(sum(scores) / len(scores))
    print(f"LPIPS mean: {lp_mean:.4f}")
    return lp_mean

@torch.no_grad()
def generate_images(config, netG, device):
    student_id = str(config["system"]["student_id"])
    out_dir = student_id
    os.makedirs(out_dir, exist_ok=True)

    n = int(config.get("eval", {}).get("generate_count", 2000))
    zdim = int(config["gan"]["latent_dim"])

    print(f"Generating {n} images to folder: {out_dir}")
    netG.eval()
    for i in tqdm(range(n), desc="Generating", leave=True):
        z = torch.randn(1, zdim, 1, 1, device=device)
        img = netG(z)
        img01 = ((img + 1) * 0.5).clamp(0, 1)
        save_image(img01, os.path.join(out_dir, f"{student_id}-{i:04d}.png"))
