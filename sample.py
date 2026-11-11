import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import csv
import time
import argparse
from PIL import Image
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from model import UNet
from diffusion import GaussianDiffusion, EMA


def timed_sample(fn, *args, device=None, **kwargs):
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn(*args, device=device, **kwargs)
    if device and device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t0


def save_grid_figure(tensors, nrow, title, path):
    grid = vutils.make_grid(tensors, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(max(nrow * 1.8, 6.0), 3.8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    plt.axis("off")
    plt.title(title, fontweight="bold", fontsize=11)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/ckpt_latest.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument("--schedule", type=str, default="cosine")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    ema = EMA(model)
    diffusion = GaussianDiffusion(timesteps=1000, schedule=args.schedule).to(device)

    if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])

    model.eval()
    ema_model = ema.ema_model.eval()

    # Grilla final de 64 muestras
    torch.manual_seed(42)
    samples_64 = diffusion.ddim_sample(ema_model, (64, 3, 64, 64), steps=50, device=device)
    vutils.save_image(samples_64, f"{args.output_dir}/final_grid_64.png", nrow=8, normalize=True, value_range=(-1, 1))

    # Comparación Raw vs. EMA
    torch.manual_seed(42)
    raw = diffusion.ddim_sample(model, (8, 3, 64, 64), steps=50, device=device)
    torch.manual_seed(42)
    ema_s = diffusion.ddim_sample(ema_model, (8, 3, 64, 64), steps=50, device=device)
    save_grid_figure(
        torch.cat([raw, ema_s]),
        8,
        "Comparación: Pesos Raw (arriba) vs. EMA (abajo)",
        f"{args.output_dir}/ema_vs_raw.png"
    )

    # Comparación DDPM vs. DDIM
    torch.manual_seed(42)
    ddpm, t_ddpm = timed_sample(diffusion.ddpm_sample, ema_model, (4, 3, 64, 64), device=device)
    torch.manual_seed(42)
    ddim, t_ddim = timed_sample(diffusion.ddim_sample, ema_model, (4, 3, 64, 64), steps=50, eta=0.0, device=device)
    speedup = t_ddpm / max(t_ddim, 1e-4)
    save_grid_figure(
        torch.cat([ddpm, ddim]),
        4,
        f"DDPM (1000 pasos: {t_ddpm:.1f}s) vs. DDIM (50 pasos: {t_ddim:.1f}s) | Aceleración: {speedup:.1f}x",
        f"{args.output_dir}/ddpm_vs_ddim.png"
    )

    # Animación GIF de desruidado
    torch.manual_seed(42)
    steps = diffusion.ddim_sample(ema_model, (1, 3, 64, 64), steps=50, device=device, return_all_timesteps=True)
    frames = [
        Image.fromarray(((s[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy())
             .resize((256, 256), Image.Resampling.NEAREST)
        for s in steps
    ]
    frames.extend([frames[-1]] * 12)
    imageio.mimsave(f"{args.output_dir}/denoising.gif", frames, fps=10, loop=0)

    # Curva de pérdida
    loss_csv = "outputs/logs/loss.csv"
    if os.path.exists(loss_csv):
        with open(loss_csv) as f:
            rows = list(csv.DictReader(f))
        plt.figure(figsize=(6, 4))
        plt.plot([int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows], color="#2980b9", lw=2, marker="o")
        plt.title("Curva de Pérdida (MSE)", fontweight="bold")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{args.output_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("Figuras generadas exitosamente en", args.output_dir)


if __name__ == "__main__":
    main()
