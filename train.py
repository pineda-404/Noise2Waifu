import os
import csv
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm

from model import UNet
from diffusion import GaussianDiffusion, EMA


class AnimeDataset(Dataset):
    def __init__(self, data_dir: str = "", max_images: int = None):
        exts = {".jpg", ".jpeg", ".png"}
        self.paths = [
            os.path.join(r, f)
            for r, _, files in os.walk(data_dir or "")
            for f in files if os.path.splitext(f)[1].lower() in exts
        ]

        if not self.paths:
            if data_dir:
                print(f"Aviso: No se encontraron imagenes en '{data_dir}'. Usando dataset dummy.")
            os.makedirs("data/dummy", exist_ok=True)
            self.paths = [os.path.join("data/dummy", f"dummy_{i:03d}.png") for i in range(64)]
            for i, p in enumerate(self.paths):
                if not os.path.exists(p):
                    Image.new("RGB", (64, 64), ((i * 35) % 256, (i * 65) % 256, (i * 95) % 256)).save(p)

        if max_images:
            self.paths = self.paths[:max_images]

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--max_images", type=int, default=30000)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    for folder in ["checkpoints", "samples", "logs", "figures"]:
        os.makedirs(f"outputs/{folder}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    dataset = AnimeDataset(data_dir=args.data_dir, max_images=args.max_images)
    print(f"Total imagenes cargadas: {len(dataset)}")

    batch_size = min(args.batch_size, len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) >= batch_size)
    )

    first_batch = next(iter(dataloader))
    vutils.save_image(
        first_batch[:min(64, first_batch.size(0))],
        "outputs/figures/real_samples.png",
        nrow=8,
        normalize=True,
        value_range=(-1, 1)
    )

    model = UNet().to(device)
    ema = EMA(model, decay=0.999)
    diffusion = GaussianDiffusion(timesteps=1000, schedule=args.schedule).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reanudando desde epoca {start_epoch}")

    log_file = "outputs/logs/loss.csv"
    if not os.path.exists(log_file) or start_epoch == 1:
        with open(log_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "lr"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoca {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = diffusion.p_losses(model, x, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{optimizer.param_groups[0]['lr']:.6f}"])

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(ckpt_data, "outputs/checkpoints/ckpt_latest.pt")
        if epoch % 10 == 0:
            torch.save(ckpt_data, f"outputs/checkpoints/ckpt_epoch_{epoch}.pt")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            n = min(16, args.batch_size)
            samples = diffusion.ddim_sample(ema.ema_model, (n, 3, 64, 64), steps=50, device=device)
            sample_path = f"outputs/samples/sample_epoch_{epoch:03d}.png"
            vutils.save_image(samples, sample_path, nrow=min(8, n), normalize=True, value_range=(-1, 1))
            print(f"Muestra guardada: {sample_path}")

    print("Entrenamiento completado.")


if __name__ == "__main__":
    train()
