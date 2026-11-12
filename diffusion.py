import os
import math
import copy

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_betas(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps).clamp(max=0.999)
    elif schedule == "cosine":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        f_t = torch.cos(((t / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(min=1e-5, max=0.999).float()
    raise ValueError(f"Schedule no soportado: {schedule}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    return a[t].reshape(-1, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.schedule = schedule

        betas = get_betas(schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_var",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return F.mse_loss(model(self.q_sample(x_start, t, noise), t), noise)

    def predict_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return ((x_t - sqrt_om * noise) / sqrt_ac).clamp(-1.0, 1.0)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
        eps = model(x, t)
        alpha = extract(self.alphas, t, x.shape)
        beta = extract(self.betas, t, x.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / sqrt_om) * eps)
        if t_idx == 0:
            return mean
        var = extract(self.posterior_var, t, x.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x)

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: tuple,
        device=None,
        return_all_timesteps: bool = False,
    ):
        device = device or self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img]
        for i in reversed(range(self.timesteps)):
            img = self.p_sample(model, img, i)
            if return_all_timesteps:
                imgs.append(img)
        return imgs if return_all_timesteps else img

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        steps: int = 50,
        eta: float = 0.0,
        device=None,
        return_all_timesteps: bool = False,
    ):
        device = device or self.betas.device
        img = torch.randn(shape, device=device)
        imgs = [img]
        times = torch.linspace(
            0, self.timesteps - 1, steps, dtype=torch.long, device=device
        ).flip(0)
        time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]

        for t_curr, t_prev in time_pairs:
            t = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)
            eps = model(img, t)
            x0 = self.predict_x0(img, t, eps)

            a_curr = self.alphas_cumprod[t_curr]
            a_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )

            sigma = eta * torch.sqrt(
                ((1 - a_prev) / (1 - a_curr)) * (1 - a_curr / a_prev)
            )
            dir_xt = torch.sqrt((1 - a_prev - sigma**2).clamp(min=0.0)) * eps
            noise = torch.randn_like(img) if (t_prev >= 0 and eta > 0.0) else 0.0

            img = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise
            if return_all_timesteps:
                imgs.append(img)
        return imgs if return_all_timesteps else img


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


if __name__ == "__main__":
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    args = parser.parse_args()

    os.makedirs("outputs/figures", exist_ok=True)
    timesteps = 1000

    # 1. Comparacion de Schedules
    linear_diff = GaussianDiffusion(timesteps=timesteps, schedule="linear")
    cosine_diff = GaussianDiffusion(timesteps=timesteps, schedule="cosine")

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        linear_diff.alphas_cumprod.cpu().numpy(),
        label=r"Lineal ($\beta_t \in [10^{-4}, 0.02]$)",
        color="#e74c3c",
        linewidth=2,
    )
    plt.plot(
        cosine_diff.alphas_cumprod.cpu().numpy(),
        label=r"Coseno (Nichol & Dhariwal, 2021)",
        color="#2980b9",
        linewidth=2,
    )
    plt.title(
        r"Comparación de Ruido Acumulado ($\bar{\alpha}_t$)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Paso de Difusión ($t$)", fontsize=11)
    plt.ylabel(r"$\bar{\alpha}_t$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    schedules_path = "outputs/figures/schedules_comparison.png"
    plt.savefig(schedules_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grafica guardada en: {schedules_path}")

    # 2. Forward Process visual en t = [0, 250, 500, 750, 999]
    torch.manual_seed(42)
    img_paths = [
        os.path.join(r, f)
        for r, _, files in os.walk(args.data_dir or "")
        for f in sorted(files)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]

    if img_paths:
        tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        base = tf(Image.open(img_paths[0]).convert("RGB")).unsqueeze(0)
    else:
        x = torch.linspace(-1, 1, 64).repeat(64, 1)
        y = torch.linspace(-1, 1, 64).unsqueeze(1).repeat(1, 64)
        r = torch.sqrt(x**2 + y**2)
        base = (
            torch.stack([torch.sin(r * 4.0), torch.cos(x * 3.0), torch.sin(y * 3.0)])
            .unsqueeze(0)
            .clamp(-1.0, 1.0)
        )

    steps_to_show = [0, 250, 500, 750, 999]
    diff = GaussianDiffusion(timesteps=timesteps, schedule="cosine")
    fixed_noise = torch.randn_like(base)
    noisy_steps = [
        diff.q_sample(base, torch.tensor([s]), fixed_noise).squeeze(0)
        for s in steps_to_show
    ]

    forward_path = "outputs/figures/forward_process.png"
    vutils.save_image(
        torch.stack(noisy_steps),
        forward_path,
        nrow=5,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Tira de forward process guardada en: {forward_path}")
