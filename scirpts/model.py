import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import default, extract


class BrownianBridge(nn.Module):
    """
    Adapted from the BBDM reference implementation:
    https://github.com/xuekt98/BBDM (Li et al., 2023).
    """

    def __init__(
        self,
        denoise_fn,
        num_timesteps=1000,
        mt_type="linear",
        max_var=1.0,
        eta=1.0,
        objective="grad",
        loss_type="l2",
        channels=None,
        nll_min_var=1.0e-6,
        skip_sample=False,
        sample_type="linear",
        sample_step=200,
    ):
        super().__init__()
        objective = str(objective).lower()
        allowed_objectives = {"eps", "grad", "maxnll"}
        if objective not in allowed_objectives:
            raise ValueError(f"objective must be one of {sorted(allowed_objectives)}, got: {objective}")
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(num_timesteps)
        self.mt_type = mt_type
        self.max_var = float(max_var)
        self.eta = float(eta)
        self.objective = objective
        self.loss_type = loss_type
        self.channels = channels
        self.nll_min_var = float(nll_min_var)
        self.skip_sample = bool(skip_sample)
        self.sample_type = sample_type
        self.sample_step = int(sample_step)
        self.steps = None
        self._register_schedule()

    def _register_schedule(self):
        T = self.num_timesteps
        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = torch.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** torch.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError(f"Unknown mt_type: {self.mt_type}")

        m_tminus = torch.cat([torch.zeros(1), m_t[:-1]])
        variance_t = 2.0 * (m_t - m_t**2) * self.max_var
        variance_tminus = torch.cat([torch.zeros(1), variance_t[:-1]])
        variance_t_tminus = variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        self.register_buffer("m_t", m_t)
        self.register_buffer("m_tminus", m_tminus)
        self.register_buffer("variance_t", variance_t)
        self.register_buffer("variance_tminus", variance_tminus)
        self.register_buffer("variance_t_tminus", variance_t_tminus)
        self.register_buffer("posterior_variance_t", posterior_variance_t)

        if self.skip_sample:
            if self.sample_type == "linear":
                midsteps = torch.arange(
                    self.num_timesteps - 1, 1, step=-((self.num_timesteps - 1) / (self.sample_step - 2))
                ).long()
                self.steps = torch.cat((midsteps, torch.tensor([1, 0]).long()), dim=0)
            elif self.sample_type == "cosine":
                steps = torch.linspace(0, self.num_timesteps, self.sample_step + 1)
                steps = (torch.cos(steps / self.num_timesteps * torch.pi) + 1.0) / 2.0 * self.num_timesteps
                self.steps = steps.long()
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)

    def _format_direction(self, direction, batch, device):
        if direction is None:
            return None
        direction = torch.as_tensor(direction, device=device, dtype=torch.long)
        if direction.dim() == 0:
            direction = direction.expand(batch)
        elif direction.dim() == 1 and direction.shape[0] != batch:
            direction = direction.expand(batch)
        return direction

    def _select_head(self, pred, direction, channels):
        if direction is None:
            return pred
        if pred.shape[1] != channels * 2:
            return pred
        direction = self._format_direction(direction, pred.shape[0], pred.device)
        pred0 = pred[:, :channels]
        pred1 = pred[:, channels:]
        if direction.numel() == 1 or torch.all(direction == direction[0]):
            return pred0 if int(direction[0].item()) == 0 else pred1
        mask = direction.view(-1, 1, 1, 1).float()
        return pred0 * (1.0 - mask) + pred1 * mask

    def _build_denoise_input(self, x_t, source, target, direction):
        if direction is None:
            return torch.cat([x_t, target], dim=1)
        if source is None or target is None:
            raise ValueError("source/target must be provided when using direction conditioning.")
        direction = self._format_direction(direction, x_t.shape[0], x_t.device)
        mask = direction.view(-1, 1, 1, 1).float()
        src = torch.where(mask == 0.0, torch.zeros_like(source), source)
        tgt = torch.where(mask == 1.0, torch.zeros_like(target), target)
        return torch.cat([x_t, src, tgt], dim=1)

    def _denoise(self, x_t, y, t, source=None, target=None, direction=None):
        if direction is None:
            denoise_in = torch.cat([x_t, y], dim=1)
            return self.denoise_fn(denoise_in, t)
        denoise_in = self._build_denoise_input(x_t, source, target, direction)
        pred = self.denoise_fn(denoise_in, t, direction=direction)
        channels = self.channels or x_t.shape[1]
        return self._select_head(pred, direction, channels)


    def forward(self, x0, y, source=None, target=None, direction=None):
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device).long()
        loss, log_dict = self.p_losses(x0, y, t, source=source, target=target, direction=direction)
        return loss, log_dict

    def p_losses(self, x0, y, t, noise=None, source=None, target=None, direction=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        x_t, objective = self.q_sample(x0, y, t, noise)
        if self.objective == "maxnll":
            var_t = extract(self.variance_t, t, x0.shape)
            objective_recon = self._denoise(x_t, y, t, source=source, target=target, direction=direction)*var_t
            objective = objective*var_t
        else:
            objective_recon = self._denoise(x_t, y, t, source=source, target=target, direction=direction)
        loss_type = self.loss_type
        if loss_type == "mse":
            loss_type = "l2"

        if loss_type == "l1":
            recloss = (objective - objective_recon).abs().mean()
        elif loss_type == "l2":
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError(f"Unknown loss_type: {self.loss_type}")

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        return recloss, {"loss": recloss, "x0_recon": x0_recon}

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == "grad":
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == "eps":
            objective = noise
        elif self.objective == "maxnll":
            objective = noise
        else:
            raise NotImplementedError(f"Unknown objective: {self.objective}")

        x_t = (1.0 - m_t) * x0 + m_t * y + sigma_t * noise
        return x_t, objective

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == "grad":
            x0_recon = x_t - objective_recon
        elif self.objective in {"maxnll", "eps"}:
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1.0 - m_t)
        else:
            raise NotImplementedError(f"Unknown objective: {self.objective}")
        return x0_recon

    @torch.no_grad()
    def p_sample(self, x_t, y, i, clip_denoised=False, source=None, target=None, direction=None):
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self._denoise(x_t, y, t, source=source, target=target, direction=direction)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
            if clip_denoised:
                x0_recon = x0_recon.clamp(-1.0, 1.0)
            return x0_recon, x0_recon

        t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
        n_t = torch.full((x_t.shape[0],), self.steps[i + 1], device=x_t.device, dtype=torch.long)

        objective_recon = self._denoise(x_t, y, t, source=source, target=target, direction=direction)
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        if clip_denoised:
            x0_recon = x0_recon.clamp(-1.0, 1.0)

        m_t = extract(self.m_t, t, x_t.shape)
        m_nt = extract(self.m_t, n_t, x_t.shape)
        var_t = extract(self.variance_t, t, x_t.shape)
        var_nt = extract(self.variance_t, n_t, x_t.shape)

        sigma2_t = (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t) * self.eta

        noise = torch.randn_like(x_t)
        mean = (1.0 - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * (
            x_t - (1.0 - m_t) * x0_recon - m_t * y
        )

        return mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def sample(self, y=None, clip_denoised=True, source=None, target=None, direction=None):
        if direction is not None:
            if source is None or target is None:
                raise ValueError("source/target must be provided when using direction sampling.")
            if y is None:
                direction = self._format_direction(direction, source.shape[0], source.device)
                if direction.numel() == 1 or torch.all(direction == direction[0]):
                    y = target if int(direction[0].item()) == 0 else source
                else:
                    mask = direction.view(-1, 1, 1, 1).float()
                    y = torch.where(mask == 0.0, target, source)
        if y is None:
            raise ValueError("y must be provided for sampling.")

        img = y
        for i in range(len(self.steps)):
            img, _ = self.p_sample(
                img,
                y,
                i,
                clip_denoised=clip_denoised,
                source=source,
                target=target,
                direction=direction,
            )
        return img
