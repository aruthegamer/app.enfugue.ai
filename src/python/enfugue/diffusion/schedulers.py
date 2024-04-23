import torch
import numpy as np

from typing import Optional, List

from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver

__all__ = ["DPMSolverMultistepScheduler"]


# Add support for setting custom timesteps
class DPMSolverMultistepScheduler(DefaultDPMSolver):
    def set_timesteps(
        self,
        num_inference_steps: Optional[int]=None,
        device: Optional[torch.device]=None,
        timesteps: Optional[List[int]]=None
    ) -> None:
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return

        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(
            device=device, dtype=torch.int64
        )  # Ignore the last 0

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
