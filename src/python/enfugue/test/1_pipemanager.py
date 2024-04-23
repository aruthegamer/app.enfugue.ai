"""
Uses the pipemanager to create a simple image using default settings
"""
import os
from enfugue.diffusion.manager import DiffusionPipelineManager
from pibble.util.log import DebugUnifiedLoggingContext

SEED = 42

def main() -> None:
    with DebugUnifiedLoggingContext():
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        kwargs = {"prompt": "A happy looking puppy", "guidance_scale": 0.0}

        # No guidance
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-none.png"))

        # Classifier-Free Guidance
        kwargs["guidance_scale"] = 6.5
        kwargs["negative_prompt"] = "poor quality, blurry"
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-cfg.png"))

        # Classifier-Free Guidance + Rescale
        kwargs["guidance_rescale"] = 0.7
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-cfg-scale.png"))

        # Perturbed Self-Attention Guidance (Adversarial Guidance)
        kwargs["guidance_scale"] = 0.0
        kwargs["guidance_rescale"] = 0.0
        kwargs["pag_scale"] = 5.0
        kwargs["pag_applied_layers_index"] = ["m0"] # First middle layer
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-pag.png"))

        # Classifier-Free Guidance + Perturbed Self-Attention Guidance
        kwargs["pag_scale"] = 3.0
        kwargs["guidance_scale"] = 4.0
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-cfg-pag.png"))

        # Classifier-Free Guidance + Perturbed Self-Attention Guidance + Rescale
        kwargs["guidance_rescale"] = 0.7
        manager.seed = SEED
        manager(**kwargs)["images"][0].save(os.path.join(save_dir, "puppy-cfg-pag-scale.png"))

if __name__ == "__main__":
    main()
