from __future__ import annotations

from typing import Iterator, Any, Tuple, Union, Optional, TYPE_CHECKING
from contextlib import contextmanager
from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from enfugue.diffusion.support.depth.geowizard.pipeline import DepthNormalEstimationPipeline
    import torch

__all__ = ["DepthNormalDetector"]

class MidasImageProcessor(SupportModelProcessor):
    """
    Stores the depth model and transform function
    """
    def __init__(
        self,
        model: torch.nn.Module,
        include_depth: bool=True,
        include_normal: bool=True,
        **kwargs: Any
    ) -> None:
        super(MidasImageProcessor, self).__init__(**kwargs)
        self.model = model
        self.include_depth = include_depth
        self.include_normal = include_normal

    def __call__(
        self,
        image: Image,
        include_depth: Optional[bool]=None,
        include_normal: Optional[bool]=None
    ) -> Union[Image, Tuple[Image, Image]]:
        """
        Gets the depth prediction then returns to an image
        """
        include_normal = self.include_normal if include_normal is None else include_normal
        include_depth = self.include_depth if include_depth is None else include_depth
        size = image.size
        depth, normal = self.model(
            image,
            output_type = "pil",
            image_resolution=max(size),
            depth_and_normal=True
        )
        if include_depth:
            depth = depth.resize(size)
        if include_normal:
            normal = normal.resize(size)
        if include_depth and include_normal:
            return depth, normal
        if include_depth:
            return depth
        if include_normal:
            return normal
        raise ValueError("At least one of include_depth or include_normal must be True")

class GeoWizardImageProcessor(SupportModelProcessor):
    """
    Stores the depth model and transform function
    """
    def __init__(
        self,
        model: DepthNormalEstimationPipeline,
        include_depth: bool=True,
        include_normal: bool=True,
        **kwargs: Any
    ) -> None:
        super(GeoWizardImageProcessor, self).__init__(**kwargs)
        self.model = model
        self.include_depth = include_depth
        self.include_normal = include_normal

    def __call__(
        self,
        image: Image,
        denoising_steps: int=10,
        ensemble_size: int=3,
        domain: Literal["outdoor", "indoor", "object"]="object",
        color_map: Literal["viridis", "plasma", "inferno", "magma","binary"]="binary",
        processing_resolution: int=0,
        include_depth: Optional[bool]=None,
        include_normal: Optional[bool]=None,
        seed: Optional[int]=None
    ) -> Union[Image, Tuple[Image, Image]]:
        """
        Gets the depth prediction then returns to an image
        """
        from PIL import Image
        from enfugue.diffusion.util import seed_all
        from random import randint

        if seed is None:
            seed = randint(2**16, 2**32)
        seed_all(seed)

        size = image.size
        include_normal = self.include_normal if include_normal is None else include_normal
        include_depth = self.include_depth if include_depth is None else include_depth
        output = self.model(
            image,
            denoising_steps=denoising_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_resolution,
            domain=domain,
            batch_size=0,
            color_map=color_map
        )

        depth = output.depth_colored
        normal = output.normal_colored

        if include_depth:
            depth = depth.resize(size)
        if include_normal:
            normal = normal.resize(size)
        if include_depth and include_normal:
            return depth, normal
        if include_depth:
            return depth
        if include_normal:
            return normal
        raise ValueError("At least one of include_depth or include_normal must be True")

class DepthNormalDetector(SupportModel):
    """
    Uses various models to predict depth and/or normals.
    """

    MIDAS_MODEL_TYPE = "dpt_hybrid"
    MIDAS_MODEL_URL = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
    GEOWIZARD_PATH = "lemonaddie/geowizard"
    GEOWIZARD_REVISION = "85be565717799bac9bfae5c3f82446b16c2a6b5b"

    @contextmanager
    def midas(
        self,
        include_depth: bool=True,
        include_normal: bool=True,
    ) -> Iterator[MidasImageProcessor]:
        """
        Executes MiDaS depth detection and normal estimation
        """
        import torch

        with self.context():
            from enfugue.diffusion.support.depth.midas import MidasDetector # type: ignore
            model = MidasDetector.from_pretrained(
                self.get_model_file(self.MIDAS_MODEL_URL),
                model_type=self.MIDAS_MODEL_TYPE,
            )
            model = model.to(self.device)
            processor = MidasImageProcessor(
                model,
                include_depth=include_depth,
                include_normal=include_normal
            )
            yield processor
            del model
            del processor

    @contextmanager
    def geowizard(
        self,
        include_depth: bool=True,
        include_normal: bool=True,
    ) -> Iterator[GeoWizardImageProcessor]:
        """
        Executes GeoWizard depth detection and normal estimation
        """
        from diffusers import (
            DDIMScheduler,
            AutoencoderKL
        )
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModelWithProjection
        )
        from enfugue.diffusion.support.depth.geowizard.models.unet_2d_condition import UNet2DConditionModel
        from enfugue.diffusion.support.depth.geowizard.pipeline import DepthNormalEstimationPipeline

        sd_path = "stabilityai/stable-diffusion-2-1-unclip"
        sd_variations_path = "lambdalabs/sd-image-variations-diffusers"

        with self.context():
            vae = AutoencoderKL.from_pretrained(
                sd_path,
                subfolder="vae",
                cache_dir=self.kwargs.get("vae_dir", self.model_dir),
            )
            scheduler = DDIMScheduler.from_pretrained(
                sd_path,
                subfolder="scheduler",
            )
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                sd_variations_path,
                subfolder="image_encoder",
                cache_dir=self.kwargs.get("vision_dir", self.model_dir)
            )
            feature_extractor = CLIPImageProcessor.from_pretrained(
                sd_variations_path,
                subfolder="feature_extractor",
                cache_dir=self.kwargs.get("vision_dir", self.model_dir)
            )
            unet = UNet2DConditionModel.from_pretrained(
                self.GEOWIZARD_PATH,
                cache_dir=self.model_dir,
                subfolder="unet",
                revision=self.GEOWIZARD_REVISION
            )
            pipe = DepthNormalEstimationPipeline(
                vae=vae,
                scheduler=scheduler,
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
                unet=unet,
            )

            try:
                import xformers
                xformers # silence importchecker
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass

            pipe.to(device=self.device, dtype=self.dtype)
            processor = GeoWizardImageProcessor(
                pipe,
                include_depth=include_depth,
                include_normal=include_normal
            )
            yield processor
            del pipe
            del processor
