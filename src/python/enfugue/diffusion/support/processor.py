from __future__ import annotations

from typing import Iterator, Tuple, Callable, Optional, TYPE_CHECKING

from functools import cached_property
from contextlib import contextmanager, ExitStack

from enfugue.diffusion.constants import CONTROLNET_LITERAL
from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image
    from enfugue.diffusion.support.depth import DepthNormalDetector
    from enfugue.diffusion.support.normal import NormalDetector
    from enfugue.diffusion.support.edge import EdgeDetector
    from enfugue.diffusion.support.line import LineDetector
    from enfugue.diffusion.support.pose import PoseDetector

class PassThroughImageProcessor(SupportModelProcessor):
    """
    Does not process an image.
    """
    def __call__(self, image: Image) -> Image:
        return image

class ControlImageProcessor(SupportModel):
    """
    Amalgamates all controlnet processors.
    Allows multiple contexts at once
    """

    @contextmanager
    def processors(self, *controlnets: CONTROLNET_LITERAL) -> Iterator[Tuple[SupportModelProcessor, ...]]:
        """
        Gets any number of controlnet processors in context.
        """
        with ExitStack() as stack:
            uniques = set(controlnets)
            processors = dict([
                (
                    controlnet,
                    stack.enter_context(self.processor(controlnet))
                )
                for controlnet in uniques
            ])
            yield tuple([
                processors[controlnet]
                for controlnet in controlnets
            ])

    @contextmanager
    def processor(self, controlnet: CONTROLNET_LITERAL) -> Iterator[SupportModelProcessor]:
        """
        Gets one controlnet processor in context.
        """
        context: Callable
        kwargs: Dict[str, Any] = {}
        if controlnet == "canny":
            context = self.edge_detector.canny
        elif controlnet == "pidi":
            context = self.edge_detector.pidi
        elif controlnet == "hed":
            context = self.edge_detector.hed
        elif controlnet in ["scribble", "sparse-scribble"]:
            context = self.edge_detector.scribble
        elif controlnet == "depth":
            context = self.depth_normal_detector.midas
            kwargs["include_normal"] = False
        elif controlnet == "normal":
            context = self.depth_normal_detector.midas
            kwargs["include_depth"] = False
        elif controlnet == "pose":
            context = self.pose_detector.best
        elif controlnet == "line":
            context = self.line_detector.lineart
        elif controlnet == "anime":
            context = self.line_detector.anime
        elif controlnet == "mlsd":
            context = self.line_detector.mlsd
        else:
            context = PassThroughImageProcessor
        with context(**kwargs) as processor:
            yield processor

    @cached_property
    def edge_detector(self) -> EdgeDetector:
        """
        Gets the edge detector.
        """
        from enfugue.diffusion.support.edge import EdgeDetector
        return EdgeDetector.clone(self)

    @cached_property
    def line_detector(self) -> LineDetector:
        """
        Gets the line detector.
        """
        from enfugue.diffusion.support.line import LineDetector
        return LineDetector.clone(self)

    @cached_property
    def depth_normal_detector(self) -> DepthNormalDetector:
        """
        Gets the depth detector.
        """
        from enfugue.diffusion.support.depth import DepthNormalDetector
        return DepthNormalDetector.clone(self)

    @cached_property
    def pose_detector(self) -> PoseDetector:
        """
        Gets the pose detector.
        """
        from enfugue.diffusion.support.pose import PoseDetector
        return PoseDetector.clone(self)

    def __call__(self, controlnet: CONTROLNET_LITERAL, image: Image) -> Image:
        """
        A shorthand for executing with a single processor.
        """
        with self.processor(controlnet) as process:
            return process(image)
