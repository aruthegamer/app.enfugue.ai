from __future__ import annotations

from typing import List, Union, Iterator, TYPE_CHECKING
from contextlib import contextmanager

from enfugue.diffusion.support.model import SupportModel, SupportModelProcessor

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image
    from enfugue.diffusion.support.segmentation.densepose import DensePoseMaskedColormapResultsVisualizer

__all__ = ["SegmentationDetector"]

class SegmentationImageProcessor(SupportModelProcessor):
    """
    Holds a reference to the segment anything model.
    """
    def __init__(self, generator: SamAutomaticMaskGenerator) -> None:
        self.generator = generator

    @staticmethod
    def viridis(value: float) -> Tuple[int, int, int]:
        """
        Converts a value in the range [0, 1] to a viridis color.
        """
        import numpy as np
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")

        # Viridis colormap data points
        viridis_data = [
            (0.267004, 0.004874, 0.329415),
            (0.278826, 0.17549,  0.483397),
            (0.229739, 0.322361, 0.545706),
            (0.127568, 0.566949, 0.550556),
            (0.369214, 0.788888, 0.382914),
            (0.993248, 0.906157, 0.143936)
        ]

        # Map the value to the appropriate segment
        index = value * (len(viridis_data) - 1)
        lower_index = int(np.floor(index))
        upper_index = int(np.ceil(index))
        ratio = index - lower_index

        # Linear interpolation between the lower and upper color
        lower_color = np.array(viridis_data[lower_index])
        upper_color = np.array(viridis_data[upper_index])
        color = lower_color + (upper_color - lower_color) * ratio

        # Convert to 8-bit RGB values
        rgb = tuple(np.round(color * 255).astype(int))
        return rgb

    def masks(
        self,
        image: Union[str, Image],
        output_type: Literal["binary-mask", "mask", "slice", "cropped-slice"]="mask"
    ) -> List[Image]:
        """
        Calls the mask generator.
        """
        from PIL import Image
        from enfugue.diffusion.util import ComputerVision

        if isinstance(image, str):
            image = Image.open(image)

        results = self.generator.generate(ComputerVision.convert_image(image))

        if output_type == "binary-mask":
            return [result["segmentation"] for result in results]

        masks = [
            Image.fromarray(result["segmentation"])
            for result in results
        ]
        if output_type == "mask":
            return masks

        images: List[Image.Image] = []
        for mask in masks:
            blank = Image.new("RGBA", image.size)
            blank.paste(image, mask=mask)
            images.append(blank)
        if output_type == "slice":
            return images

        return [
            image.crop(image.getbbox())
            for image in images
        ]

    def __call__(
        self,
        image: Union[str, Image],
    ) -> List[Image]:
        """
        Calls the generator and merges the masks.
        """
        return self.masks(image, output_type="slice")

    def __call__(
        self,
        image: Union[str, Image],
    ) -> Image:
        """
        Calls the generator and merges the masks.
        """
        from PIL import Image
        masks = self.masks(image, output_type="mask")
        num_masks = len(masks)
        image = Image.new("RGBA", image.size)
        for i, mask in enumerate(masks):
            rgb = self.viridis(i / num_masks)
            image.paste(Image.new("RGB", image.size, rgb), mask=mask)
        return image

class DensePoseProcessor(SupportModel):
    """
    Holds a reference to the DensePose model.
    """
    def __init__(
        self,
        densepose: torch.jit.ScriptModule,
        visualizer: DensePoseMaskedColormapResultsVisualizer,
        device: torch.device
    ) -> None:
        self.densepose = densepose
        self.visualizer = visualizer
        self.device = device

    def __call__(
        self,
        image: Union[str, Image],
        detect_resolution: int=512,
        color_map: Literal["viridis", "parula"]="viridis",
    ) -> Image:
        """
        Calls the DensePose model.
        """
        import cv2
        import torch
        import numpy as np
        from PIL import Image
        from einops import rearrange
        from enfugue.util import scale_image
        from enfugue.diffusion.util import ComputerVision
        from enfugue.diffusion.support.segmentation.densepose import densepose_chart_predictor_output_to_result_with_confidences

        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        input_image = ComputerVision.convert_image(scale_image(image, nearest=64))
        H, W  = input_image.shape[:2]

        hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
        hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])

        input_image = rearrange(torch.from_numpy(input_image).to(self.device), 'h w c -> c h w')

        pred_boxes, corase_segm, fine_segm, u, v = self.densepose(input_image)

        extractor = densepose_chart_predictor_output_to_result_with_confidences
        densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

        if color_map == "viridis":
            self.visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
            hint_image = self.visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
            hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
            hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
            hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
        else:
            self.visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
            hint_image = self.result_visualizer.visualize(hint_image_canvas, densepose_results)
            hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(hint_image)
        return image.resize((width, height), Image.BICUBIC)

class SegmentationDetector(SupportModel):
    """
    Used to separate images into their constituent segments.
    """
    SAM_MODEL_PATH = "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth"
    DENSEPOSE_MODEL_PATH = "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r101_fpn_dl.torchscript"

    @property
    def segment_anything_checkpoint(self) -> str:
        """
        Gets the SAM checkpoint.
        """
        return self.get_model_file(self.SAM_MODEL_PATH)

    @property
    def densepose_checkpoint(self) -> str:
        """
        Gets the DensePose checkpoint.
        """
        return self.get_model_file(self.DENSEPOSE_MODEL_PATH)

    @contextmanager
    def densepose(
        self,
        num_part_labels: int=24
    ) -> Iterator[SegmentationImageProcessor]:
        """
        Gets the DensePose model.
        """
        import torch
        import torchvision
        from enfugue.diffusion.support.segmentation.densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr
        with self.context():
            densepose = torch.jit.load(self.densepose_checkpoint, map_location="cpu")
            densepose.to(self.device)
            visualizer = DensePoseMaskedColormapResultsVisualizer(
                alpha=1,
                data_extractor=_extract_i_from_iuvarr,
                segm_extractor=_extract_i_from_iuvarr,
                val_scale = 255.0 / num_part_labels
            )
            processor = DensePoseProcessor(
                densepose=densepose,
                visualizer=visualizer,
                device=self.device
            )
            yield processor
            del processor
            del densepose

    @contextmanager
    def sam(self, crop_n_layers: int=0) -> Iterator[SegmentationImageProcessor]:
        """
        Gets the segment anything model.
        """
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=self.segment_anything_checkpoint)
        sam = sam.to(device=self.device)
        generator = SamAutomaticMaskGenerator(
            sam,
            crop_n_layers=1,
            min_mask_region_area=1e4
        )
        processor = SegmentationImageProcessor(generator)
        yield processor
        del processor
        del generator
