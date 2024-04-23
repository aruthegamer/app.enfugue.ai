import io
import os
import requests

from PIL import Image

from enfugue.diffusion.manager import DiffusionPipelineManager

from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    with DebugUnifiedLoggingContext():
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images")
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "segmentation-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        image = Image.open(os.path.join(input_dir, "one.png"))
        image.save(os.path.join(save_dir, "base.png"))

        with manager.segmentation_detector.sam() as sam:
            segmentation = sam(image)
            segmentation.save(os.path.join(save_dir, "sam.png"))
        with manager.segmentation_detector.densepose() as densepose:
            segmentation = densepose(image)
            segmentation.save(os.path.join(save_dir, "densepose.png"))

if __name__ == "__main__":
    main()
