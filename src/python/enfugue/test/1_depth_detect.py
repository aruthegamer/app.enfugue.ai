import io
import os
import requests

from PIL import Image

from enfugue.diffusion.manager import DiffusionPipelineManager

from pibble.util.log import DebugUnifiedLoggingContext

def main() -> None:
    with DebugUnifiedLoggingContext():
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-images")
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-results", "depth-detection")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        manager = DiffusionPipelineManager()
        image = Image.open(os.path.join(input_dir, "one.png"))
        image.save(os.path.join(save_dir, "base.png"))

        with manager.control_image_processor.depth_normal_detector.geowizard() as geowizard:
            depth, normal = geowizard(image)
            depth.convert("L").save(os.path.join(save_dir, "geowizard-depth.png"))
            normal.save(os.path.join(save_dir, "geowizard-normal.png"))

        with manager.control_image_processor.depth_normal_detector.midas() as midas:
            depth, normal = midas(image)
            depth.save(os.path.join(save_dir, "midas-depth.png"))
            normal.save(os.path.join(save_dir, "midas-normal.png"))

if __name__ == "__main__":
    main()
