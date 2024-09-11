from ultralytics import SAM
from PIL import Image
import os
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import supervision as sv


class ImageProcessor:

    def __init__(self):
        Image.MAX_IMAGE_PIXELS = None  # Set to None to remove the limit

    @staticmethod
    def side_by_side_comparison(original_image: Image.Image, segmented_image: Image.Image) -> Image.Image:
        if original_image.size != segmented_image.size:
            segmented_image = segmented_image.resize(original_image.size)

        # Create a new image with double the width of the original (side by side)
        new_width = original_image.width + segmented_image.width
        new_height = original_image.height
        comparison_image = Image.new('RGB', (new_width, new_height))

        # Paste the original image on the left
        comparison_image.paste(original_image, (0, 0))

        # Paste the segmented image on the right
        comparison_image.paste(segmented_image, (original_image.width, 0))

        return comparison_image


class TiffImageHandler:

    def __init__(self, input_file: str, output_dir: str, tile_size: int = 1024, dpi: int = 600):
        self.input_file = input_file
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.dpi = dpi

    def convert_tiff_to_png(self):
        img = Image.open(self.input_file)
        width, height = img.size

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                # Define the box to crop
                box = (i, j, min(i + self.tile_size, width), min(j + self.tile_size, height))

                # Crop the tile and save it
                tile = img.crop(box)
                tile_path = os.path.join(self.output_dir, f'tile_{i}_{j}.png')
                tile.save(tile_path, format='PNG')
                print(f'Saved {tile_path}')

    def split_images(self):
        img = Image.open(self.input_file)
        os.makedirs(self.output_dir, exist_ok=True)

        num_frames = getattr(img, 'n_frames', 1)  # Number of frames in the TIFF image

        for frame in range(num_frames):
            img.seek(frame)
            img_rgb = img.convert('RGB')
            output_path = os.path.join(self.output_dir, f'image_{frame + 1}.png')
            img_rgb.save(output_path, dpi=(self.dpi, self.dpi))
            print(f'Saved {output_path}')


class ImageSegmentation:

    def __init__(self, model_path: str, model_type: str, image_path: str):
        self.model_checkpoint = model_path
        self.imagePath = image_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_type == "vit_h":
            self.model_vit = self.load_model_vit(self.model_checkpoint)
        else:
            self.model_sam = self.load_model_sam(self.model_checkpoint)

    def load_model_vit(self, model_checkpoint):
        model_type = "vit_h"
        self.model_checkpoint = model_checkpoint
        Sam = sam_model_registry[model_type](checkpoint=self.model_checkpoint)
        Sam.to(device=self.device)
        return Sam

    def load_model_sam(self, model_checkpoint):
        # Load a model
        self.model_checkpoint = model_checkpoint
        sam_model = SAM(model_checkpoint)
        return sam_model

    def generate_masks(self, image_path: str):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_generator = SamAutomaticMaskGenerator(self.model_vit)
        result = mask_generator.generate(image_rgb)
        return result, image_bgr

    @staticmethod
    def apply_segmentation_masks(original_image: Image.Image, masks: torch.Tensor) -> Image.Image:
        if masks.is_cuda:
            masks = masks.cpu()

        masks_np = masks.numpy()
        colored_mask = np.zeros((masks_np.shape[1], masks_np.shape[2], 3), dtype=np.uint8)

        # Generate a color map
        colors = np.random.randint(0, 255, size=(masks_np.shape[0], 3), dtype=np.uint8)

        for i in range(masks_np.shape[0]):
            colored_mask[masks_np[i] > 0] = colors[i]

        original_image = original_image.convert("RGB")
        mask_image = Image.fromarray(colored_mask)
        blended_image = Image.blend(original_image, mask_image, alpha=0.5)

        return blended_image

    def segment_and_annotate_vit(self):
        results, image_bgr = self.generate_masks(self.imagePath)

        # Create Detections from SAM results with a default class_id
        detections = sv.Detections.from_sam(results)
        detections.class_id = [0] * len(detections)

        # Annotate the image
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(image_bgr, detections)

        # Save the annotated image
        cv2.imwrite("annotated_image.png", annotated_image)
        print("Annotated image saved successfully.")
        return annotated_image

    def segment_SAM(self):

        # Run inference
        results = self.model_sam.predict(self.imagePath, save=True)
        for result in results:
            print(f"Detected - {len(result.masks)} masks , {dir(result.masks)}")
            original_image = Image.fromarray(result.orig_img)  # Convert numpy array to PIL image
            blended_image = self.apply_segmentation_masks(original_image, result.masks.data)
            blended_image.save("segmented_overlay.png")
            return blended_image


def main():
    # Example usage of TiffImageHandler
    # tiff_handler = TiffImageHandler(
    #     input_file='Sample_Village/581304_PANASPETA_SANTHAKAVITI_VIZIANAGARAM/581304/581304_PANASPETA_GTIFF.tif',
    #     output_dir='output_images'
    # )
    # tiff_handler.split_images()

    # Example usage of ImageSegmentation
    # segmentation_handler = ImageSegmentation(model_checkpoint="./sam_vit_h_4b8939.pth")
    # segmentation_handler.segment_and_annotate_vit()

    sam = ImageSegmentation(model_path="Models/sam_l.pt", model_type='', image_path="input/1.png")
    sam.segment_SAM()


# if __name__ == "__main__":
#     main()