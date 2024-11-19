# coding: utf-8

# Standard imports
import argparse
import logging
import os

# External imports
import torch
import pandas as pd
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
import PIL.Image as Image
import matplotlib.pyplot as plt


# The show_mask functions comes from the SAM2 repository
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        logging.info(f"Mask shape : {mask.shape}")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


# The show_box functions comes from the SAM2 repository
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def infer_on_image(modelname, img_path, box_prompts, output_path):
    """
    Infer on an image using the SAM2 model

    Args:
        modelname (str): Name of the model to use (e.g. hiera-large)
        img_path (str): Path to the image to infer on

    """
    modelpath = f"facebook/sam2-{modelname}"
    predictor = SAM2ImagePredictor.from_pretrained(modelpath)

    img = Image.open(img_path)
    img = np.array(img.convert("RGB"))
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    point_coords = None
    point_labels = None
    boxes = box_prompts
    multimask_output = False

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            multimask_output=multimask_output,
        )
        if len(masks.shape) == 3:
            masks = masks[np.newaxis, ...]

        # Save the results on disk
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, (box, mask) in enumerate(zip(boxes, masks)):
            plt.figure()
            plt.imshow(img)
            show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            show_box(box, plt.gca())
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path + f"mask_{i}.png")
            plt.close()

    return masks


def parse_prompts(shape_name, shape_points):
    if shape_name == "Rectangle":
        # The points are (x1, y1, x2, y2, x3, y3, x4, y4)
        # a list in a string
        box_points = eval(shape_points)
        x_coords = box_points[::2]
        y_coords = box_points[1::2]

        # We must build an axis aligned box for SAM2
        # with two points : top left and bottom right
        x1 = min(x_coords)
        x2 = max(x_coords)
        y1 = min(y_coords)
        y2 = max(y_coords)
        return [x1, y1, x2, y2]
    else:
        raise NotImplementedError("Only rectangles are supported for now")


def parse_labels(labels_path):
    labels = pd.read_csv(labels_path)

    # Get the filenames as a list
    filenames = labels.filename.tolist()
    shapes_names = labels.shape_name.tolist()
    shape_points = labels.points.tolist()
    shape_prompts = [
        parse_prompts(shape_name, shape_points)
        for shape_name, shape_points in zip(shapes_names, shape_points)
    ]

    # Group the labels by filename
    grouped_labels = {}
    for filename, shape_prompt in zip(filenames, shape_prompts):
        if filename not in grouped_labels:
            grouped_labels[filename] = []
        grouped_labels[filename].append(shape_prompt)

    return grouped_labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Infer on an image using SAM2")
    parser.add_argument(
        "img_directory",
        type=str,
        help="Path to the image directory to infer on",
    )
    parser.add_argument(
        "prompts_path",
        type=str,
        help="Path to the prompts to use for inference. Expected to be as a Biigle export format (Image annotation report / CSV)",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        help="Name of the model to use (e.g. hiera-large",
        default="hiera-large",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the output directory",
        default="output",
    )

    args = parser.parse_args()
    img_directory = args.img_directory
    prompts_path = args.prompts_path
    modelname = args.modelname

    prompts = parse_labels(prompts_path)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Perform inference on every image
    for img_name, img_prompts in prompts.items():
        img_path = f"{img_directory}/{img_name}"
        output_path = f"{args.output_directory}/{img_name}/"
        logging.info(f"Infering on {img_path}")
        infer_on_image(modelname, img_path, img_prompts, output_path)
