# coding: utf-8

# Standard imports
import argparse
import logging

# External imports
import torch
import pandas as pd
from sam2.sam2_image_predictor import SAM2ImagePredictor


def infer_on_image(modelname, img_path, box_prompts):
    """
    Infer on an image using the SAM2 model

    Args:
        modelname (str): Name of the model to use (e.g. hiera-large)
        img_path (str): Path to the image to infer on

    """
    modelpath = f"facebook/sam2-{modelname}"
    predictor = SAM2ImagePredictor.from_pretrained(modelpath)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        point_coords = None
        point_labels = None
        box = None
        multimask_output = False
        predictor.set_image(img_path)
        masks, scores, _ = predictor.predict(
            point_coords, point_labels, box, multimask_output
        )

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

    args = parser.parse_args()
    img_directory = args.img_directory
    prompts_path = args.prompts_path
    modelname = args.modelname

    prompts = parse_labels(prompts_path)

    # Perform inference on every image
    for img_name, img_prompts in prompts.items():
        img_path = f"{img_directory}/{img_name}"
        logging.info(f"Infering on {img_path}")
        infer_on_image(modelname, img_path, img_prompts)
