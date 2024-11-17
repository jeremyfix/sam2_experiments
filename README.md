# Experiments using SAM2 and box prompting

A repository to test SAM2 with bounding box prompting.

## Dependencies

```.python
python3 -m venv venv
source venv/bin/activate
python -m pip install sam2 huggingface_hub
```

## Usage 

To use this script with Biigle annotations, here is the way to proceed.

1. Download the annotations from Biigle as "Reports / Image annotation report / CSV", let us call it `labels.csv`
2. Place your images into the `images/` directory
3. Run the script `python infer.py images labels.csv hiera-large`

The last argument specifies the model to be used. You can test others such as hiera-tiny for example.
