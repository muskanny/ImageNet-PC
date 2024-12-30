
ImageNet-PC - Project Overview
===============================

This project aims to apply partial corruptions to the ImageNet dataset, perform inference using various Convolutional Neural Network (CNN) and Vision Transformer (ViT) models, and evaluate the robustness of these models using the Learned Perceptual Image Patch Similarity (LPIPS) metric.

## Project Structure

The directory structure for this project is as follows:
'''
your_project/
├── data/
│   ├── raw/                   # Original datasets (e.g., ImageNet images)
│   ├── processed/             # Processed datasets (e.g., images with bounding boxes and corruptions)
│   └── annotations/           # Annotations such as bounding boxes for images
├── scripts/
│   ├── apply_bounding_boxes.py  # Apply bounding boxes to the dataset
│   ├── apply_corruptions.py     # Apply partial corruptions to images
│   ├── inference.py             # Run inference using pre-trained models
│   ├── lpips_evaluation.py      # Calculate LPIPS scores for image similarity
│   └── clip_evaluate.py         # Optional: Evaluate with CLIP model
├── requirements.txt              # List of dependencies required for the project
├── setup.py                      # Setup script for the package
└── README.txt                    # This file
'''
## Prerequisites

Before running the pipeline, make sure to have the following prerequisites:

1. Python 3.x installed on your system.
2. Clone the repository to your local machine.

### Steps to Clone the Repository:

```bash
git clone https://github.com/yourusername/ImageNet-PC.git
cd ImageNet-PC
```

3. Install dependencies from `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

Alternatively, you can use `setup.py` for package installation:

```bash
python setup.py install
```

### Data Preparation:

- **Raw Dataset**: Download and place the raw ImageNet dataset in the `data/raw/` directory.
- **Annotations**: Ensure that bounding box annotations for the dataset are in the `data/annotations/` directory.

## Running the Pipeline

Once the data and environment are set up, the main pipeline can be run using the `main.py` script. The pipeline involves the following steps:

1. **Apply Bounding Boxes**: This step adds bounding boxes around objects in the images based on predefined annotations.
2. **Apply Corruptions**: This step applies partial corruptions (e.g., lens flare, occlusion) to the images within the bounding boxes.
3. **Inference**: Perform inference on the corrupted images using pre-trained models such as ResNet, MobileNet, and Vision Transformers.
4. **LPIPS Evaluation**: Calculate the LPIPS score to assess perceptual differences between the original and corrupted images.

### Running the Pipeline:

To execute the entire pipeline:

```bash
python main.py
```

This will sequentially run the following processes:
- Apply bounding boxes
- Apply corruptions
- Run inference on the corrupted images
- Calculate LPIPS scores

### Running Individual Scripts:

If you prefer to run specific parts of the pipeline, you can execute each script individually:

- Apply bounding boxes:
  ```bash
  python scripts/apply_bounding_boxes.py
  ```

- Apply corruptions:
  ```bash
  python scripts/apply_corruptions.py
  ```

- Run inference:
  ```bash
  python scripts/inference.py
  ```

- Evaluate LPIPS scores:
  ```bash
  python scripts/lpips_evaluation.py
  ```

### Results:

After running the pipeline, the results will be saved in the `results/` directory. The results include:
- Inference results (predictions) from various models.
- LPIPS evaluation results for the robustness of the models to corruptions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the authors of the ImageNet dataset.
- This project uses several pre-trained models available from public repositories.
- Special thanks to the authors of the LPIPS metric for their work on perceptual similarity.


### Explanation of the Sections:
1. **Project Structure**: Clearly lists the files and directories involved in the project.
2. **Prerequisites**: Instructions to set up the environment and clone the repository.
3. **Data Preparation**: Where and how to place the dataset and annotations.
4. **Running the Pipeline**: Instructions to run the full pipeline with the `main.py` script or individual components.
5. **Results**: Where results will be saved and what they will include.
6. **License & Acknowledgments**: Licensing information and acknowledgments.
'''