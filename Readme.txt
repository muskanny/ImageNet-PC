# Image Augmentation Pipeline with Corruptions

This repository provides a pipeline for applying image augmentations and corruptions to a dataset. The pipeline includes the following **seven corruption types**:

1. **Lens Flare**: Adds lens flare artifacts to the image. Severity increases the intensity and size of the flare.
2. **Occlusion**: Randomly occludes a part of the image. Severity increases the size and randomness of the occlusion.
3. **Object Focus Shift**: Applies a Gaussian blur to regions of interest (bounding boxes) in the image. Severity increases the blur intensity.
4. **Illumination Variation**: Alters the brightness and contrast of the image. Severity increases the variation in illumination.
5. **Fingerprint Variation**: Simulates fingerprints on the lens, creating smudge-like distortions. Severity increases the size and opacity of the fingerprint.
6. **Dust and Scratch**: Simulates dust or scratches on the lens. Severity increases the number and size of the dust/scratch marks.
7. **Camouflage Variation**: Alters the color and texture to simulate camouflage patterns. Severity increases the intensity and coverage of the camouflage effect.

Each corruption has **5 severity levels**, ranging from 1 (least severe) to 5 (most severe). Higher severity levels cause more significant distortions or augmentations to the images.

## Setup

1. **Clone the repository** or download the code and dataset.
2. **Install required dependencies**:
    ```bash
    pip install opencv-python albumentations numpy
    ```

3. **Dataset Structure**:
    - Place your dataset in the following structure:
    ```
    dataset/
    ├── class_1/
    │   ├── image_1.jpg
    │   └── image_2.jpg
    ├── class_2/
    │   ├── image_1.jpg
    │   └── image_2.jpg
    └── ...
    ```

4. **Annotation Files**:
    - The annotations for each image should be in a separate text file, with each line in the format:
    ```
    class_id x_center y_center width height
    ```

5. **Corruption Output**:
    - The augmented images with corruptions will be saved in the following structure:
    ```
    corruptions/
    ├── lens_flare/
    │   ├── severity_1/
    │   │   ├── class_1/
    │   │   └── class_2/
    │   ├── severity_2/
    │   └── ...
    ├── occlusion/
    │   ├── severity_1/
    │   └── ...
    ├── object_focus_shift/
    ├── illumination_variation/
    ├── fingerprint_variation/
    ├── dust_and_scratch/
    └── camouflage_variation/
    ```

## How to Run

1. Set the `input_dir` and `labels_dir` to point to the dataset and annotation directories, respectively.
2. Run the pipeline:
    ```bash
    python augmentations_pipeline.py
    ```

This will process all the images in your dataset, apply the seven corruptions with 5 different severity levels, and save the augmented images in the appropriate folders.

## Corruption Types and Severity Levels:

- **Lens Flare**: Adds lens flare artifacts to the image.
    - Severity 1: Mild flare
    - Severity 2: Light flare
    - Severity 3: Medium flare
    - Severity 4: Strong flare
    - Severity 5: Extreme flare

- **Occlusion**: Randomly occludes a part of the image.
    - Severity 1: Small occlusion
    - Severity 2: Moderate occlusion
    - Severity 3: Large occlusion
    - Severity 4: Very large occlusion
    - Severity 5: Extreme occlusion

- **Object Focus Shift**: Applies Gaussian blur to bounding box regions.
    - Severity 1: Mild blur
    - Severity 2: Medium blur
    - Severity 3: Strong blur
    - Severity 4: Very strong blur
    - Severity 5: Extreme blur

- **Illumination Variation**: Alters brightness and contrast.
    - Severity 1: Slight variation
    - Severity 2: Moderate variation
    - Severity 3: Strong variation
    - Severity 4: Very strong variation
    - Severity 5: Extreme variation

- **Fingerprint Variation**: Simulates fingerprints on the lens.
    - Severity 1: Small fingerprint smudges
    - Severity 2: Moderate fingerprint smudges
    - Severity 3: Strong smudges
    - Severity 4: Very strong smudges
    - Severity 5: Extreme smudges

- **Dust and Scratch**: Simulates dust or scratches on the lens.
    - Severity 1: Light dust/scratches
    - Severity 2: Moderate dust/scratches
    - Severity 3: Strong dust/scratches
    - Severity 4: Very strong dust/scratches
    - Severity 5: Extreme dust/scratches

- **Camouflage Variation**: Alters the color and texture to simulate camouflage.
    - Severity 1: Light camouflage pattern
    - Severity 2: Moderate camouflage pattern
    - Severity 3: Strong camouflage pattern
    - Severity 4: Very strong camouflage pattern
    - Severity 5: Extreme camouflage pattern
