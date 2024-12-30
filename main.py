import os
import torch
from scripts.apply_bounding_boxes import apply_bounding_boxes
from scripts.apply_corruptions import apply_corruptions
from scripts.inference import run_inference
from scripts.lpips_evaluation import calculate_lpips
from scripts.clip_evaluate import evaluate_clip
from pathlib import Path

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths to your datasets and results directory
raw_data_path = 'data/raw'  # Path to the raw dataset (ImageNet)
processed_data_path = 'data/processed'  # Path to the processed dataset after applying bounding boxes and corruptions
annotations_path = 'data/annotations'  # Path to the bounding box annotations
results_path = 'results'  # Directory to save the final results

# Ensure the results directory exists
os.makedirs(results_path, exist_ok=True)

def preprocess_dataset():
    """
    Preprocess the dataset: Apply bounding boxes and save the processed dataset.
    """
    print("Starting to apply bounding boxes to the dataset...")
    apply_bounding_boxes(raw_data_path, annotations_path, processed_data_path)
    print("Bounding boxes applied successfully.")

def apply_corruptions_to_dataset():
    """
    Apply corruptions to the images, keeping the bounding boxes intact.
    """
    print("Starting to apply corruptions to the processed images...")
    apply_corruptions(processed_data_path, results_path)
    print("Corruptions applied and saved successfully.")

def run_inference_on_processed_images():
    """
    Run inference on the processed images using CNN and Vision Transformer models.
    """
    print("Starting inference on the processed and corrupted images...")
    run_inference(results_path, device)
    print("Inference completed successfully.")

def compute_lpips():
    """
    Compute LPIPS scores between original and corrupted images.
    """
    print("Starting to compute LPIPS scores...")
    lpips_scores = calculate_lpips(raw_data_path, processed_data_path, results_path, device)
    
    print("LPIPS scores computed:")
    # Print the LPIPS results for each severity level
    for severity, score in lpips_scores.items():
        print(f"Average LPIPS score for Severity {severity}: {score}")

def evaluate_clip_model():
    """
    Evaluate the results using the CLIP model (optional).
    """
    print("Starting CLIP evaluation...")
    evaluate_clip(results_path, device)
    print("CLIP evaluation completed successfully.")

def main():
    """
    Main function to run the whole process: Preprocessing, Corruptions, Inference, LPIPS, CLIP.
    """
    # Step 1: Preprocess the dataset by applying bounding boxes
    preprocess_dataset()

    # Step 2: Apply corruptions to the dataset
    apply_corruptions_to_dataset()

    # Step 3: Run inference on the corrupted images
    run_inference_on_processed_images()

    # Step 4: Compute LPIPS scores
    compute_lpips()

    # Step 5: (Optional) Evaluate using CLIP
    evaluate_clip_model()

if __name__ == "__main__":
    main()
