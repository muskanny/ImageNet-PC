import os
import cv2
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load YOLOv8 model using Ultralytics
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        logging.info("YOLO model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        raise

# Detect objects and return bounding boxes
def detect_objects(image, model):
    try:
        results = model(image)
        bboxes, class_ids, confidences = [], [], []

        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0]
                class_id = int(box.cls[0])
                confidence = box.conf[0]

                bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
                class_ids.append(class_id)
                confidences.append(float(confidence))

        return bboxes, class_ids, confidences
    except Exception as e:
        logging.error(f"Error detecting objects: {e}")
        return [], [], []

# Save YOLO labels in YOLO format
def save_yolo_labels(image_path, bboxes, output_dir, class_ids):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Failed to read image: {image_path}")
            return

        img_h, img_w = img.shape[:2]
        label_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        with open(label_path, 'w') as file:
            for bbox, class_id in zip(bboxes, class_ids):
                x_min, y_min, x_max, y_max = bbox
                cx = (x_min + x_max) / (2 * img_w)
                cy = (y_min + y_max) / (2 * img_h)
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h
                file.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        logging.info(f"Labels saved to {label_path}")
    except Exception as e:
        logging.error(f"Error saving YOLO labels: {e}")

# Generate bounding boxes for all images in the input folder
def generate_bounding_boxes(input_folder, model_path, labels_folder):
    try:
        model = load_yolo_model(model_path)
        for class_dir in os.listdir(input_folder):
            class_path = os.path.join(input_folder, class_dir)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                label_output_dir = os.path.join(labels_folder, class_dir)
                os.makedirs(label_output_dir, exist_ok=True)

                for image_name in image_files:
                    image_path = os.path.join(class_path, image_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        logging.warning(f"Failed to read image: {image_name}, skipping...")
                        continue

                    bboxes, class_ids, confidences = detect_objects(image, model)
                    if not bboxes:
                        logging.info(f"No bounding boxes found in {image_name}, skipping...")
                        continue

                    save_yolo_labels(image_path, bboxes, label_output_dir, class_ids)
                    logging.info(f"Processed image: {image_name}")

        logging.info("Bounding box generation complete!")
    except Exception as e:
        logging.error(f"Error during bounding box generation: {e}")

# Command-line script entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        logging.error("Usage: python apply_bounding_boxes.py <input_folder> <model_path> <labels_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    model_path = sys.argv[2]
    labels_folder = sys.argv[3]

    generate_bounding_boxes(input_folder, model_path, labels_folder)
