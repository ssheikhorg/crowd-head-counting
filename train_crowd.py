import os
import cv2
import numpy as np
from ultralytics import YOLO
import json


class CrowdAnalyzer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.person_counts = []
        self.image_paths = []

    def process_directory(self, image_dir, max_images=None):
        """Process all images in a directory and count persons"""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f)) and
                            os.path.splitext(f)[1].lower() in image_extensions]

        if max_images:
            self.image_paths = self.image_paths[:max_images]

        print(f"Processing {len(self.image_paths)} images...")

        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.resize(image, (1280, 720))
            person_count = self._count_persons(image)
            self.person_counts.append(person_count)

        return self._get_results()

    def _count_persons(self, image):
        """Count persons in a single image with error handling"""
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input")
            return 0

        # Run inference with slightly higher confidence to reduce false positives
        results = self.model(image, conf=0.3, verbose=False)
        person_count = 0

        for result in results:
            # Check if results have boxes attribute
            if not hasattr(result, 'boxes'):
                continue

            for box in result.boxes:
                # Skip if box doesn't have expected attributes
                if not all(hasattr(box, attr) for attr in ['cls', 'conf']):
                    continue

                class_id = int(box.cls)
                confidence = float(box.conf)

                # Only count if class is person and confidence > threshold
                if self.model.names[class_id] == "person" and confidence > 0.3:
                    person_count += 1

        return person_count

    def _get_results(self):
        """Return results as a dictionary"""
        return {
            "image_paths": self.image_paths,
            "person_counts": self.person_counts,
            "total_people": sum(self.person_counts),
            "average_per_image": sum(self.person_counts) / len(self.person_counts) if self.person_counts else 0
        }

    def save_model(self, output_dir="model"):
        """Save the model for later use"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.export(format="onnx", imgsz=[720, 1280], simplify=True)
            print(f"Model saved to {output_dir}/")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def save_results(self, output_file="results.json"):
        """Save analysis results to JSON"""
        results = self._get_results()
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    analyzer = CrowdAnalyzer()
    results = analyzer.process_directory(image_dir="data/UoB_Graduation_Ceremony_Day", max_images=10)
    analyzer.save_model()
    analyzer.save_results()

    print("\nAnalysis Results:")
    print(f"Total people detected: {results['total_people']}")
    print(f"Average per image: {results['average_per_image']:.1f}")
