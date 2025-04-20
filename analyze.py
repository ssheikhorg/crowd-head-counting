import cv2
from ultralytics import YOLO
import os

class CrowdAnalyzer:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.person_counts = []
        self.image_paths = []
        self.visualizations = []  # Store visualization results

    def process_directory(self, image_dir, max_images=None, visualize=False):
        """Process images with optional visualization"""
        # ... [existing image loading code] ...

        for image_path in self.image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.resize(image, (1280, 720))
            result = self._analyze_image(image)
            self.person_counts.append(result["count"])

            if visualize:
                vis_img = self._visualize_results(image, result)
                self.visualizations.append(vis_img)
                cv2.imwrite(f"output/{os.path.basename(image_path)}", vis_img)

    def _analyze_image(self, image):
        """Enhanced detection with head position tracking"""
        results = self.model(image, conf=0.5, verbose=False)  # Higher confidence
        person_count = 0
        head_positions = []  # Store head coordinates

        for result in results:
            if not hasattr(result, 'boxes'):
                continue

            for box in result.boxes:
                if not all(hasattr(box, attr) for attr in ['cls', 'xyxy']):
                    continue

                if self.model.names[int(box.cls)] == "person":
                    person_count += 1
                    # Get head position (top center of bounding box)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    head_x = int((x1 + x2) / 2)
                    head_y = int(y1 + (y2 - y1) * 0.2)  # 20% down from top
                    head_positions.append((head_x, head_y))

        return {"count": person_count, "heads": head_positions}

    def _visualize_results(self, image, analysis_result):
        """Draw precise dots on heads"""
        vis_img = image.copy()

        for x, y in analysis_result["heads"]:
            # Draw small red dot on head position
            cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)  # 3px red dot

        # Add count text
        cv2.putText(vis_img, f"Count: {analysis_result['count']}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return vis_img
