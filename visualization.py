import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class HeadVisualizer:
    def __init__(self, dot_radius=3, dot_color=(0, 0, 255),
                 count_color=(0, 0, 255), font_scale=1.2):
        """
        Initialize visualizer with customization options

        Args:
            dot_radius: Radius of head markers in pixels (default: 3)
            dot_color: BGR color tuple for dots (default: red)
            count_color: BGR color tuple for count text
            font_scale: Size of count text
        """
        self.dot_radius = dot_radius
        self.dot_color = dot_color
        self.count_color = count_color
        self.font_scale = font_scale
        self.threshold = 30  # Density threshold for head detection
        self.min_contour_area = 3  # Minimum contour area to consider as head

    def mark_heads(self, original_img, density_map):
        """
        Mark detected heads on the original image

        Args:
            original_img: Input image (RGB format)
            density_map: Model output (1xHxWx1)

        Returns:
            Marked image (RGB format)
        """
        # Convert to working copy
        marked_img = original_img.copy()
        orig_h, orig_w = original_img.shape[:2]

        # Process density map
        density = density_map.squeeze()
        density = (density - density.min()) / (density.max() - density.min() + 1e-7)

        # Create binary map of head locations
        binary = cv2.threshold((density * 255).astype(np.uint8),
                               self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Find and scale contours
        scale_x = orig_w / density_map.shape[2]
        scale_y = orig_h / density_map.shape[1]

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Draw markers
        head_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    head_x = int(M["m10"] / M["m00"] * scale_x)
                    head_y = int(M["m01"] / M["m00"] * scale_y)
                    cv2.circle(marked_img, (head_x, head_y),
                               self.dot_radius, self.dot_color, -1)
                    head_count += 1

        # Add count text
        cv2.putText(marked_img, f"{head_count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    self.count_color, 2, cv2.LINE_AA)

        return marked_img

    @staticmethod
    def preprocess_image(image_bytes, target_size=256):
        """Helper method to preprocess uploaded images"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size))
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


if __name__ == "__main__":
    # Initialize visualizer with custom settings
    visualizer = HeadVisualizer(
        dot_radius=4,
        dot_color=(0, 255, 255),  # Yellow dots
        count_color=(0, 255, 0),  # Green text
        font_scale=1.5
    )

    # Example image path - adjust as needed
    image_path = "data/UoB_Graduation_Ceremony_Day/DSC01305.jpg"

    # Load and process image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Create a more realistic dummy density map
    h, w = 256, 256  # Match your model's expected input size

    # Option 1: Random head placements (simple)
    dummy_density = np.zeros((1, h, w, 1))
    for _ in range(150):  # Simulate 150 detected heads
        x, y = np.random.randint(10, w - 10), np.random.randint(10, h - 10)
        dummy_density[0, y - 3:y + 4, x - 3:x + 4, 0] = np.random.uniform(0.5, 1.0, (7, 7))

    # Option 2: Load actual model output if available (better)
    # dummy_density = model.predict(preprocess_image(original_img))

    # Proper Gaussian blur application
    density_2d = dummy_density[0].squeeze()  # Remove batch and channel dims
    density_blurred = cv2.GaussianBlur(density_2d, (15, 15), 0)
    dummy_density[0] = density_blurred.reshape((h, w, 1))  # Restore dimensions

    # Generate visualization
    marked_img = visualizer.mark_heads(original_img, dummy_density)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.imshow(marked_img)
    plt.axis('off')
    plt.title("Graduation Ceremony Crowd Detection")
    plt.show()

    # Save the result
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "graduation_marked.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output_path}")