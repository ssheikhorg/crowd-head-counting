import os

import matplotlib.pyplot as plt
import json
from train_crowd import CrowdAnalyzer


class CrowdVisualizer:
    def __init__(self, results_file="results.json"):
        with open(results_file) as f:
            self.results = json.load(f)

    def show_bar_chart(self):
        """Show bar chart of person counts per image"""
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.results["image_paths"])),
                self.results["person_counts"],
                color='skyblue')

        plt.xlabel('Image')
        plt.ylabel('Number of Persons Detected')
        plt.title('Crowd Engagement Analysis\nNumber of Persons Detected in Each Image')

        # Use shortened image names for x-axis
        image_names = [os.path.basename(path)[:15] + "..." for path in self.results["image_paths"]]
        plt.xticks(range(len(image_names)), image_names, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def show_pie_chart(self):
        """Show pie chart of person distribution"""
        plt.figure(figsize=(8, 8))

        # Only show labels for images with significant counts (>5% of total)
        total = sum(self.results["person_counts"])
        threshold = total * 0.05
        labels = []
        for i, count in enumerate(self.results["person_counts"]):
            if count > threshold:
                short_name = os.path.basename(self.results["image_paths"][i])[:10] + "..."
                labels.append(f"{short_name}\n{count} people")
            else:
                labels.append("")

        plt.pie(self.results["person_counts"],
                labels=labels,
                autopct=lambda p: f"{p:.1f}%\n({int(p / 100 * total)})" if p > 5 else "",
                startangle=140)

        plt.title('Distribution of Detected Persons Across Images')
        plt.show()

    def show_summary_stats(self):
        """Print summary statistics"""
        print("\nCrowd Engagement Summary:")
        print(f"Total images analyzed: {len(self.results['image_paths'])}")
        print(f"Total people detected: {self.results['total_people']}")
        print(f"Average per image: {self.results['average_per_image']:.1f}")
        print(f"Most crowded image: {max(self.results['person_counts'])} people")
        print(f"Least crowded image: {min(self.results['person_counts'])} people")


if __name__ == "__main__":
    # Example usage
    visualizer = CrowdVisualizer()
    visualizer.show_bar_chart()
    visualizer.show_pie_chart()
    visualizer.show_summary_stats()