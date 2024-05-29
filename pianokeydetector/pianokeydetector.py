import cv2
import numpy as np
import matplotlib.pyplot as plt


class PianoKeyDetector:
    def __init__(self, image_data):  # Image Data can be either path to image, or cv2 matrix
        if type(image_data) == str:
            self.image = cv2.imread(image_data)
        else:
            self.image = image_data
        self.corner_points = []
        self.key_bounds = []

    def sort_points(self, point_list):
        point_list = sorted(point_list)
        bottom_left, top_left = (point_list[0], point_list[1]) if point_list[0][1] >= point_list[1][1] else (
            point_list[1], point_list[0])
        bottom_right, top_right = (point_list[2], point_list[3]) if point_list[2][1] >= point_list[3][1] else (
            point_list[3], point_list[2])
        return [top_left, bottom_left, bottom_right, top_right]

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corner_points) < 4:
            self.corner_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', self.image)

    def get_corner_points(self):
        cv2.imshow('Image', self.image)
        cv2.setMouseCallback('Image', self.click_event)
        cv2.waitKey(0)

    def process_image(self, nkeys):
        if len(self.corner_points) != 4:
            raise ValueError("Exactly 4 corner points are required!")

        src_points = np.array(self.sort_points(self.corner_points), dtype="float32")

        # Original Image Mask And Crop
        mask = np.zeros_like(self.image)
        cv2.fillPoly(mask, [np.int32(src_points)], (255, 255, 255))
        cropped_image = cv2.bitwise_and(self.image, mask)

        # Define the points for the top-down view
        width = int(min((src_points[2][0] - src_points[1][0]) ** 2 + (src_points[2][1] - src_points[1][1]) ** 2,
                        (src_points[3][0] - src_points[0][0]) ** 2 + (src_points[3][1] - src_points[0][1]) ** 2) ** 0.5)
        height = int(min((src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2,
                         (src_points[2][0] - src_points[3][0]) ** 2 + (
                                 src_points[2][1] - src_points[3][1]) ** 2) ** 0.5)
        dst_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype="float32")

        # Get the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_transform_matrix = np.linalg.inv(transform_matrix)

        # Apply the perspective transformation
        corrected_image = cv2.warpPerspective(self.image, transform_matrix, (width, height))

        # Draw the key divisions
        key_width = width // nkeys
        for i in range(nkeys+1):
            x = i * key_width
            cv2.line(corrected_image, (x, 0), (x, height), (255, 0, 0), 2)

            # Transform the points back to the original image
            pt1 = np.array([x, 0, 1])
            pt2 = np.array([x, height, 1])
            orig_pt1 = inverse_transform_matrix.dot(pt1)
            orig_pt2 = inverse_transform_matrix.dot(pt2)
            orig_pt1 = (orig_pt1 / orig_pt1[2])[:2].astype(int)
            orig_pt2 = (orig_pt2 / orig_pt2[2])[:2].astype(int)

            # Draw the transformed lines on the original image
            cv2.line(cropped_image, tuple(orig_pt1), tuple(orig_pt2), (255, 0, 0), 2)
            self.key_bounds.append([tuple(orig_pt1), tuple(orig_pt2)])

        # Convert images from BGR to RGB for matplotlib
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)

        # Display the result using matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(corrected_image_rgb)
        axes[0].set_title('Corrected Image with Lines')
        axes[0].axis('off')

        axes[1].imshow(cropped_image_rgb)
        axes[1].set_title('Original Image with Lines')
        axes[1].axis('off')

        plt.show()

    def run(self, nkeys):
        self.get_corner_points()
        self.process_image(nkeys)
        return self.key_bounds

