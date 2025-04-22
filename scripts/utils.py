import numpy as np
import torch
import cv2
import os

def draw_arrows_on_image(image_array, points):
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Points tensor must have shape (N, 2).")
    
    points_np = points.to(dtype=torch.float32).cpu().numpy().astype(int)
   
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy.ndarray.")
    
    for i in range(len(points_np) - 1):
        pt1 = tuple(points_np[i])      # start point (x1, y1)
        pt2 = tuple(points_np[i + 1])  # end point (x2, y2)
        cv2.arrowedLine(image_array, pt1, pt2, color=(0, 0, 255), thickness=2, tipLength=0.3)
    
    return image_array


def draw_arrows_on_image_cv2(image_array, points, save_path="output_image.jpg"):
    """
    Args:
        image_array (numpy.ndarray): The image array with shape (H, W, 3).
        points (torch.Tensor): A tensor containing points with shape (N, 2), representing the x and y coordinates of N points.
        save_path (str): The path to save the image.
    """
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Points tensor must have shape (N, 2).")
    
    points_np = points.to(dtype=torch.float32).cpu().numpy().astype(int)
   
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy.ndarray.")
    

    for i in range(len(points_np) - 1):
        pt1 = tuple(points_np[i])
        pt2 = tuple(points_np[i + 1]) 
        cv2.arrowedLine(image_array, pt1, pt2, color=(0, 255, 0), thickness=2, tipLength=0.3) 

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, image_array)
    


def draw_text_on_image(image: np.ndarray, text: str, font_scale: float = 1.0, color: tuple = (0, 0, 0), thickness: int = 5) -> np.ndarray:
    """
    Draw text on the top-right corner of a given image (NumPy array).

    Parameters:
        image (np.ndarray): The input image (H, W, C) as a NumPy array.
        text (str): The text to draw.
        font_scale (float): Font size for the text.
        color (tuple): Text color in BGR (default is white).
        thickness (int): Thickness of the text.

    Returns:
        np.ndarray: Image with the text drawn on it.
    """
    # Make a copy of the image to avoid modifying the original
    image_with_text = image.copy()

    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text box
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position for the text (right-top corner)
    x = image_with_text.shape[1] - text_size[0] - 10  # 10px padding from the right edge
    y = text_size[1] + 10  # 10px padding from the top edge

    # Draw the text on the image
    cv2.putText(image_with_text, text, (x, y), font, font_scale, color, thickness)

    return image_with_text