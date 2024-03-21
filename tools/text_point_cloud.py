import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_text_point_cloud(text,
                              font_path,
                              font_size=200,
                              jitter_amount=3,
                              down_sample_force=2):
    # Define the image font and size
    font = ImageFont.truetype(font=font_path, size=font_size)

    # Create an image with white background
    image = Image.new("RGB", (font_size * len(text), font_size),
                      (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Get the size text will take up and create a new image of that size
    text_size = draw.textsize(text, font=font)
    image = Image.new("RGB", text_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    draw.text((0, 0), text, (0, 0, 0), font=font)

    # Convert image to grayscale and then to an array
    grayscale = image.convert("L")
    arr = np.array(grayscale)

    # Get x, y coordinates of points that are black
    y, x = np.where(arr < 128)  # Text is black (less than 128 in grayscale)

    # Jitter the points to create the point cloud effect
    x_jitter = np.random.normal(loc=0, scale=jitter_amount, size=x.shape)
    y_jitter = np.random.normal(loc=0, scale=jitter_amount, size=y.shape)
    x = x + x_jitter
    y = y + y_jitter

    # Subsample the points to reduce density
    indices = np.random.choice(len(x),
                               len(x) // down_sample_force,
                               replace=False)
    x = x[indices]
    y = y[indices]

    # Flip y-coordinates to match image coordinate system with origin at top-left
    y = max(y) - y

    return x, y
