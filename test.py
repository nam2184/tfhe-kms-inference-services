import os
import glob

def keep_only_one_image(directory, image_to_keep):
    """
    Removes all images in `directory` except `image_to_keep`.

    Args:
        directory (str): Path to the directory containing images.
        image_to_keep (str): Filename of the image to keep (just name, not path).
    """
    # Build full path of the image to keep
    keep_path = os.path.join(directory, image_to_keep)

    # Get all image files in directory (png, jpg, jpeg, gif, webp, bmp)
    image_files = glob.glob(os.path.join(directory, "*.[pjgPJGwWbB][pnifPNIFebEB]*"))

    for file_path in image_files:
        if os.path.abspath(file_path) != os.path.abspath(keep_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

# Example usage:
keep_only_one_image("dataset/drawings", "fff19ef43add7cc9a51f043fac3130f2b7ec74e75f662daa3eb9042314f5266e.jpg")

