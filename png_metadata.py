from PIL import Image
import piexif
import argparse

def get_png_metadata(image_path):
    print(image_path)
    with Image.open(image_path) as img:
        metadata = img.info  # Get basic metadata
        print("Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata from a PNG file.")
    parser.add_argument("image_path", help="Path to the PNG file")
    args = parser.parse_args()
    
    get_png_metadata(args.image_path)
