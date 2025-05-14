from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# Load pre-trained model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image (either from local or URL)
image_path = "yourimage.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')

# Process image and generate caption
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Image Description:", caption)
