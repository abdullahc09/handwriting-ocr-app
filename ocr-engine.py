from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

raw_image = Image.open("test_line.png").convert("RGB")

result = processor(images=raw_image, return_tensors="pt")
pixel_vals = result.pixel_values
device_pixel_vals = pixel_vals.to(device)

generated_IDs = model.generate(device_pixel_vals)
decoded_output = processor.batch_decode(generated_IDs, skip_special_tokens=True)
print(decoded_output[0])