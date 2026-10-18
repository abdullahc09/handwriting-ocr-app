from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
import torch
import csv

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
csv_path = Path("results/baseline.csv")
processed_files = set()
image_folder = Path("data")

if csv_path.exists():
    with open(csv_path, mode = "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if row:
                 processed_files.add(row[0])

with open(csv_path, mode = "a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    for image_path in sorted(image_folder.iterdir()):
        file_name = image_path.name
        if file_name in processed_files:
            continue
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            raw_image = Image.open(image_path).convert("RGB")

            result = processor(images=raw_image, return_tensors="pt")
            pixel_vals = result.pixel_values
            device_pixel_vals = pixel_vals.to(device)

            generated_ids = model.generate(device_pixel_vals)
            decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
            model_output = decoded_output[0]
            writer.writerow([file_name, "", model_output, ""])
            processed_files.add(file_name)
            
            print("File:", image_path.name, "Text:", decoded_output[0])
        else:
            print("Incompatible file format. Skipping.")