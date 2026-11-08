from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def recognise_text(image, processor, model, device):
    raw_image = image.convert("RGB")
    result = processor(images=raw_image, return_tensors="pt")
    pixel_vals = result.pixel_values
    device_pixel_vals = pixel_vals.to(device)
    generated_ids = model.generate(device_pixel_vals)
    decoded_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
    model_output = decoded_output[0]
    return model_output