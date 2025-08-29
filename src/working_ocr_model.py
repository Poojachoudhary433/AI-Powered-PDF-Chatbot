import os
import torch
from datetime import datetime
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from hugging_face_login import hugging_face_login
import pytesseract
from PIL import Image
import time
from datetime import datetime


## Used this OCR model beacuse it is very accurate although it takes time

# Check for MPS availability and PyTorch version
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

class OptimizedOCRModel:
    def __init__(self, model_path="nanonets/Nanonets-OCR-s", local_path=rf"local_models/nanonets_ocr_s"):
        self.device = device
        self.model = None
        self.processor = None

        self.load_model(model_path, local_path)

    def load_model(self, model_path, local_path=None):
        """Load model with MPS optimization"""
        try:
            # Try loading from local path first
            if local_path and os.path.exists(local_path):
                print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''',end = " ")
                print(f"Loading model from local path: {local_path}")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    local_path,
                    torch_dtype=torch.float16 if self.device == "mps" else "auto"
                )
                self.processor = AutoProcessor.from_pretrained(local_path)
            else:
                print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
                hugging_face_login()
                print(f"Hugging face login successful")
                print(f"Loading model from HuggingFace: {model_path}")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "mps" else "auto"
                )
                self.processor = AutoProcessor.from_pretrained(model_path)
                print(f"Loaded model from HuggingFace: {model_path}")
                # Save locally for future use
                if local_path:
                    os.makedirs(local_path, exist_ok=True)
                    self.model.save_pretrained(local_path)
                    self.processor.save_pretrained(local_path)
                    print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
                    print(f"Model saved to: {local_path}")

            # Move model to MPS device and optimize
            self.model = self.model.to(self.device)
            self.model.eval()

            # Enable MPS optimizations
            if self.device == "mps":
                # Clear MPS cache if available (PyTorch 2.0+)
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
                print("MPS optimizations enabled")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def preprocess_image(self, image_path):
        """Optimized image preprocessing"""
        image = Image.open(image_path)

        # Optimize image size for better processing speed
        max_size = 720
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
            print(f"Resized image to: {new_size}")
        return image

    def ocr_from_image(self, image_path, max_new_tokens=4096):
        """Optimized OCR function using MPS"""
        start_time = time.time()
        print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
        print(f"Starting OCR for: {image_path}")

        prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        image = self.preprocess_image(image_path)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs with MPS optimization
        with torch.no_grad():
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate with optimized settings for MPS
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
                'pad_token_id': self.processor.tokenizer.pad_token_id,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
            }

            # Use autocast only if supported
            if self.device == 'mps' and hasattr(torch, 'autocast'):
                try:
                    with torch.autocast(device_type='mps', enabled=True):
                        generated_ids = self.model.generate(**inputs, **generation_kwargs)
                except:
                    # Fallback without autocast if it fails
                    print("MPS autocast failed, using fallback")
                    generated_ids = self.model.generate(**inputs, **generation_kwargs)
            else:
                generated_ids = self.model.generate(**inputs, **generation_kwargs)

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]

        elapsed_time = time.time() - start_time
        print(f'''{datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}''', end=" ")
        print(f"OCR completed in {elapsed_time:.2f} seconds")

        # Clear MPS cache to free memory if available
        if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()

        return output_text
    def predict(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs)
        return self.processor.decode(generated[0], skip_special_tokens=True)



# Main execution
def main():
    # Paths
    local_model_path = "/Users/dilipdilip/AI4Classroom/local_models/nanonets_ocr_s"
    image_path = "/Users/dilipdilip/AI4Classroom/Automation/OCR2Text/images/img.jpeg"
  
    # Initialize optimized OCR model
    result_1 = OptimizedOCRModel().ocr_from_image(image_path, max_new_tokens=1500)


    print("\n" + "=" * 50)
    print("OCR RESULT from Image:")
    print("=" * 50)
    print(result_1)
    start = result_1.find('{')
    end = result_1.rfind('}')
   


if __name__ == "__main__":
    # main()
    pass
