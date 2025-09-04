import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from llms.BaseModel import BaseVideoModel

class FastVLM(BaseVideoModel):
    def __init__(self, model_type="apple/FastVLM-0.5B", tp=1):
        """
        Initialize the FastVLM model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        self.model_type = model_type
        self.IMAGE_TOKEN_INDEX = -200  # what the model code looks for
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_type,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if tp == 1 else f"auto-{tp}",
            trust_remote_code=True,
        )
    
    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text and optionally video/images
            {
                "text": str,
                "video": list[Image.Image] (optional)
            }
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for generation

        Returns:
            str: Generated response.
        """
        assert "text" in inputs.keys(), "Please provide a text prompt."
        
        if "video" in inputs.keys() and len(inputs["video"]) > 0:
            # Handle image/video input - use first image for now
            img = inputs["video"][0]  # FastVLM typically handles single image
            
            # Build chat template with image placeholder
            messages = [
                {"role": "user", "content": f"<image>\n{inputs['text']}"}
            ]
            
            # Render to string so we can place <image> exactly
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            # Split around the image token
            pre, post = rendered.split("<image>", 1)
            
            # Tokenize the text around the image token
            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
            
            # Splice in the IMAGE token id at the placeholder position
            img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.model.device)
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
            
            # Preprocess image via the model's own processor
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                raise ValueError("Image must be PIL Image or path string")
                
            px = self.model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px = px.to(self.model.device, dtype=self.model.dtype)
            
            # Generate with image
            with torch.no_grad():
                out = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature if temperature > 0 else None,
                )
        else:
            # Handle text-only input
            messages = [
                {"role": "user", "content": inputs["text"]}
            ]
            
            # Render and tokenize
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids.to(self.model.device)
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
            
            # Generate text-only
            with torch.no_grad():
                out = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature if temperature > 0 else None,
                )
        
        # Decode and return response
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return response
    
    def batch_generate_response(self, batch_inputs, max_batch_size=64, max_new_tokens=512, temperature=0.5):
        """
        Generate responses for a batch of inputs.
        
        Note: This is a simple implementation that processes items sequentially.
        For true batch processing, you'd need to implement padding and batching logic.
        
        Args:
            batch_inputs (list): List of input dictionaries
            max_batch_size (int): Maximum batch size (currently not used in sequential processing)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for generation
            
        Returns:
            list[str]: List of generated responses
        """
        responses = []
        
        # Process each input sequentially
        # Note: True batching would require more complex padding/attention mask handling
        for inputs in batch_inputs:
            response = self.generate_response(inputs, max_new_tokens, temperature)
            responses.append(response)
            
        return responses

def main():
    print("Initializing FastVLM model...")
    model = FastVLM("apple/FastVLM-0.5B")
    print("Model loaded successfully!")
    
    # Test 1: Text-only input
    print("\n=== Test 1: Text-only ===")
    text_input = {
        "text": "What is artificial intelligence?"
    }
    
    try:
        response = model.generate_response(text_input, max_new_tokens=128)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in text-only test: {e}")
    
    # Test 2: Image + text input
    print("\n=== Test 2: Image + Text ===")
    try:
        img_path = "datas/image.png"
        img = Image.open(img_path).convert("RGB")
        
        image_input = {
            "text": "Describe this image in detail.",
            "video": [img]  # List of PIL Images
        }
        
        response = model.generate_response(image_input, max_new_tokens=128)
        print(f"Response: {response}")
        
    except FileNotFoundError:
        print("Image file 'test.jpg' not found. Skipping image test.")
        print("Please place an image file named 'test.jpg' in the current directory to test image functionality.")
    except Exception as e:
        print(f"Error in image test: {e}")
    
    # Test 3: Batch processing
    print("\n=== Test 3: Batch Processing ===")
    batch_inputs = [
        {"text": "What is the capital of France?"},
        {"text": "Explain quantum computing in simple terms."},
        {"text": "What are the benefits of renewable energy?"}
    ]
    
    try:
        import time
        start_time = time.time()
        responses = model.batch_generate_response(batch_inputs, max_new_tokens=64)
        for i, response in enumerate(responses):
            print(f"Batch {i+1}: {response}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
    except Exception as e:
        print(f"Error in batch test: {e}")

if __name__ == "__main__":
    main()