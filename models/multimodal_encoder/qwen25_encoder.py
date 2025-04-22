from transformers import AutoTokenizer, AutoModel
import torch


# model_name = "qwen/Qwen-2.5-7b"  

class Qwen2Embedder:
    available_models = "Qwen/Qwen2.5-7B"

    def __init__(
        self,
        device,
        from_pretrained=available_models,
        cache_dir=None,
        model_max_length=120,
        local_files_only=False,
    ):

        # assert from_pretrained in self.available_models 
        self.tokenizer = AutoTokenizer.from_pretrained(
            # "Qwen/Qwen2.5-7B",
            from_pretrained,
            model_max_length=model_max_length,
            # cache_dir=cache_dir,
            local_files_only=local_files_only,)
        
        self.model = AutoModel.from_pretrained(
            # "Qwen/Qwen2.5-7B",
            from_pretrained,
            # cache_dir=cache_dir,
            local_files_only=local_files_only,).eval()

        self.model.to(device)
        
        self.model_max_length = model_max_length

if __name__ == "__main__":
    qwen = Qwen2Embedder(cache_dir='~/hf-cache')
    tokenizer, text_encoder = qwen.tokenizer, qwen.model

    texts = [
        "Insert a lemon slice into the paper cup, then place it on the rim of the wine glass.",
        "Pour the water from the bottle into the mug."
    ]

    # encode text
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = text_encoder(**inputs)

    token_embeddings = outputs.last_hidden_state  # shape [batch_size, seq_len, hidden_dim]

    print(f"Token embeddings shape: {token_embeddings.shape}")