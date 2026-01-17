# tools/models/clip_model.py
import torch.nn as nn
from typing import Tuple, Union
import torch
from transformers import CLIPModel, CLIPProcessor


# 1. Model definition (unchanged)
class LearnablePromptCLIP(nn.Module):
    def __init__(self, model_name, class_names, device):
        super().__init__()
        self.device = device
        self.class_names = class_names

        # External path handling, receive absolute path directly
        print(f"Loading CLIP from: {model_name}")
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.n_ctx = 16
        ctx_dim = self.clip_model.config.text_config.hidden_size

        # Warm Start initialization
        prompt_init = "a photo of a"
        init_tokenized = self.processor(text=prompt_init, return_tensors="pt").to(device)
        with torch.no_grad():
            init_embedding = self.clip_model.text_model.embeddings.token_embedding(init_tokenized['input_ids'])

        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        valid_vectors = init_embedding[0, 1:-1, :]
        num_valid = min(self.n_ctx, valid_vectors.shape[0])
        ctx_vectors[:num_valid, :] = valid_vectors[:num_valid, :]

        self.ctx = nn.Parameter(ctx_vectors)

        # Preprocess class names
        dummy_prompts = [f"placeholder {name}" for name in class_names]
        tokenized = self.processor(
            text=dummy_prompts, padding="max_length", max_length=77, return_tensors="pt", truncation=True
        ).to(device)

        self.tokenized_prompts = tokenized['input_ids']
        self.embedding_layer = self.clip_model.text_model.embeddings.token_embedding
        self.position_embedding = self.clip_model.text_model.embeddings.position_embedding
        self.text_encoder = self.clip_model.text_model

    def forward(self, images):
        batch_size = images.shape[0]
        n_class = len(self.class_names)

        with torch.no_grad():
            input_ids = self.tokenized_prompts.to(self.device)
            embedding_tensor = self.embedding_layer(input_ids)

        ctx_expanded = self.ctx.unsqueeze(0).expand(n_class, -1, -1)
        prefix = embedding_tensor[:, :1, :]
        suffix = embedding_tensor[:, 2:, :]
        combined_embeds = torch.cat([prefix, ctx_expanded, suffix], dim=1)[:, :77, :]

        x = combined_embeds
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(n_class, -1)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = x + pos_embeds

        encoder_outputs = self.text_encoder.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.text_encoder.final_layer_norm(encoder_outputs.last_hidden_state)
        pooled_output = last_hidden_state[:, 76, :]

        text_features = self.clip_model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        image_outputs = self.clip_model.vision_model(pixel_values=images)
        image_features = self.clip_model.visual_projection(image_outputs.pooler_output)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return 100.0 * image_features @ text_features.T


def load_clip(
    model_name_or_path: str,
    device: Union[str, torch.device],
    use_fast: bool = False,
    eval_mode: bool = True,
) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and Processor (support local directory or HF name).
    """
    model = CLIPModel.from_pretrained(model_name_or_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_name_or_path, use_fast=use_fast)

    if eval_mode:
        model.eval()

    return model, processor