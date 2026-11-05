"""
LoRA fine-tuning for olmOCR-2 and LightOnOCR.

Usage:
    python main.py finetune olmocr2 --epochs 2
or from Python:
    from training.lora import train
    train("lightonocr", epochs=2)

The result is a folder with the fine-tuned model (LoRA merged into the base
model), its processor and training_info.json. Use it for transcription with:
    python main.py transcribe <model> --finetuned <folder>
"""

import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoProcessor,
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from dataset import DATASET_DIR, SUBSET, check_images_exist, load_records

PROMPT = "Transcribe the document."

MODELS = {
    "olmocr2": {
        "model_id": "allenai/olmOCR-2-7B-1025",
        "lora_rank": 16,
        "max_image_size": 1288,
        "image_in_message": True,    # message holds the image: {"type": "image", "image": img}
        "gradient_checkpointing": True,   # needed to fit 7B training on 2 x 24 GB
    },
    "lightonocr": {
        "model_id": "lightonai/LightOnOCR-2-1B",
        "lora_rank": 8,
        "max_image_size": 768,       # NOTE: LightonOcrModel transcribes with 1288 px
        "image_in_message": False,   # message only marks the image: {"type": "image"}
        "gradient_checkpointing": False,
    },
}

LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

WEIGHT_DECAY = 0.01
WARMUP_DIVISOR = 10
MAX_GRAD_NORM = 1.0
LOG_EVERY_STEPS = 20
IGNORE_INDEX = -100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(
    model: str,
    epochs: int = 2,
    train_size: int | None = None,
    subset: str = SUBSET,
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path | None = None,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> Path:
    """Fine-tune one model with LoRA and return the folder of the saved model.

    train_size: use only this many training images, spread over the campaigns
                (None = all images, as in the original script).
    """
    settings = MODELS[model]
    set_seed(seed)  # Later fix: same shuffling and LoRA initialisation on every run

    # Training data: the train split, optionally only train_size images.
    records = load_records("train", subset, dataset_dir)
    check_images_exist(records)
    if train_size:
        records = records.loc[sample_per_campaign(records["campaign"], train_size, seed)]
    samples = list(zip(records["image_path"], records["text"]))
    print(f"Training {model} on {len(samples)} images for {epochs} epochs")

    # Base model with LoRA layers
    base_model, processor = load_model(model)
    if settings["gradient_checkpointing"]:
        # Recompute activations in the backward pass instead of storing them:
        # much less GPU memory, same gradients (only slower).
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_model = get_peft_model(base_model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings["lora_rank"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    ))
    lora_model.print_trainable_parameters()

    loader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(make_batch, processor=processor, settings=settings),
    )

    train_loop(lora_model, loader, epochs, learning_rate)

    # Save the model with LoRA merged in
    output_dir = output_dir or Path("checkpoints") / f"{model}__lora__ep{epochs}__n{len(samples)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_model.merge_and_unload().save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    save_training_info(output_dir, model, settings, list(records.index), epochs, batch_size,
                       learning_rate, seed)
    print(f"Fine-tuned model saved to {output_dir}")
    return output_dir

def sample_per_campaign(campaigns, n: int, seed: int) -> list[str]:
    """Ids of n images spread evenly over the campaigns (for learning curves).

    campaigns: image id -> campaign name (the "campaign" column of load_records).
    """
    rng = random.Random(seed)

    by_campaign = defaultdict(list)
    for image_id, campaign in campaigns.items():
        by_campaign[campaign].append(image_id)

    groups = list(by_campaign.values())
    rng.shuffle(groups)            # so small n does not always pick the same campaigns
    for group in groups:
        rng.shuffle(group)

    # Take one image per campaign in turn until n are chosen.
    chosen = []
    while len(chosen) < min(n, len(campaigns)):
        for group in groups:
            if group and len(chosen) < n:
                chosen.append(group.pop())
    return chosen

def load_image(path: Path, max_size: int) -> Image.Image:
    """Open as RGB and shrink so the longer side is at most max_size."""
    with Image.open(path) as img:
        image = img.convert("RGB")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def make_batch(batch, processor, settings: dict) -> dict:
    """Turn (image path, transcription) pairs into model inputs with labels.

    The model sees: user turn (image + PROMPT) followed by the assistant turn
    (the transcription). The loss is only computed on the transcription.
    """
    images, full_texts, prompt_texts = [], [], []
    for image_path, transcription in batch:
        image = load_image(image_path, settings["max_image_size"])
        if settings["image_in_message"]:
            image_part = {"type": "image", "image": image}
        else:
            image_part = {"type": "image"}
        question = {"role": "user", "content": [image_part, {"type": "text", "text": PROMPT}]}
        answer = {"role": "assistant", "content": transcription}

        images.append(image)
        
        full_texts.append(processor.apply_chat_template(
            [question, answer], tokenize=False, add_generation_prompt=False))
        
        prompt_texts.append(processor.apply_chat_template(
            [question], tokenize=False, add_generation_prompt=True))

    inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    inputs = move_to_device(inputs, settings)

    labels = inputs["input_ids"].clone()
    for i, (image, prompt_text) in enumerate(zip(images, prompt_texts)):
        prompt_length = processor(text=prompt_text, images=image, return_tensors="pt")[
            "input_ids"].shape[1]
        labels[i, :prompt_length] = IGNORE_INDEX
    inputs["labels"] = labels
    return inputs


def move_to_device(inputs, settings: dict) -> dict:
    """Move all tensors to the GPU, exactly as in the original script."""
    if settings["image_in_message"]:  # olmOCR-2: tensors keep their dtype
        return {key: value.to(DEVICE) for key, value in inputs.items()}
    # LightOnOCR: float tensors (the image) are converted to bfloat16 like the model.
    return {
        key: value.to(DEVICE, dtype=torch.bfloat16) if value.is_floating_point()
        else value.to(DEVICE)
        for key, value in inputs.items()
    }

def load_model(model: str):
    """Load the base model and processor, exactly as in the original script."""
    model_id = MODELS[model]["model_id"]

    if model == "olmocr2":
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        if isinstance(getattr(config, "text_config", None), dict):
            config.text_config = Qwen2Config(**config.text_config)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # spread the 7B model over all visible GPUs
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        base_model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(DEVICE)
        processor = LightOnOcrProcessor.from_pretrained(model_id)

    return base_model, processor

def train_loop(model, loader: DataLoader, epochs: int, learning_rate: float) -> None:
    """AdamW with cosine learning rate schedule and warmup, gradient clipping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // WARMUP_DIVISOR,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if step % LOG_EVERY_STEPS == 0:
                print(f"Epoch {epoch} | step {step} | loss {loss.item():.4f}")
        print(f"Epoch {epoch} | average loss {epoch_loss / len(loader):.4f}")

def save_training_info(output_dir: Path, model: str, settings: dict, train_ids: list,
                       epochs: int, batch_size: int, learning_rate: float, seed: int) -> None:
    """Write all settings next to the model, so every result can be traced back."""
    info = {
        "model": model,
        **settings,
        "prompt": PROMPT,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": f"1/{WARMUP_DIVISOR} of all steps",
        "max_grad_norm": MAX_GRAD_NORM,
        "train_images": train_ids,
    }
    (output_dir / "training_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")