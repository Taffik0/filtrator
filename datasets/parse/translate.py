import json
import torch
from transformers import MarianMTModel, MarianTokenizer

print("CUDA доступна:", torch.cuda.is_available())
print(
    "GPU:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "нет"
)

MODEL = "Helsinki-NLP/opus-mt-en-ru"
BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = MarianTokenizer.from_pretrained(MODEL)

model = MarianMTModel.from_pretrained(MODEL)
model = model.to(device)
model.eval()

print("Model loaded on:", device)

with open(
    "./datasets/dataset_normal_test_en.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

texts = [x["body"] for x in data]

translations = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]

    tokens = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # CPU → GPU
    tokens = {
        key: value.to(device)
        for key, value in tokens.items()
    }

    with torch.no_grad():
        output = model.generate(**tokens)

    translated = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )

    translations.extend(translated)

    print(
        f"{len(translations)}/{len(texts)}"
    )

for item, translation in zip(data, translations):
    item["body_ru"] = translation

with open(
    "output.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )