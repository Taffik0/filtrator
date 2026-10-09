import json
import re

with open("./datasets/raw/genered_normal.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = re.compile(
    r"Subject:(.*?)\nBody:(.*?)(?:\n\n|\Z)",
    re.DOTALL
)

data: list[dict[str, str]] = [
    {
        "from": "",
        "to": "",
        "subject": subject.strip(),
        "body": body.strip()
    }
    for subject, body in pattern.findall(text)
]

print(len(data))

with open("dataset_generated_normal.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
