import json
from email import policy
from email.parser import Parser
from pathlib import Path
import random


def select_files(root: Path, n: int, attempts: int=1000) -> list[Path]:
    root = Path(root)
    result = []

    for _ in range(attempts):
        dirs = [root]

        while dirs:
            directory = random.choice(dirs)
            dirs.remove(directory)

            try:
                entries = list(directory.iterdir())
            except PermissionError:
                continue

            random.shuffle(entries)

            for entry in entries:
                if entry.is_file():
                    result.append(entry)

                    if len(result) >= n:
                        return result

                elif entry.is_dir():
                    dirs.append(entry)

    return result

def parse_email(raw: str) -> dict[str, str]:
    # Отделяем заголовки от тела.
    # Parser сам корректно обработает folded headers.
    msg = Parser(policy=policy.default).parsestr(raw)

    # Берём только текст до Original Message.
    body = msg.get_body(preferencelist=("plain",))

    if body is not None:
        body_text = body.get_content()
    else:
        body_text = msg.get_payload(decode=True)
        if isinstance(body_text, bytes):
            body_text = body_text.decode(
                msg.get_content_charset() or "utf-8",
                errors="replace"
            )

    if body_text is None:
        body_text = ""

    # Original проигнорировать
    original_marker = "-----Original Message-----"
    body_text = body_text.split(original_marker, 1)[0]

    return {
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "body": body_text.strip(),
    }


if __name__ == "__main__":
    files = select_files(Path("C:/Users/Admin/Desktop/maildir"), 100)
    parsed: list[dict[str, str]] = []
    for file in files:
        with open(file, "r") as f:
            raw = f.read()
            result = parse_email(raw)
            parsed.append(result)
    with open("./dataset_normal_test_en.json", "w") as f:
        json.dump(parsed, f)