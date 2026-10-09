import json
import random
import re


def generate_training_file(normal: list[dict[str, str]], suspicion_danger: list[dict[str, str]], suspicion_terrorism: list[dict[str, str]]):
    mails = [{**n, "type": "normal"} for n in normal] + \
        [{**s, "type": "suspicion"} for s in suspicion_danger] + \
        [{**s, "type": "suspicion"} for s in suspicion_terrorism]
    random.shuffle(mails)
    with open("train_fast_text_suspicion.txt", "w", encoding="utf-8") as f:
        for mail in mails:
            text = mail["body"]
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
            text = text.translate(str.maketrans(
                '', '', ''.join(map(chr, range(32)))))
            f.write(f"""__label__{mail["type"]} {text}\n""")


def generate_test_file(normal: list[dict[str, str]], suspicion_danger: list[dict[str, str]], suspicion_terrorism: list[dict[str, str]]):
    mails = [{**n, "type": "normal"} for n in normal] + \
        [{**s, "type": "suspicion"} for s in suspicion_danger] + \
        [{**s, "type": "suspicion"} for s in suspicion_terrorism]
    random.shuffle(mails)
    with open("test_fast_text_suspicion.txt", "w", encoding="utf-8") as f:
        for mail in mails:
            text = mail["body"]
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
            text = text.translate(str.maketrans(
                '', '', ''.join(map(chr, range(32)))))
            f.write(f"""__label__{mail["type"]} {text}\n""")


if __name__ == "__main__":
    normal_train: list[dict[str, str]] = []
    normal_train_ru: list[dict[str, str]] = []
    normal_test: list[dict[str, str]] = []
    normal_test_ru: list[dict[str, str]] = []
    normal_generated_train: list[dict[str, str]] = []
    normal_generated_test: list[dict[str, str]] = []

    suspicion_danger_train: list[dict[str, str]] = []
    suspicion_danger_test: list[dict[str, str]] = []

    suspicion_terrorist_train: list[dict[str, str]] = []
    suspicion_terrorist_test: list[dict[str, str]] = []

    with open("./datasets/dataset_normal_train_en.json", "r", encoding="utf-8") as f:
        normal_train = json.load(f)
    with open("./datasets/dataset_normal_train_ru.json", "r", encoding="utf-8") as f:
        normal_train_ru = json.load(f)
    normal_train += [{**n, "body": n["body_ru"]} for n in normal_train_ru]
    normal_train = normal_train[:50]

    with open("./datasets/dataset_normal_test_en.json", "r", encoding="utf-8") as f:
        normal_test = json.load(f)
    with open("./datasets/dataset_normal_test_ru.json", "r", encoding="utf-8") as f:
        normal_test_ru = json.load(f)
    normal_test += [{**n, "body": n["body_ru"]} for n in normal_test_ru]
    normal_test = normal_test[:50]

    normal_generated: list[dict[str, str]] = []
    with open("./datasets/dataset_generated_normal.json", "r", encoding="utf-8") as f:
        normal_generated = json.load(f)

    normal_generated_test = normal_generated[0:20]
    normal_generated_train = normal_generated[20:]
    normal_test += normal_generated_test
    normal_train += normal_generated_train

    suspicion_danger: list[dict[str, str]] = []
    with open("./datasets/suspiction_danger.json", "r", encoding="utf-8") as f:
        suspicion_danger = json.load(f)
    random.shuffle(suspicion_danger)
    suspicion_danger_test = suspicion_danger[:10]
    suspicion_danger_train = suspicion_danger[10:]

    suspicion_terrorism: list[dict[str, str]] = []
    with open("./datasets/suspicion_terrorism.json", "r", encoding="utf-8") as f:
        suspicion_terrorism = json.load(f)
    random.shuffle(suspicion_terrorism)
    suspicion_terrorism_test = suspicion_terrorism[:10]
    suspicion_terrorism_train = suspicion_terrorism[10:]

    with open("./datasets/suspicion_terrorism_my.json", "r", encoding="utf-8") as f:
        mails = json.load(f)
    random.shuffle(mails)
    suspicion_terrorism_train += mails

    generate_training_file(
        normal_train, suspicion_danger_train, suspicion_terrorism_train)
    generate_test_file(normal_test, suspicion_danger_test,
                       suspicion_terrorist_test)
