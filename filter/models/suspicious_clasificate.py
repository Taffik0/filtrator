from dataclasses import dataclass
from pathlib import Path
import fasttext
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class SuspicionModelResponse():
    is_suspicious: bool
    label: str
    suspicion_probability: float


class SuspicionModel():
    def __init__(self, save_model: Path):
        self._save_mode = save_model
        self._model = fasttext.load_model(save_model)

    def classify(self, text: str, threshold: float = 0.3) -> SuspicionModelResponse:
        resp = classify(self._model, text, threshold=threshold)
        return SuspicionModelResponse(
            is_suspicious=resp["is_suspicious"],
            label=resp["label"],
            suspicion_probability=resp["suspicion_probability"],
        )


_model: SuspicionModel | None = None


def get_model():
    global _model
    if _model is not None:
        return _model
    raise "bly"


def init_model():
    global _model
    _model = SuspicionModel(Path("classifier.bin"))


def train_and_save():
    model = fasttext.train_supervised(
        input="train_fast_text_suspicion.txt",
        lr=0.1,
        epoch=20,
        wordNgrams=3,
        dim=100,
        loss="softmax"
    )

    model.save_model("classifier.bin")


def test_model(model):
    result = model.test("test_fast_text_suspicion.txt")

    print("N =", result[0])
    print("precision =", result[1])
    print("recall =", result[2])

    y_true = []
    y_pred = []

    with open("test_fast_text_suspicion.txt", encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split(" ", 1)

            predicted = model.predict(text)[0][0]

            y_true.append(label)
            y_pred.append(predicted)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def classify(model, text: str, threshold: float = 0.1) -> dict:
    labels, probabilities = model.predict(text, k=2)

    probs = dict(zip(labels, probabilities))

    suspicion_probability = float(
        probs.get("__label__suspicion", 0.0)
    ) + (1 - float(probs.get("__label__suspicion", 0.0)))

    is_suspicious = suspicion_probability >= threshold

    return {
        "label": "suspicion" if is_suspicious else "normal",
        "suspicion_probability": suspicion_probability,
        "is_suspicious": is_suspicious,
    }


def load_model(save_model: Path):
    return fasttext.load_model(save_model)


if __name__ == "__main__":
    texts = ["Я заложил бомбу под вашим домом", "Я взорву бомбу",
             "Привет", "Я заложил бомбу зданием"]

    train_and_save()
    model = load_model("classifier.bin")
    test_model(model)

    for t in texts:
        result = classify(model, t)

        print("Класс:", result["label"])
        print("Вероятность suspicion:",
              round(result["suspicion_probability"], 4))
        print("Подозрительный:", result["is_suspicious"])
