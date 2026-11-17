"""Fast sanity tests (no trained model needed except where noted).  Run: pytest -q"""
import pandas as pd

from src.config import DATA_DIR, GAP_COLS, TASKS
from src.data import Vocab, tokenize
from src.slots import evidence_present, extract_slots


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Write a Python function, please!") == ["write", "a", "python", "function", "please"]


def test_vocab_handles_unknown_words():
    v = Vocab.build(["write a function", "write a function"], min_freq=1)
    ids = v.encode("write a zzzunknown")
    assert ids[-1] == 1  # UNK id


def test_slots_find_language_and_limits():
    s = extract_slots("Write a Python script, under 100 words, as a table")
    assert s["language"].lower() == "python" and s["has_limit"] and s["has_format_words"]
    assert evidence_present("keep it under 50 words") == {"constraints"}


def test_dataset_files_are_valid():
    for split in ["train", "val", "test", "manual_test"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        assert set(df["task"]) <= set(TASKS)
        assert set(GAP_COLS) <= set(df.columns)
        assert df[GAP_COLS].isin([0, 1]).all().all()


def test_no_train_test_leakage():
    train = set(pd.read_csv(DATA_DIR / "train.csv")["text"])
    test = set(pd.read_csv(DATA_DIR / "test.csv")["text"])
    assert not train & test
