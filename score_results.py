"""Re-score the saved Mistral 7B outputs in results/ offline, next to Table 1 of the report.

No LLM run needed: results/*.txt hold the raw Mistral answers (pickled lists of strings,
one per input text, in dataset order). This script turns each answer into a label, takes
the same 90% test split as the report (sklearn train_test_split, random_state=42), and
prints F1-macro and accuracy beside the published numbers.

Honest caveat (found in 2026, see README): these saved outputs do NOT reproduce Table 1
exactly. The paper numbers came from an evaluation pass that is not fully captured in
this repo, and results/DS2_3_Shot_res.txt was corrupted when it was committed in 2024
(its binary pickle bytes were re-encoded as UTF-8 text), so it cannot be loaded at all.

Two label parsers are shown:
  repo     data_evaluator.cleaner as committed. Note: its phonetic fallback maps the
           answer "Neutral" to "Positive" on the 2-class DS1, which drags DS1 down.
  prefix   the answer's first word: pos* -> Positive, neg* -> Negative, neu* -> neutral
           (DS1 has no neutral class, so neutral and anything unparseable -> Negative,
           the DS1 majority class).

Needs both datasets in place (README, "Datasets"): the ground-truth labels live there.

    python score_results.py
"""
import contextlib
import io
import os
import pickle
import sys

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# dataset, prompting, saved outputs, cleaner type, fallback label, Table 1 (F1-macro, acc)
RUNS = [
    ("DS1", "zero-shot", "results/DS1_0_shot_res.txt", "simple", "Negative", (0.74, 0.85)),
    ("DS1", "3-shot", "results/DS1_3_shot_res.txt", "simple", "Negative", (0.78, 0.90)),
    ("DS2", "zero-shot", "results/DS2_0_Shot_res.txt", "tweet", "Positive", (0.59, 0.64)),
    ("DS2", "3-shot", "results/DS2_3_Shot_res.txt", "tweet", "Positive", (0.66, 0.70)),
]


def prefix_label(answer, kind):
    s = answer.strip().strip("'\"").lower()
    if s.startswith("pos"):
        return "Positive"
    if s.startswith("neg"):
        return "Negative"
    if s.startswith("neu") and kind == "tweet":
        return "neutral"
    return "Negative"


def test_split(seq):
    """The report's split: 10% train (unused here), 90% test, random_state=42."""
    _, test = train_test_split(seq, test_size=0.9, train_size=0.1, random_state=42)
    return test


def scores(true, pred):
    return f1_score(true, pred, average="macro"), accuracy_score(true, pred)


def main():
    with contextlib.redirect_stdout(io.StringIO()):  # the 2024 code prints every row on import
        import data_evaluator

    print(f"{'run':18s} {'n test':>7s}   {'paper F1/acc':>12s}   {'repo F1/acc':>11s}   {'prefix F1/acc':>13s}")
    for ds, mode, path, kind, maj, paper in RUNS:
        name = f"{ds} {mode}"
        paper_s = f"{paper[0]:.2f}/{paper[1]:.2f}"
        try:
            with open(os.path.join(HERE, path), "rb") as f:
                answers = pickle.load(f)
        except pickle.UnpicklingError:
            print(f"{name:18s} {'-':>7s}   {paper_s:>12s}   saved outputs corrupted in the 2024 commit, cannot score")
            continue

        labels = data_evaluator.simpleSentiments if kind == "simple" else data_evaluator.tweetSentiments
        true = test_split(list(labels.values[: len(answers)]))

        with contextlib.redirect_stdout(io.StringIO()):
            repo_pred, _ = data_evaluator.cleaner(os.path.join(HERE, path), kind, majMap=maj)
        repo = scores(true, test_split(list(repo_pred)))
        prefix = scores(true, test_split([prefix_label(a, kind) for a in answers]))

        print(f"{name:18s} {len(true):7d}   {paper_s:>12s}   {repo[0]:.2f}/{repo[1]:.2f}"
              f"     {prefix[0]:.2f}/{prefix[1]:.2f}")


if __name__ == "__main__":
    main()
