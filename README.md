# Sentiment analysis with a pretrained, non-fine-tuned LLM

Can a general-purpose local LLM (Mistral 7B Instruct, prompted only, never fine-tuned) classify the sentiment of short texts as well as purpose-built models? Bachelor semester project at the [Idiap Research Institute](https://www.idiap.ch) (EPFL, spring 2024), published as Idiap research report [Idiap-RR-05-2024](https://publications.idiap.ch/attachments/reports/2024/Huou_Idiap-RR-05-2024.pdf).

## Results

Prompted Mistral 7B lands between a classical ML baseline and a RoBERTa model fine-tuned for emotion detection. 3-shot prompting with randomly drawn examples beats zero-shot on both datasets.

![Results on the 90% test split](docs/results.png)

| Model (90% test split) | DS1 Messages F1-macro | DS1 accuracy | DS2 Tweets F1-macro | DS2 accuracy |
|---|---|---|---|---|
| Baseline: Decision Tree | 0.56 | 0.87 | 0.62 | 0.72 |
| Mistral 7B, zero-shot | 0.74 | 0.85 | 0.59 | 0.64 |
| Mistral 7B, 3-shot | 0.78 | 0.90 | 0.66 | 0.70 |
| Baseline: RoBERTa (fine-tuned for emotion) | 0.80 | 0.91 | 0.75 | 0.78 |

Source: Table 1 of the report. The report also shows that the Decision Tree degrades sharply when trained on one dataset and tested on the other, while the prompted LLM needs no training data at all.

## How it works

1. `data_retriever.py` loads both emotion datasets and maps emotions to sentiment (joy to Positive; disgust, sadness, fear, anger, shame, guilt to Negative; neutral to Neutral; surprise dropped).
2. `data_player.py` prints dataset statistics after NLTK stopword removal.
3. `data_modeliser.py` runs Mistral 7B Instruct (GGUF, via `llama-cpp-python`, full GPU offload) over every text with a prompt from `prompts/`, optionally adding 3 random labelled examples (`--random_few_shot`). Predictions are pickled into `results/`.
4. `data_evaluator.py` cleans the free-form LLM output (NLTK normalisation, edit distance, fuzzy match, fallback label `majMap`) and produces confusion matrices and classification reports on the same 90% split as the baselines.
5. `baselines/` holds the notebooks for the classical ML classifiers (Logistic Regression, SVM, Decision Tree, Random Forest) and the RoBERTa baseline; `baselines/preRunRoberta/` keeps the executed runs used in the report.

## Run it

Untested end to end in this cleanup (a full LLM pass takes 3 to 6 hours per dataset on a single GPU). Requires Python 3.9+.

```bash
git clone https://github.com/ahuou/SentimentAnalysis-PretrainedLLM
cd SentimentAnalysis-PretrainedLLM
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU inference, install `llama-cpp-python` with CUDA enabled (see the [llama-cpp-python install docs](https://github.com/abetlen/llama-cpp-python#installation); on Windows, [issue #721](https://github.com/abetlen/llama-cpp-python/issues/721) fixed the silent `BLAS = 0` CPU fallback).

**Datasets** (third-party, not redistributed here):

- DS1 (messages): download [Text Dataset for Text Emotion Detection](https://www.kaggle.com/datasets/jarvis11/text-dataset-for-text-emotion-detection) from Kaggle and place its `text.txt` at `dataset_textEmotion/text.txt`.
- DS2 (tweets): `git clone https://github.com/SannketNikam/Emotion-Detection-in-Text dataset_emoDetectInText/EmotionDetectInText` (the code reads `data/emotion_dataset.csv`).

**Model**: download `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (for example from [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)) into the repo root.

```bash
python data_retriever.py           # sanity check: loads and prints both datasets
python data_modeliser.py --prompts_file prompts/DS2_3_Shot_prompt.txt \
  --model_path mistral-7b-instruct-v0.2.Q4_K_M.gguf --marker DS2 --random_few_shot
python data_evaluator.py           # evaluates results/DS2_3_Shot_res.txt
```

`data_modeliser.py` arguments: `--marker DS1|DS2` picks the dataset, `--start N` resumes a crashed run from row N, `--random_few_shot` enables 3-shot prompting, `--classified` forces one example per label (not recommended: it biased the model towards "Neutral"). In `data_evaluator.py`, call `evaluator(path, "simple" | "tweet", ...)` for DS1 or DS2 respectively.

## My role

Solo project by Alexandre Huou ([@ahuou](https://github.com/ahuou)): dataset preprocessing and label mapping, the prompting pipeline and GPU setup, output post-processing, the classical and RoBERTa baselines, and the report.

## Credits

- Supervisors: Prof. Petr Motlicek and Dr. Esaú Villatoro-Tello, Idiap Research Institute.
- Datasets: [jarvis11 on Kaggle](https://www.kaggle.com/datasets/jarvis11/text-dataset-for-text-emotion-detection) (DS1) and [SannketNikam/Emotion-Detection-in-Text](https://github.com/SannketNikam/Emotion-Detection-in-Text) (DS2).
- Report: A. Huou, P. Motlicek, E. Villatoro-Tello, *Sentiment Analysis using pretrained LLMs*, Idiap-RR-05-2024, July 2024. [PDF](https://publications.idiap.ch/attachments/reports/2024/Huou_Idiap-RR-05-2024.pdf)
