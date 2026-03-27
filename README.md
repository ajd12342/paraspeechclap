# ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining

[[Paper]](#) [[HuggingFace Collection]](https://huggingface.co/collections/ajd12342/paraspeechclap-dual-encoder-speech-text-model)

Official code and model release for the paper:

**ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining**\
Anuj Diwan, Eunsol Choi, David Harwath\
*Under Review*

ParaSpeechCLAP is a CLAP-style dual-encoder model that maps speech and rich textual style descriptions into a common embedding space, supporting a wide range of **intrinsic** (speaker-level: pitch, texture, clarity, volume, rhythm) and **situational** (utterance-level: emotion, speaking style) descriptors.

## Models

| Model | Description | HuggingFace |
|---|---|---|
| **ParaSpeechCLAP-Intrinsic** | Specialized for intrinsic (speaker-level) style attributes. Trained with contrastive + classification loss and class-balanced sampling. | [ajd12342/paraspeechclap-intrinsic](https://huggingface.co/ajd12342/paraspeechclap-intrinsic) |
| **ParaSpeechCLAP-Situational** | Specialized for situational (utterance-level) style attributes. Trained with contrastive loss. | [ajd12342/paraspeechclap-situational](https://huggingface.co/ajd12342/paraspeechclap-situational) |
| **ParaSpeechCLAP-Combined** | Unified model trained on both intrinsic and situational data. Best for compositional style descriptions. | [ajd12342/paraspeechclap-combined](https://huggingface.co/ajd12342/paraspeechclap-combined) |

**Architecture:** WavLM-Large (317M) speech encoder + Granite Embedding (278M) text encoder, with projection heads mapping to a shared 768-dimensional embedding space.

## Datasets

All models are trained on [ParaSpeechCaps](https://huggingface.co/datasets/ajd12342/paraspeechcaps).

| Dataset | HuggingFace |
|---|---|
| Intrinsic Training | [ajd12342/paraspeechcaps-intrinsic-train](https://huggingface.co/datasets/ajd12342/paraspeechcaps-intrinsic-train) |
| Situational Training | [ajd12342/paraspeechcaps-situational-train](https://huggingface.co/datasets/ajd12342/paraspeechcaps-situational-train) |
| Intrinsic Eval | [ajd12342/paraspeechclap-eval-intrinsic](https://huggingface.co/datasets/ajd12342/paraspeechclap-eval-intrinsic) |
| Situational Eval | [ajd12342/paraspeechclap-eval-situational](https://huggingface.co/datasets/ajd12342/paraspeechclap-eval-situational) |
| Combined Eval | [ajd12342/paraspeechclap-eval-combined](https://huggingface.co/datasets/ajd12342/paraspeechclap-eval-combined) |

## Installation

```bash
git clone https://github.com/ajd12342/paraspeechclap.git
cd paraspeechclap
pip install -r requirements.txt
```

## Download Models

```bash
mkdir -p checkpoints
huggingface-cli download ajd12342/paraspeechclap-intrinsic paraspeechclap-intrinsic.pth.tar --local-dir checkpoints
huggingface-cli download ajd12342/paraspeechclap-situational paraspeechclap-situational.pth.tar --local-dir checkpoints
huggingface-cli download ajd12342/paraspeechclap-combined paraspeechclap-combined.pth.tar --local-dir checkpoints
```

All examples below assume checkpoints are stored in `./checkpoints/`.

## Quick Start: Inference

### Command-line

```bash
# Intrinsic (speaker-level): similarity with a style description
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-intrinsic.pth.tar \
  --audio_path /path/to/audio.wav \
  --text "A person speaks in a deep, guttural tone."

# Intrinsic: zero-shot classification across candidate styles
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-intrinsic.pth.tar \
  --audio_path /path/to/audio.wav \
  --candidates deep shrill nasal husky raspy

# Situational (utterance-level): similarity with an emotion/speaking-style description
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-situational.pth.tar \
  --audio_path /path/to/audio.wav \
  --text "A person is speaking in a whispered style."

# Situational: zero-shot classification across emotion/speaking-style candidates
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-situational.pth.tar \
  --audio_path /path/to/audio.wav \
  --candidates angry happy calm whispered enthusiastic saddened anxious

# Combined (compositional): similarity with a description mixing both attribute types
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-combined.pth.tar \
  --audio_path /path/to/audio.wav \
  --text "A person with a deep, raspy voice is speaking in a whispered style."

# Combined: zero-shot classification (intrinsic or situational candidates)
python scripts/inference.py \
  --checkpoint_path ./checkpoints/paraspeechclap-combined.pth.tar \
  --audio_path /path/to/audio.wav \
  --candidates angry happy calm whispered enthusiastic saddened anxious
```

### Python

The model loading and audio preprocessing is the same for all ParaSpeechCLAP models. The only difference is the checkpoint path and what you query with.

```python
import torch
import torchaudio
import torchaudio.transforms as T
from paraspeechclap.model import CLAP
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

SPEECH_MODEL = "microsoft/wavlm-large"
TEXT_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a ParaSpeechCLAP model — swap the checkpoint path for intrinsic/situational/combined
model = CLAP(
    speech_name=SPEECH_MODEL,
    text_name=TEXT_MODEL,
    embedding_dim=768,
)
state_dict = torch.load("./checkpoints/paraspeechclap-intrinsic.pth.tar", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()

# Initialize preprocessors
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SPEECH_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

# Load and preprocess audio (resample to 16 kHz mono)
waveform, sr = torchaudio.load("path/to/audio.wav")
if sr != 16000:
    waveform = T.Resample(sr, 16000)(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
audio = feature_extractor(
    waveform.squeeze(0), sampling_rate=16000, return_tensors="pt"
).input_values.to(DEVICE)  # (1, num_samples)

with torch.no_grad():
    audio_emb = model.get_audio_embedding(audio, normalize=True)  # (1, 768)
```

#### Intrinsic (speaker-level) styles

```python
# Similarity with a free-form intrinsic description
text_tokens = tokenizer(
    "A person speaks in a deep, guttural tone.",
    return_tensors="pt", padding=True, truncation=True, max_length=512
)
text_tokens = {k: v.to(DEVICE) for k, v in text_tokens.items()}

with torch.no_grad():
    text_emb = model.get_text_embedding(text_tokens, normalize=True)  # (1, 768)
    similarity = (audio_emb @ text_emb.T).item()
    print(f"Similarity: {similarity:.4f}")

# Zero-shot classification across intrinsic candidate styles
candidates = ["deep", "shrill", "nasal", "husky", "raspy"]
prompts = [f"A person is speaking in a {s} style." for s in candidates]
tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

with torch.no_grad():
    text_embs = model.get_text_embedding(tokens, normalize=True)  # (5, 768)
    scores = (audio_emb @ text_embs.T).squeeze(0)  # (5,)
    pred = candidates[scores.argmax().item()]
    print(f"Predicted style: {pred}")
```

#### Situational (utterance-level) styles

```python
# Similarity with a free-form situational description
text_tokens = tokenizer(
    "A person is speaking in a whispered style.",
    return_tensors="pt", padding=True, truncation=True, max_length=512
)
text_tokens = {k: v.to(DEVICE) for k, v in text_tokens.items()}

with torch.no_grad():
    text_emb = model.get_text_embedding(text_tokens, normalize=True)  # (1, 768)
    similarity = (audio_emb @ text_emb.T).item()
    print(f"Similarity: {similarity:.4f}")

# Zero-shot classification across situational candidate styles
candidates = ["angry", "happy", "calm", "whispered", "enthusiastic", "saddened", "anxious"]
prompts = [f"A person is speaking in a {s} style." for s in candidates]
tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

with torch.no_grad():
    text_embs = model.get_text_embedding(tokens, normalize=True)  # (7, 768)
    scores = (audio_emb @ text_embs.T).squeeze(0)  # (7,)
    pred = candidates[scores.argmax().item()]
    print(f"Predicted style: {pred}")
```

#### Combined (compositional): intrinsic + situational

```python
# Similarity with a compositional description (intrinsic + situational)
text_tokens = tokenizer(
    "A person with a deep, raspy voice is speaking in a whispered style.",
    return_tensors="pt", padding=True, truncation=True, max_length=512
)
text_tokens = {k: v.to(DEVICE) for k, v in text_tokens.items()}

with torch.no_grad():
    text_emb = model.get_text_embedding(text_tokens, normalize=True)  # (1, 768)
    similarity = (audio_emb @ text_emb.T).item()
    print(f"Similarity: {similarity:.4f}")

# Zero-shot classification — works for both intrinsic and situational candidates
candidates = ["angry", "happy", "calm", "whispered", "enthusiastic", "saddened", "anxious"]
prompts = [f"A person is speaking in a {s} style." for s in candidates]
tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

with torch.no_grad():
    text_embs = model.get_text_embedding(tokens, normalize=True)  # (7, 768)
    scores = (audio_emb @ text_embs.T).squeeze(0)  # (7,)
    pred = candidates[scores.argmax().item()]
    print(f"Predicted style: {pred}")
```

## Evaluation

Evaluation scripts require audio files to be present locally. Pass `data.audio_root=/path/to/audio_root` to point to a directory organized as `${audio_root}/{source}/`, where `{source}` matches the `source` column in each dataset (e.g., `voxceleb`, `expresso`, `ears`). See the [ParaSpeechCaps audio setup instructions](https://github.com/ajd12342/paraspeechcaps/tree/main/dataset#22-processing-dataset-audio) for how to download and organize each source.

### Intrinsic

```bash
# Retrieval (R@1, R@10, Median Rank)
python scripts/evaluate_retrieval.py \
  --config-name eval/retrieval \
  checkpoint_path=./checkpoints/paraspeechclap-intrinsic.pth.tar \
  data.dataset_name=ajd12342/paraspeechclap-eval-intrinsic \
  data.audio_root=/path/to/audio_root \
  meta.results=./results/retrieval/paraspeechclap-eval-intrinsic/ajd12342-paraspeechclap-intrinsic

# Per-attribute classification (UAR, Macro F1)
for attr in clarity pitch rhythm texture volume; do
  python scripts/evaluate_classification.py \
    --config-name eval/classification/${attr} \
    checkpoint_path=./checkpoints/paraspeechclap-intrinsic.pth.tar \
    data.audio_root=/path/to/audio_root \
    meta.results=./results/classification/paraspeechclap-eval-intrinsic/ajd12342-paraspeechclap-intrinsic/${attr}
done
```

Available classification configs: `eval/classification/pitch`, `eval/classification/texture`, `eval/classification/volume`, `eval/classification/clarity`, `eval/classification/rhythm`. Each loads the corresponding split from `ajd12342/paraspeechclap-eval-intrinsic` (e.g., `classification_pitch`).

### Situational

```bash
# Retrieval (R@1, R@10, Median Rank)
python scripts/evaluate_retrieval.py \
  --config-name eval/retrieval \
  checkpoint_path=./checkpoints/paraspeechclap-situational.pth.tar \
  data.dataset_name=ajd12342/paraspeechclap-eval-situational \
  data.audio_root=/path/to/audio_root \
  meta.results=./results/retrieval/paraspeechclap-eval-situational/ajd12342-paraspeechclap-situational

# Classification (UAR, Macro F1 — 21 situational classes)
python scripts/evaluate_classification.py \
  --config-name eval/classification/situational \
  checkpoint_path=./checkpoints/paraspeechclap-situational.pth.tar \
  data.audio_root=/path/to/audio_root \
  meta.results=./results/classification/paraspeechclap-eval-situational/ajd12342-paraspeechclap-situational/
```

### Combined

```bash
# Retrieval (R@1, R@10, Median Rank) — compositional descriptions
python scripts/evaluate_retrieval.py \
  --config-name eval/retrieval \
  checkpoint_path=./checkpoints/paraspeechclap-combined.pth.tar \
  data.dataset_name=ajd12342/paraspeechclap-eval-combined \
  data.audio_root=/path/to/audio_root \
  meta.results=./results/retrieval/paraspeechclap-eval-combined/ajd12342-paraspeechclap-combined
```

## Training

Train ParaSpeechCLAP models using distributed data parallel (DDP) on 4 GPUs:

```bash
# ParaSpeechCLAP-Intrinsic (contrastive + classification loss, class-balanced sampling)
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train/intrinsic \
  data.audio_root=/path/to/audio_root \
  meta.results=./experiments

# ParaSpeechCLAP-Situational (contrastive loss)
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train/situational \
  data.audio_root=/path/to/audio_root \
  meta.results=./experiments

# ParaSpeechCLAP-Combined (contrastive loss, interleaved intrinsic + situational data)
torchrun --nproc_per_node=4 scripts/train.py \
  --config-name train/combined \
  data.audio_root=/path/to/audio_root \
  meta.results=./experiments
```

## Best-of-N Reranking

Use ParaSpeechCLAP as an inference-time reward model to select the best speech clip from N candidates, typically generated by a style-prompted TTS model.

```bash
# Expects pre-generated candidate speech clips organized as:
# /path/to/tts_outputs/
#   iter_1/audios/0.wav, 1.wav, ...
#   iter_2/audios/0.wav, 1.wav, ...
#   ...
#   iter_10/audios/0.wav, 1.wav, ...
#   iter_1/input_descriptions.txt  (one style prompt per line)

# Select best candidates using ParaSpeechCLAP-Intrinsic
python scripts/best_of_n.py \
  checkpoint_path=./checkpoints/paraspeechclap-intrinsic.pth.tar \
  input_base_dir=/path/to/tts_outputs \
  output_dir_name=best_of_N_paraspeechclap_intrinsic

# Select best candidates using ParaSpeechCLAP-Situational
python scripts/best_of_n.py \
  checkpoint_path=./checkpoints/paraspeechclap-situational.pth.tar \
  input_base_dir=/path/to/tts_outputs \
  output_dir_name=best_of_N_paraspeechclap_situational

# Select best candidates using ParaSpeechCLAP-Combined
python scripts/best_of_n.py \
  checkpoint_path=./checkpoints/paraspeechclap-combined.pth.tar \
  input_base_dir=/path/to/tts_outputs \
  output_dir_name=best_of_N_paraspeechclap_combined
```

## Repository Structure

```
paraspeechclap/
├── paraspeechclap/
│   ├── model.py                   # CLAP dual-encoder architecture
│   ├── loss.py                    # ClipLoss and MultiTaskLoss
│   ├── dataset.py                 # ParaSpeechCaps dataset loader
│   ├── utils.py                   # Collate functions, utilities
│   ├── balanced_sampler.py        # Class-balanced sampling
│   ├── evaluation_utils.py        # Model loading, metric computation
│   └── debug_utils.py             # Logging utilities
├── scripts/
│   ├── inference.py               # Simple inference script
│   ├── train.py                   # DDP training script
│   ├── evaluate_classification.py  # Classification eval
│   ├── evaluate_retrieval.py      # Retrieval eval
│   └── best_of_n.py              # Best-of-N reranking
├── configs/
│   ├── train/
│   │   ├── base.yaml                # Shared training defaults
│   │   ├── intrinsic.yaml
│   │   ├── situational.yaml
│   │   └── combined.yaml
│   ├── eval/
│   │   ├── base.yaml                # Shared evaluation defaults
│   │   ├── retrieval.yaml
│   │   └── classification/
│   │       ├── base.yaml            # Shared classification defaults
│   │       ├── rhythm.yaml
│   │       ├── texture.yaml
│   │       ├── pitch.yaml
│   │       ├── clarity.yaml
│   │       ├── volume.yaml
│   │       └── situational.yaml     # paraspeechclap-eval-situational (21 classes)
│   └── best_of_n/
│       └── base.yaml                # Best-of-N defaults
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@inproceedings{diwan2026paraspeechclap,
  title={{ParaSpeechCLAP}: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining},
  author={Diwan, Anuj and Choi, Eunsol and Harwath, David},
  journal={Under Review},
  year={2026}
}
```

## Acknowledgements

This codebase builds on the following projects:

- [ParaSpeechCaps](https://github.com/ajd12342/paraspeechcaps) — Paralinguistic speech captioning dataset
- [KeiKinn/ParaCLAP](https://github.com/KeiKinn/ParaCLAP) — CLAP model for speech styles

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
