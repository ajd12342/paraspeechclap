"""
ParaSpeechCLAP Inference Script

Computes speech-text similarity using a ParaSpeechCLAP model checkpoint.
Can be used for:
  1. Computing similarity between a speech clip and a text style description
  2. Ranking multiple text descriptions for a speech clip (retrieval)
  3. Classifying a speech clip into one of several style categories

Usage:
  # Intrinsic (speaker-level) similarity:
  python scripts/inference.py \
    --checkpoint_path ./checkpoints/paraspeechclap-intrinsic.pth.tar \
    --audio_path /path/to/audio.wav \
    --text "A person speaks in a deep, guttural tone."

  # Intrinsic classification with multiple candidate styles:
  python scripts/inference.py \
    --checkpoint_path ./checkpoints/paraspeechclap-intrinsic.pth.tar \
    --audio_path /path/to/audio.wav \
    --candidates "deep" "shrill" "nasal" "husky" "raspy"

  # Situational (utterance-level) similarity:
  python scripts/inference.py \
    --checkpoint_path ./checkpoints/paraspeechclap-situational.pth.tar \
    --audio_path /path/to/audio.wav \
    --text "A person is speaking in a whispered style."

  # Situational classification across emotion/speaking-style candidates:
  python scripts/inference.py \
    --checkpoint_path ./checkpoints/paraspeechclap-situational.pth.tar \
    --audio_path /path/to/audio.wav \
    --candidates "angry" "happy" "calm" "whispered" "enthusiastic" "saddened" "anxious"
"""

import argparse
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from paraspeechclap.model import CLAP
from paraspeechclap.evaluation_utils import CLASSIFICATION_TEMPLATE
from paraspeechclap.utils import TARGET_SR

# Default model configuration matching the paper
DEFAULT_SPEECH_MODEL = "microsoft/wavlm-large"
DEFAULT_TEXT_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEFAULT_EMBEDDING_DIM = 768


def load_model(checkpoint_path, device, speech_model=None, text_model=None, embedding_dim=None):
    """Load a ParaSpeechCLAP model from a checkpoint."""
    speech_model = speech_model or DEFAULT_SPEECH_MODEL
    text_model = text_model or DEFAULT_TEXT_MODEL
    embedding_dim = embedding_dim or DEFAULT_EMBEDDING_DIM

    model = CLAP(
        speech_name=speech_model,
        text_name=text_model,
        embedding_dim=embedding_dim,
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        import logging
        logging.warning(f"Strict loading failed ({e}), falling back to non-strict loading.")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logging.warning(f"Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            logging.warning(f"Unexpected keys: {result.unexpected_keys}")
    model.to(device)
    model.eval()
    return model


def load_audio(audio_path, feature_extractor):
    """Load and preprocess an audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze(0)

    # Normalize with feature extractor
    inputs = feature_extractor(
        waveform,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="do_not_pad",
    )
    return inputs.input_values.squeeze(0)


def compute_similarity(model, audio_tensor, text_strings, tokenizer, device):
    """Compute cosine similarity between one audio clip and one or more text descriptions."""
    # Get audio embedding
    audio_input = audio_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        audio_emb = model.get_audio_embedding(audio_input, normalize=True)

    # Get text embeddings
    text_tokens = tokenizer.batch_encode_plus(
        text_strings,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_emb = model.get_text_embedding(text_tokens, normalize=True)

    # Cosine similarity
    similarities = (audio_emb @ text_emb.T).squeeze(0)
    return similarities


def main():
    parser = argparse.ArgumentParser(description="ParaSpeechCLAP Inference: Compute speech-text style similarity")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to ParaSpeechCLAP model checkpoint")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file (.wav)")
    parser.add_argument("--text", type=str, default=None, help="Text style description to compare against")
    parser.add_argument("--candidates", type=str, nargs="+", default=None,
                        help="Candidate style labels for classification (e.g., deep shrill nasal)")
    parser.add_argument("--speech_model", type=str, default=DEFAULT_SPEECH_MODEL)
    parser.add_argument("--text_model", type=str, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if not specified)")
    args = parser.parse_args()

    if args.text is None and args.candidates is None:
        parser.error("Must provide either --text or --candidates")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model
    print(f"Loading ParaSpeechCLAP model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, device, args.speech_model, args.text_model, args.embedding_dim)

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.speech_model)

    # Load audio
    print(f"Loading audio from {args.audio_path}...")
    audio_tensor = load_audio(args.audio_path, feature_extractor)

    if args.text:
        # Single text similarity
        similarity = compute_similarity(model, audio_tensor, [args.text], tokenizer, device)
        print(f"\nSimilarity between audio and \"{args.text}\": {similarity.item():.4f}")

    if args.candidates:
        text_prompts = [CLASSIFICATION_TEMPLATE.format(c) for c in args.candidates]
        similarities = compute_similarity(model, audio_tensor, text_prompts, tokenizer, device)

        # Apply softmax for probabilities
        probs = F.softmax(similarities, dim=0)

        print(f"\nClassification Results:")
        print(f"{'Style':<20} {'Similarity':>12} {'Probability':>12}")
        print("-" * 46)
        for candidate, sim, prob in sorted(
            zip(args.candidates, similarities.tolist(), probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"{candidate:<20} {sim:>12.4f} {prob:>11.1%}")

        best_idx = similarities.argmax().item()
        print(f"\nPredicted style: {args.candidates[best_idx]}")


if __name__ == "__main__":
    main()
