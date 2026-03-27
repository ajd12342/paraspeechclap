import math
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
)
from paraspeechclap.debug_utils import logger, debug_tensor


class Projection(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear2 = torch.nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_out)
        self.drop = torch.nn.Dropout(p)
        logger.debug(f"Initialized Projection with d_in={d_in}, d_out={d_out}, p={p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_tensor("Projection input", x)
        
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        debug_tensor("Projection output", embeds)
        
        return embeds


class SpeechEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        logger.info(f"Initializing SpeechEncoder with model_name={model_name}")
        try:
            self.model_name = model_name
            self.is_wavlm = "wavlm" in self.model_name.lower()

            config = AutoConfig.from_pretrained(self.model_name)
            config.layerdrop = 0.0
            logger.info(f"Disabled LayerDrop by setting config.layerdrop to {config.layerdrop}")

            self.base = AutoModel.from_pretrained(self.model_name, config=config)
            
            if self.is_wavlm:
                self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("Enabled gradient checkpointing for WavLM model.")

            self.hidden_size = self.base.config.hidden_size
            logger.debug(f"SpeechEncoder initialized with hidden_size={self.hidden_size}")
        except Exception as e:
            logger.error(f"Error initializing SpeechEncoder: {e}")
            raise

    def forward(self, x, attention_mask=None):
        debug_tensor("SpeechEncoder input", x)
        if attention_mask is not None:
            debug_tensor("SpeechEncoder attention mask", attention_mask)
        
        try:
            if self.is_wavlm:
                # For WavLM, use return_dict=False to potentially save memory by 
                # not returning extract_features
                output_tuple = self.base(x, attention_mask=attention_mask, return_dict=False)
                last_hidden_state = output_tuple[0]
                debug_tensor("AudioBranch last_hidden_state from WavLM (return_dict=False)", last_hidden_state)
            else:
                output = self.base(x, attention_mask=attention_mask)
                last_hidden_state = output.last_hidden_state
                debug_tensor("AudioBranch last_hidden_state from audio_model", last_hidden_state)
            
            pooled_output = torch.mean(last_hidden_state, dim=1)
            debug_tensor("After mean pooling", pooled_output)
            
            return pooled_output
        except Exception as e:
            logger.error(f"Error in SpeechEncoder forward pass: {e}")
            raise


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        logger.info(f"Initializing TextEncoder with model_name={model_name}")
        try:
            self.base = AutoModel.from_pretrained(model_name)
            logger.debug(f"TextEncoder initialized with config=", extra={"config": self.base.config.to_dict()})
        except Exception as e:
            logger.error(f"Error initializing TextEncoder: {e}")
            raise

    def forward(self, x):
        logger.debug("TextEncoder input keys", extra={"keys": list(x.keys())})
        
        try:
            outputs = self.base(**x)
            
            # Consistently use CLS token from last_hidden_state as per user clarification
            hidden_states = outputs.last_hidden_state # Or outputs[0]
            out = hidden_states[:, 0, :]
            debug_tensor("Using CLS token from last_hidden_state", out)
            
            return out
        except Exception as e:
            logger.error(f"Error in TextEncoder forward pass: {e}")
            raise


class CLAP(torch.nn.Module):
    def __init__(self, speech_name: str, text_name: str, embedding_dim: int = 768, projection_dropout: float = 0.5):
        super().__init__()
        logger.info(f"Initializing CLAP with speech_name={speech_name}, text_name={text_name}, embedding_dim={embedding_dim}, projection_dropout={projection_dropout}")

        try:
            self.audio_branch = SpeechEncoder(model_name=speech_name)
            logger.debug("Audio branch initialized")

            self.text_branch = TextEncoder(model_name=text_name)
            logger.debug("Text branch initialized")

            self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim, p=projection_dropout)
            logger.debug(f"Audio projection initialized with input_dim={self.audio_branch.hidden_size}, output_dim={embedding_dim}")

            self.text_projection = Projection(self.text_branch.base.config.hidden_size, embedding_dim, p=projection_dropout)
            logger.debug(f"Text projection initialized with input_dim={self.text_branch.base.config.hidden_size}, output_dim={embedding_dim}")

            # Initialize logit scale (stores log(1/temperature))
            self.log_logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
            logger.debug(f"Log logit scale initialized to {self.log_logit_scale.item()}")
            
        except Exception as e:
            logger.error(f"Error initializing CLAP model: {e}")
            raise


    def forward(self, audio, text, audio_attention_mask=None):
        debug_tensor("CLAP audio input", audio)
        if audio_attention_mask is not None:
            debug_tensor("CLAP audio attention mask", audio_attention_mask)
        logger.debug("CLAP text input keys", extra={"keys": list(text.keys()) if isinstance(text, dict) else "Not a dict"})
        
        try:
            # Process audio, passing the attention mask
            speech_emb = self.audio_branch(audio, attention_mask=audio_attention_mask)
            debug_tensor("Raw speech embeddings", speech_emb)
            
            # Process text
            text_emb = self.text_branch(text)
            debug_tensor("Raw text embeddings", text_emb)
            
            # Project embeddings
            speech_emb = self.audio_projection(speech_emb)
            debug_tensor("Projected speech embeddings", speech_emb)
            
            text_emb = self.text_projection(text_emb)
            debug_tensor("Projected text embeddings", text_emb)
            
            # Calculate the *actual* scaling factor (exponentiated log value)
            logit_scale = self.log_logit_scale.exp()
            logger.debug(f"Logit scale (actual value): {logit_scale.item()}")
            
            # Return embeddings and the actual logit_scale
            return text_emb, speech_emb, logit_scale
            
        except Exception as e:
            logger.error(f"Error in CLAP forward pass: {e}")
            raise

    def get_audio_embedding(self, audio: torch.Tensor, attention_mask: torch.Tensor = None, normalize: bool = True) -> torch.Tensor:
        """
        Computes the audio embedding for the given audio input.

        Args:
            audio (torch.Tensor): Raw audio waveform tensor.
            attention_mask (torch.Tensor, optional): Attention mask for the audio. Defaults to None.
            normalize (bool, optional): Whether to L2 normalize the final embedding. Defaults to True.

        Returns:
            torch.Tensor: The computed audio embedding.
        """
        debug_tensor("get_audio_embedding input", audio)
        if attention_mask is not None:
             debug_tensor("get_audio_embedding attention_mask", attention_mask)
        try:
            speech_emb = self.audio_branch(audio, attention_mask=attention_mask)
            speech_emb = self.audio_projection(speech_emb)
            if normalize:
                speech_emb = F.normalize(speech_emb, dim=-1)
                debug_tensor("Normalized projected speech embeddings", speech_emb)
            else:
                 debug_tensor("Unnormalized projected speech embeddings", speech_emb)
            return speech_emb
        except Exception as e:
            logger.error(f"Error in get_audio_embedding: {e}")
            raise

    def get_text_embedding(self, text_input: dict, normalize: bool = True) -> torch.Tensor:
        """
        Computes the text embedding for the given tokenized text input.

        Args:
            text_input (dict): A dictionary containing tokenized text output
                                (e.g., from tokenizer.batch_encode_plus).
                                Expected keys like 'input_ids', 'attention_mask'.
            normalize (bool, optional): Whether to L2 normalize the final embedding. Defaults to True.

        Returns:
            torch.Tensor: The computed text embedding (CLS token representation).
        """
        logger.debug("get_text_embedding input keys", extra={"keys": list(text_input.keys())})
        try:
            text_emb = self.text_branch(text_input)
            text_emb = self.text_projection(text_emb)
            if normalize:
                text_emb = F.normalize(text_emb, dim=-1)
                debug_tensor("Normalized projected text embeddings", text_emb)
            else:
                 debug_tensor("Unnormalized projected text embeddings", text_emb)
            return text_emb
        except Exception as e:
            logger.error(f"Error in get_text_embedding: {e}")
            raise


