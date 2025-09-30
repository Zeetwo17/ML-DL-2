# Image Captioning Network (PyTorch + TIMM + Transformer Decoder)

An end-to-end **image captioning** pipeline built in a single notebook: a **CNN image encoder** (from `timm`) projects image features into a latent space, which a **Transformer Decoder** (PyTorch `nn.TransformerDecoder`) uses to generate captions token-by-token. Tokenization is handled with a Hugging Face `AutoTokenizer`.

> Optimized for Google Colab (works locally too).

---

##  Highlights

- **Encoder–Decoder**: EfficientNet-B0 (feature extractor) → projection → Transformer Decoder.
- **Tokenizer**: `bert-base-uncased` (subword/BPE) via `transformers.AutoTokenizer`.
- **Training-ready**: Albumentations augmentations, teacher forcing, padding & masks.
- **Simple inference**: Greedy decoding loop with BOS/CLS → EOS/SEP.
- **Flexible captions file**: Parses multiple common formats for `captions.txt`.

---

##  Requirements

Install these (Colab cell included in the notebook):

```bash
pip install -U timm albumentations transformers opencv-python-headless
