import streamlit as st
import torch
import torch.nn as nn
from model import build_transformer
from config import get_config, latest_weights_file_path
from tokenizers import Tokenizer

# --- PAGE SETUP ---
st.set_page_config(page_title="Translator Demo", page_icon="🇮🇹")
st.title("🇮🇹 Transformer Translator")
st.subheader("English to Italian")

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_all_assets():
    config = get_config()
    # Load Tokenizers
    tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file("tokenizer_it.json")
    
    # Build Model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config["seq_len"], 
        config["seq_len"], 
        d_model=config["d_model"]
    )
    
    # Load weights - use latest checkpoint, map to CPU for local demo
    model_path = latest_weights_file_path(config)
    if not model_path:
        raise FileNotFoundError("No checkpoint found in weights/. Run train.py first.")
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    return model, tokenizer_src, tokenizer_tgt, config

# --- INFERENCE ENGINE ---
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def translate(sentence, model, tokenizer_src, tokenizer_tgt, config, device):
    model.eval()
    with torch.no_grad():
        # Pre-process source text
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        ], dim=0).to(device).unsqueeze(0)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # 1. Encoder pass
        encoder_output = model.encode(source, source_mask)

        # 2. Iterative Decoding (Greedy)
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        
        while True:
            if decoder_input.size(1) == config['seq_len']:
                break
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            
            # FIXED: Use model.decode to include embedding logic
            out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
            
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

        return tokenizer_tgt.decode(decoder_input.squeeze(0).cpu().numpy())

# --- UI LOGIC ---
try:
    model, tokenizer_src, tokenizer_tgt, config = load_all_assets()
    device = torch.device("cpu") # Force CPU for local demo stability

    input_text = st.text_area("English Input:", height=150, placeholder="Enter a sentence to translate...")

    if st.button("Translate"):
        if input_text:
            with st.spinner("Generating translation..."):
                translation = translate(input_text, model, tokenizer_src, tokenizer_tgt, config, device)
                st.success(f"**Italian Translation:** {translation}")
        else:
            st.error("Please enter a sentence.")

except FileNotFoundError:
    st.error("Required files (weights or tokenizers) not found in the local directory.")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("Model Info")
    st.info("Architecture: Transformer (Encoder-Decoder)")
    st.write(f"Vocab Size (EN): {tokenizer_src.get_vocab_size() if 'tokenizer_src' in locals() else 'N/A'}")
    st.write(f"Vocab Size (IT): {tokenizer_tgt.get_vocab_size() if 'tokenizer_tgt' in locals() else 'N/A'}")