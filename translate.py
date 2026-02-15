import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import BilingualDataset

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def sample_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, top_k=50, top_p=0.9, temperature=1.0):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    word_probabilities = []

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # 1. Get Logits and scale by temperature
        logits = model.project(out[:, -1]) / temperature
        
        # 2. Apply Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # 3. Apply Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token that exceeds the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')

        # 4. Final Sampling
        probs = F.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, num_samples=1)
        
        # Confidence of the specific chosen word
        word_probabilities.append(probs[0, next_word.item()].item() * 100)
        
        decoder_input = torch.cat([decoder_input, next_word], dim=1)
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0), word_probabilities

def translate(input_sentence, top_k=50, top_p=0.9, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), 
                              config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename:
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        return

    model.eval()
    with torch.no_grad():
        source_ids = tokenizer_src.encode(input_sentence).ids
        source_ids = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(source_ids) - 2), dtype=torch.int64)
        ]).to(device).unsqueeze(0)
        source_mask = (source_ids != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        translated_ids, probs = sample_decode(model, source_ids, source_mask, tokenizer_src, tokenizer_tgt, 
                                              config['seq_len'], device, top_k=top_k, top_p=top_p, temperature=temperature)

        words = [tokenizer_tgt.id_to_token(idx.item()) for idx in translated_ids[1:] if tokenizer_tgt.id_to_token(idx.item()) != '[EOS]']
        
        print(f"\nEnglish: {input_sentence}")
        print("Italian: ", end="")
        for word, p in zip(words, probs):
            print(f"{word}({p:.1f}%) ", end="")
        print("\n" + "="*50)


if __name__ == "__main__":
    while True:
        sentence = input("\nEnter English sentence (or 'exit'): ")
        if sentence.lower() == 'exit': break
        translate(sentence, top_k=50, top_p=0.9, temperature=0.8)