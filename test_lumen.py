"""
╔══════════════════════════════════════════════════════════════════════╗
║              LUMEN — Script de Inferencia e Interacción             ║
║                                                                      ║
║  Modelo:  Lumen (asistente IA on-device para Nexo / UPLA)           ║
║  Autor:   Alessandro Villogas Gaspar — Auralix Studio               ║
║  Fecha:   Junio 2026                                                 ║
║                                                                      ║
║  Este script carga un checkpoint del modelo Lumen (.pt) y permite    ║
║  chatear de manera interactiva o generar respuestas a preguntas.     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# Tokens especiales
TOK_USER  = "<|user|>"
TOK_LUMEN = "<|lumen|>"
TOK_END   = "<|end|>"
TOK_PAD   = "<|pad|>"
TOK_BOS   = "<|bos|>"
TOK_EOS   = "<|eos|>"

# Arquitectura por defecto (debe coincidir con la de train_lumen.py)
class Config:
    D_MODEL           = 512
    N_HEADS           = 8
    N_LAYERS          = 8
    D_FF              = 2048
    MAX_SEQ_LEN       = 512

# === Capas del Transformer ===

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return cos_freqs, sin_freqs

def apply_rope(x, cos_freqs, sin_freqs):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    seq_len = x.shape[2]
    cos_f = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
    sin_f = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x2 * cos_f + x1 * sin_f
    return torch.cat([out1, out2], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos_freqs, sin_freqs, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos_freqs, sin_freqs)
        k = apply_rope(k, cos_freqs, sin_freqs)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, cos_freqs, sin_freqs, mask=None):
        x = x + self.attn(self.attn_norm(x), cos_freqs, sin_freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LumenModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 8, d_ff: int = 2048, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(0.0)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, 0.0)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        cos_freqs, sin_freqs = precompute_rope_freqs(d_model // n_heads, max_seq_len)
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_seq_len, f"Secuencia {T} > max {self.max_seq_len}"
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        h = self.dropout(self.token_emb(x))
        for layer in self.layers:
            h = layer(h, self.cos_freqs, self.sin_freqs, mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits, None

# === Generación ===

@torch.no_grad()
def generate(
    model: LumenModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    bos_id: int = 1,
    eos_id: int = 2,
    end_id: int = 5,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = [bos_id] + ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        if input_ids.shape[1] > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]

        logits, _ = model(input_ids)
        logits = logits[:, -1, :]
        
        # Ajuste de temperatura
        if temperature > 0:
            logits = logits / temperature
        else:
            # Argmax si temp es 0
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if next_token.item() in (eos_id, end_id):
                break
            input_ids = torch.cat([input_ids, next_token], dim=1)
            continue

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative - probs > top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() in (eos_id, end_id):
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    output_ids = input_ids[0].tolist()[len(ids):]
    return tokenizer.decode(output_ids)

def main():
    parser = argparse.ArgumentParser(description="Inferencia interactiva de Lumen")
    parser.add_argument("-c", "--checkpoint", type=str, help="Ruta al checkpoint (.pt)")
    parser.add_argument("-t", "--tokenizer", type=str, help="Ruta al tokenizador (.model)")
    parser.add_argument("-p", "--prompt", type=str, help="Ejecutar una consulta y salir")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperatura de generación (0.0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtrado")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleo")
    parser.add_argument("--max_tokens", type=int, default=256, help="Máximo de tokens a generar")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Usando dispositivo: {device}")

    # Rutas por defecto
    work_dir = Path("./lumen_training")
    
    # 1. Resolver tokenizador
    tok_path = None
    if args.tokenizer:
        tok_path = Path(args.tokenizer)
    else:
        # Buscar en ubicaciones típicas
        search_paths = [
            work_dir / "lumen_final" / "tokenizer.model",
            work_dir / "lumen_tokenizer.model",
            Path("/content/drive/MyDrive/lumen_models/lumen_final/tokenizer.model"),
            Path("./tokenizer.model")
        ]
        for p in search_paths:
            if p.exists():
                tok_path = p
                break

    if not tok_path or not tok_path.exists():
        print("[ERROR] No se encontró el tokenizador (tokenizer.model). Especifícalo con --tokenizer <ruta>")
        sys.exit(1)

    print(f"[OK] Cargando tokenizador: {tok_path}")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tok_path))

    # Identificar IDs especiales
    pad_id = tokenizer.piece_to_id(TOK_PAD)
    bos_id = tokenizer.piece_to_id(TOK_BOS)
    eos_id = tokenizer.piece_to_id(TOK_EOS)
    user_id = tokenizer.piece_to_id(TOK_USER)
    lumen_id = tokenizer.piece_to_id(TOK_LUMEN)
    end_id = tokenizer.piece_to_id(TOK_END)

    # 2. Resolver checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Buscar el checkpoint más reciente o el mejor
        search_paths = [
            work_dir / "lumen_final" / "lumen_model.pt",
            work_dir / "checkpoints" / "lumen_best.pt",
            Path("/content/drive/MyDrive/lumen_models/lumen_final/lumen_model.pt"),
            Path("/content/drive/MyDrive/lumen_models/checkpoints/lumen_best.pt")
        ]
        # También buscar en checkpoints genéricos
        if (work_dir / "checkpoints").exists():
            search_paths.extend(sorted((work_dir / "checkpoints").glob("lumen_step_*.pt"), reverse=True))
        
        for p in search_paths:
            if p.exists():
                ckpt_path = p
                break

    if not ckpt_path or not ckpt_path.exists():
        print("[ERROR] No se encontró ningún checkpoint (.pt). Especifícalo con --checkpoint <ruta>")
        sys.exit(1)

    print(f"[OK] Cargando modelo desde: {ckpt_path}")
    
    # 3. Inicializar modelo
    model = LumenModel(
        vocab_size=tokenizer.get_piece_size(),
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN
    )

    # Cargar pesos (soporta tanto checkpoints de entrenamiento como exportados)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "desconocido")
        loss = checkpoint.get("loss", 0.0)
        print(f"[OK] Checkpoint de entrenamiento cargado (Paso: {step}, Loss: {loss:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Pesos finales cargados correctamente")

    model.to(device)
    model.eval()

    # 4. Modo de ejecución
    if args.prompt:
        # Ejecutar consulta única
        prompt_formatted = f"{TOK_USER} {args.prompt} {TOK_LUMEN}"
        print(f"\n[Pregunta]: {args.prompt}")
        response = generate(
            model, tokenizer, prompt_formatted, device,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            bos_id=bos_id,
            eos_id=eos_id,
            end_id=end_id
        )
        print(f"[Lumen]: {response}\n")
    else:
        # Modo chat interactivo
        print("\n" + "=" * 60)
        print("                 CHAT INTERACTIVO CON LUMEN")
        print("   Pregúntame sobre la UPLA, Nexo, o hazme consultas.")
        print("   (Escribe 'salir', 'exit' o 'quit' para terminar)")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Tú: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ("salir", "exit", "quit"):
                print("\n¡Hasta luego! Éxitos con la universidad. ✨\n")
                break

            prompt_formatted = f"{TOK_USER} {user_input} {TOK_LUMEN}"
            response = generate(
                model, tokenizer, prompt_formatted, device,
                max_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                bos_id=bos_id,
                eos_id=eos_id,
                end_id=end_id
            )
            print(f"Lumen: {response}\n")

if __name__ == "__main__":
    main()
