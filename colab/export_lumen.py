"""
╔══════════════════════════════════════════════════════════════════════╗
║              LUMEN — Script de Exportación de Modelos                ║
║                                                                      ║
║  Modelo:  Lumen (asistente IA on-device para Nexo / UPLA)           ║
║  Autor:   Alessandro Villogas Gaspar — Auralix Studio               ║
║  Fecha:   Junio 2026                                                 ║
║                                                                      ║
║  Este script toma un checkpoint de entrenamiento (.pt) y lo exporta  ║
║  al formato de distribución final con config.json, safetensors,      ║
║  y el archivo de tokenizador para su uso posterior.                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import sentencepiece as spm

# Re-declaración de la arquitectura para poder cargar el estado
class Config:
    D_MODEL           = 512
    N_HEADS           = 8
    N_LAYERS          = 8
    D_FF              = 2048
    MAX_SEQ_LEN       = 512

# Tokens especiales
TOK_USER  = "<|user|>"
TOK_LUMEN = "<|lumen|>"
TOK_END   = "<|end|>"
TOK_PAD   = "<|pad|>"
TOK_BOS   = "<|bos|>"
TOK_EOS   = "<|eos|>"

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
        
        # Atención
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        import torch.nn.functional as F
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
        import torch.nn.functional as F
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

    def forward(self, x):
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        h = self.dropout(self.token_emb(x))
        for layer in self.layers:
            h = layer(h, self.cos_freqs, self.sin_freqs, mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits, None

def main():
    parser = argparse.ArgumentParser(description="Exportador de checkpoints Lumen")
    parser.add_argument("-c", "--checkpoint", type=str, help="Ruta al archivo de checkpoint (.pt)")
    parser.add_argument("-t", "--tokenizer", type=str, help="Ruta al archivo del tokenizador (.model)")
    parser.add_argument("-o", "--output_dir", type=str, default="./lumen_export", help="Directorio de salida")
    args, _ = parser.parse_known_args()

    work_dir = Path("./lumen_training")

    # 1. Resolver tokenizador
    tok_path = None
    if args.tokenizer:
        tok_path = Path(args.tokenizer)
    else:
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

    # 2. Resolver checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        search_paths = [
            work_dir / "lumen_final" / "lumen_model.pt",
            work_dir / "checkpoints" / "lumen_best.pt",
            Path("/content/drive/MyDrive/lumen_models/lumen_final/lumen_model.pt"),
            Path("/content/drive/MyDrive/lumen_models/checkpoints/lumen_best.pt")
        ]
        if (work_dir / "checkpoints").exists():
            search_paths.extend(sorted((work_dir / "checkpoints").glob("lumen_step_*.pt"), reverse=True))
        for p in search_paths:
            if p.exists():
                ckpt_path = p
                break

    if not ckpt_path or not ckpt_path.exists():
        print("[ERROR] No se encontró ningún checkpoint (.pt). Especifícalo con --checkpoint <ruta>")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cargando tokenizador...")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tok_path))
    vocab_size = tokenizer.get_piece_size()

    print(f"[INFO] Inicializando modelo...")
    model = LumenModel(
        vocab_size=vocab_size,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN
    )

    print(f"[INFO] Cargando pesos del checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Manejar formatos de guardado (diccionario completo de checkpoint o solo state dict)
    state_dict = checkpoint
    step = "desconocido"
    loss = "desconocido"
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step", "desconocido")
        loss = checkpoint.get("loss", "desconocido")
        print(f"[OK] Cargado checkpoint de entrenamiento (Paso: {step}, Loss: {loss})")
    else:
        print("[OK] Cargado state_dict crudo")

    model.load_state_dict(state_dict)
    model.eval()

    # 1. Guardar pesos PyTorch crudos (.pt)
    pt_path = out_dir / "lumen_model.pt"
    torch.save(state_dict, pt_path)
    print(f"[OK] Pesos PyTorch exportados a: {pt_path} ({pt_path.stat().st_size / 1e6:.2f} MB)")

    # 2. Guardar pesos en formato SafeTensors (.safetensors)
    try:
        from safetensors.torch import save_model
        st_path = out_dir / "lumen_model.safetensors"
        save_model(model, str(st_path))
        print(f"[OK] SafeTensors exportado a: {st_path} ({st_path.stat().st_size / 1e6:.2f} MB)")
    except ImportError:
        print("[WARN] Biblioteca 'safetensors' no instalada. Omitiendo conversión a safetensors.")

    # 3. Guardar archivo config.json
    config = {
        "model_name": "Lumen",
        "creator": "Alessandro Villogas Gaspar",
        "organization": "Auralix Studio",
        "description": "Asistente IA on-device para Nexo / UPLA",
        "architecture": "decoder-only-transformer",
        "vocab_size": vocab_size,
        "d_model": Config.D_MODEL,
        "n_heads": Config.N_HEADS,
        "n_layers": Config.N_LAYERS,
        "d_ff": Config.D_FF,
        "max_seq_len": Config.MAX_SEQ_LEN,
        "special_tokens": {
            "pad": TOK_PAD,
            "bos": TOK_BOS,
            "eos": TOK_EOS,
            "user": TOK_USER,
            "lumen": TOK_LUMEN,
            "end": TOK_END
        }
    }
    
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[OK] Configuración exportada a: {config_path}")

    # 4. Copiar archivos del tokenizador
    import shutil
    shutil.copy2(tok_path, out_dir / "tokenizer.model")
    print(f"[OK] Tokenizador copiado a: {out_dir / 'tokenizer.model'}")
    
    # Si hay vocabulario, copiarlo también
    vocab_path = tok_path.with_suffix(".vocab")
    if vocab_path.exists():
        shutil.copy2(vocab_path, out_dir / "tokenizer.vocab")
        print(f"[OK] Vocabulario del tokenizador copiado a: {out_dir / 'tokenizer.vocab'}")

    print("\n" + "="*60)
    print("                 EXPORTACIÓN COMPLETADA CON ÉXITO")
    print(f"   Carpeta final: {out_dir.resolve()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
