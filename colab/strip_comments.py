import io
import tokenize
from pathlib import Path

def strip_only_comments(source_path: Path, dest_path: Path):
    source = source_path.read_text(encoding="utf-8")
    io_obj = io.StringIO(source)
    out = []
    last_lineno = -1
    last_col = 0
    
    tokens = tokenize.generate_tokens(io_obj.readline)
    for toktype, ttext, (sline, scol), (eline, ecol), ltext in tokens:
        # Solo omitir comentarios reales
        if toktype == tokenize.COMMENT:
            continue
            
        if sline > last_lineno:
            last_col = 0
        if scol > last_col:
            out.append(" " * (scol - last_col))
        
        out.append(ttext)
        last_lineno = eline
        last_col = ecol

    # Escribir código limpio
    raw_code = "".join(out)
    
    # Reducir líneas vacías consecutivas a una sola línea vacía
    clean_lines = []
    for line in raw_code.splitlines():
        if line.strip():
            clean_lines.append(line)
        else:
            if clean_lines and clean_lines[-1].strip():
                clean_lines.append("")
                
    dest_path.write_text("\n".join(clean_lines), encoding="utf-8")

if __name__ == "__main__":
    src = Path("train_lumen.py")
    dest = Path("train_lumen_clean.py")
    strip_only_comments(src, dest)
