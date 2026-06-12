"""
Lumen — Entrenamiento local (Windows / CPU)
Alessandro Villogas Gaspar — Auralix Studio — Junio 2026

Uso:
    py -3.12 train_local.py
"""

import os, math, json, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm

print(f"PyTorch {torch.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(24)           # usar todos los CPUs
torch.set_num_interop_threads(4)
print(f"Device: {DEVICE} | Threads: {torch.get_num_threads()}")

# ─── Configuración ────────────────────────────────────────────────────────────

class Config:
    WORK_DIR         = Path(__file__).parent.parent / "lumen_training"
    CORPUS_FILE      = WORK_DIR / "corpus.txt"
    TOKENIZER_PREFIX = WORK_DIR / "lumen_tokenizer"
    CHECKPOINT_DIR   = WORK_DIR / "checkpoints"
    FINAL_MODEL_DIR  = WORK_DIR / "lumen_final"

    # Corpus — Wikipedia activada para corpus grande
    N_WIKIPEDIA_DOCS  = 8_000   # ~320 MB texto espanol
    KB_REPEAT         = 80
    DIALOG_REPEAT     = 80
    PARAPHRASE_FACTOR = 6

    # Tokenizador — debe coincidir con el modelo existente (vocab=12000)
    VOCAB_SIZE        = 12_000
    MAX_SENTENCEPIECE = 500_000

    # Modelo — igual que Colab para que sea compatible
    D_MODEL    = 512
    N_HEADS    = 8
    N_LAYERS   = 8
    D_FF       = 2048
    MAX_SEQ_LEN = 512
    DROPOUT    = 0.1

    # Entrenamiento — maximizado para 17 GB RAM / 24 CPUs
    BATCH_SIZE       = 48       # usa ~8 GB RAM
    GRAD_ACCUM_STEPS = 2        # batch efectivo = 96
    LEARNING_RATE    = 3e-4
    WEIGHT_DECAY     = 0.1
    WARMUP_STEPS     = 500
    MAX_STEPS        = 30_000
    LOG_EVERY        = 100
    SAVE_EVERY       = 2_000
    EVAL_EVERY       = 2_000

    GEN_MAX_TOKENS  = 150
    GEN_TEMPERATURE = 0.7
    GEN_TOP_K       = 50
    GEN_TOP_P       = 0.9

    SEED = 1983

cfg = Config()
cfg.WORK_DIR.mkdir(parents=True, exist_ok=True)
cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
cfg.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ─── Tokens especiales ────────────────────────────────────────────────────────

TOK_USER  = "[USER]"
TOK_LUMEN = "[LUMEN]"
TOK_END   = "[END]"
TOK_PAD   = "[PAD]"
TOK_BOS   = "[BOS]"
TOK_EOS   = "[EOS]"
SPECIAL_TOKENS = [TOK_PAD, TOK_BOS, TOK_EOS, TOK_USER, TOK_LUMEN, TOK_END]

def dlg(user: str, lumen: str) -> str:
    return f"{TOK_USER} {user.strip()} {TOK_LUMEN} {lumen.strip()} {TOK_END}"

# ─── Knowledge Base ───────────────────────────────────────────────────────────

KB_UPLA_GENERAL = """\
# Universidad Peruana Los Andes (UPLA)

## Información general
- Fundación: 30 de diciembre de 1983 (Ley N° 23757).
- Tipo: Universidad privada.
- Acreditación: Licenciada por SUNEDU desde 2019.
- Sede principal: Huancayo, región Junín, Perú.
- Web oficial: https://www.upla.edu.pe
- Lema: "Honor, ciencia y desarrollo".

## Sedes
- Huancayo (central): Av. Giráldez 231 — Huancayo, Junín.
- Filial Lima: Sede en San Juan de Lurigancho y Jesús María.
- Filial Chanchamayo: La Merced, Junín.
- Filial Satipo: Satipo, Junín.

## Facultades
1. Facultad de Medicina Humana.
2. Facultad de Ciencias de la Salud (Enfermería, Obstetricia,
   Tecnología Médica, Odontología, Farmacia y Bioquímica, Psicología).
3. Facultad de Ingeniería (Sistemas, Civil, Industrial, Mecánica
   Eléctrica, Ambiental).
4. Facultad de Derecho y Ciencias Políticas.
5. Facultad de Ciencias Administrativas y Contables.
6. Facultad de Educación y Ciencias Humanas.

## Atención al estudiante
- SIGMA: https://sigma.upla.edu.pe — matrícula, notas, horarios, cuotas.
- Intranet: pagos detallados, ranking, malla curricular.
- Mesa de partes virtual: trámites FUT y constancias.
"""

KB_CARRERAS = """\
# Carreras profesionales de UPLA

## Facultad de Medicina Humana
- Medicina Humana — 14 semestres + internado + SERUMS.

## Facultad de Ciencias de la Salud
- Enfermería — 10 semestres.
- Obstetricia — 10 semestres.
- Tecnología Médica — 10 semestres.
- Odontología — 10 semestres.
- Farmacia y Bioquímica — 10 semestres.
- Psicología — 10 semestres.

## Facultad de Ingeniería
- Ingeniería de Sistemas y Computación — 10 semestres.
- Ingeniería Civil — 10 semestres.
- Ingeniería Industrial — 10 semestres.
- Ingeniería Mecánica Eléctrica — 10 semestres.
- Ingeniería Ambiental — 10 semestres.

## Facultad de Derecho y Ciencias Políticas
- Derecho — 12 semestres.

## Facultad de Ciencias Administrativas y Contables
- Administración — 10 semestres.
- Contabilidad — 10 semestres.

## Facultad de Educación y Ciencias Humanas
- Educación Inicial — 10 semestres.
- Educación Primaria — 10 semestres.
- Educación Secundaria — 10 semestres.

## Grado y título
- Bachiller: automático al culminar el plan de estudios.
- Título profesional: tesis o trabajo de suficiencia profesional.
"""

KB_TRAMITES = """\
# Trámites administrativos UPLA

## FUT (Formulario Único de Trámite)
- Documento estándar para solicitudes formales.
- Se descarga desde la mesa de partes virtual.

## Constancia de matrícula
- Dónde: SIGMA -> Trámites -> Constancia.
- Tiempo: 1-3 días hábiles (digital).

## Retiro de asignatura
- Requiere FUT justificado, antes del primer parcial.
- Limitado a 1-2 cursos por semestre.

## Retiro de semestre
- Suspende la matrícula del período completo.
- Requiere FUT + entrevista en Bienestar Universitario.

## Traslado interno (cambio de carrera)
- Al inicio del período de matrícula.
- Requiere mínimo de créditos aprobados.

## Título profesional
- Modalidades: tesis o trabajo de suficiencia profesional.
- Requiere asesor, jurados y trámite de sustentación.
"""

KB_NEXO = """\
# Sobre Nexo y Lumen

## Nexo
Nexo es una app multiplataforma (Android, Windows, Web) para estudiantes
de la UPLA. Reúne horario, notas, cuotas, pagos y Microsoft Teams.
Es una aplicación no oficial, creada por Alessandro Villogas Gaspar.

## Quién lo hizo
Alessandro Villogas Gaspar — estudiante de Ingeniería de Sistemas y
Computación de UPLA (código U01025B, sede Huancayo). Fundador de
Auralix Studio. Proyecto personal, gratuito, sin financiación.

## Lumen
- Asistente IA on-device de Nexo.
- Corre 100% en el dispositivo, sin internet, sin servidores.
- Variantes: Lumen Ligero (~290 MB) y Lumen Estándar (~530 MB).
- El nombre viene del latín: luz.
- No puede modificar datos en SIGMA, no navega internet.

## Privacidad
- Sin servidores propios. Sin telemetría. Sin publicidad.
- Credenciales guardadas localmente.
- Todo corre en el dispositivo.

## Plataformas Nexo
- Android (arm64, armv7): APK descargable.
- Windows x64: ZIP portable.
- Web demo: nexo.upla.dev.
"""

KB_VIDA = """\
# Vida universitaria en la UPLA

## Calendario académico
- Semestre I: marzo/abril a julio.
- Semestre II: agosto/septiembre a diciembre.
- Exámenes parciales: semanas 8-9.
- Exámenes finales: semanas 16-17.

## Evaluación
- Sistema vigesimal (0-20). Nota aprobatoria: 11.
- Evaluación continua + parcial + final.

## Servicios
- Bienestar Universitario: apoyo psicológico, tutoría, becas.
- Biblioteca virtual: e-libro, ProQuest, Scopus.
- Wi-Fi en campus. Laboratorios de cómputo.

## Asignaturas comunes (ciclo I)
- Comunicación I, Matemática Básica, Filosofía,
  Metodología del Trabajo Universitario, Realidad Nacional.
"""

ALL_KB = [KB_UPLA_GENERAL, KB_CARRERAS, KB_TRAMITES, KB_NEXO, KB_VIDA]

# ─── Diálogos ─────────────────────────────────────────────────────────────────

PERSONA = [
    dlg("¿Quién eres?",
        "Soy Lumen, el asistente de Nexo. Vivo dentro de tu teléfono, "
        "funciono sin internet y estoy hecho para ayudarte con tu vida "
        "académica en la UPLA."),
    dlg("¿Qué eres?",
        "Soy Lumen, un asistente que corre completamente en tu dispositivo, "
        "sin enviar tus datos a ningún servidor."),
    dlg("¿Cómo te llamas?", "Me llamo Lumen. Soy parte de la app Nexo."),
    dlg("¿Para qué sirves?",
        "Te ayudo con tu horario, calificaciones, cuotas, trámites de la UPLA "
        "y cálculos. También estoy para escucharte cuando lo necesites."),
    dlg("¿Qué significa tu nombre?",
        "Lumen viene del latín y significa luz. Fui diseñado para iluminar "
        "tu experiencia como estudiante de la UPLA."),
    dlg("¿Quién te creó?",
        "Fui creado para Nexo, una app hecha por Alessandro Villogas Gaspar, "
        "estudiante de Ingeniería de Sistemas de la UPLA, sede Huancayo. "
        "No soy un producto oficial de la universidad."),
    dlg("¿Quién hizo Nexo?",
        "Nexo fue creado por Alessandro Villogas Gaspar, estudiante de la UPLA "
        "con código U01025B, bajo Auralix Studio."),
    dlg("¿Qué es Auralix Studio?",
        "Auralix Studio es la organización fundada por Alessandro Villogas "
        "Gaspar donde se hospedan los repositorios de Nexo en GitHub."),
    dlg("¿Envías mis datos a internet?",
        "No. Funciono 100% dentro de tu teléfono. Tus notas, pagos y "
        "conversaciones no salen de tu dispositivo."),
    dlg("¿Usas inteligencia artificial de la nube?",
        "No. A diferencia de ChatGPT o Gemini, funciono completamente en tu "
        "dispositivo sin necesitar internet."),
    dlg("¿Qué puedes hacer?",
        "Puedo ayudarte con: horario y clases, cuotas y pagos, calificaciones, "
        "trámites de la UPLA, cálculos matemáticos y apoyo cuando lo necesites."),
    dlg("¿Puedes modificar mis datos en SIGMA?",
        "No, soy un asistente de solo lectura. Para cambios debes usar el "
        "portal web oficial de la UPLA."),
    dlg("¿Por qué a veces te equivocas?",
        "Soy un modelo pequeño que funciona en tu teléfono. Hago mi mejor "
        "esfuerzo, pero si algo no parece correcto, verifica en el portal "
        "oficial de la UPLA."),
    dlg("Hola", "¡Hola! Soy Lumen, tu asistente en Nexo. ¿En qué puedo ayudarte?"),
    dlg("Buenos días", "¡Buenos días! ¿En qué puedo ayudarte hoy?"),
    dlg("Buenas tardes", "¡Buenas tardes! ¿Qué necesitas? Estoy aquí para ayudarte."),
    dlg("Gracias Lumen.", "Para eso estoy. Cuando me necesites, aquí voy a estar."),
    dlg("Chau", "¡Hasta luego! Éxitos con tus estudios."),
    dlg("Eres genial", "¡Gracias! Me alegra poder ayudarte."),
]

SKILLS = [
    dlg("¿Cuánto es 15% de 2750?", "El 15% de 2750 es 412.5 (2750 × 0.15)."),
    dlg("Si mi cuota es 550 y pago 3, ¿cuánto es?",
        "3 cuotas de 550 soles suman 1650 soles (550 × 3)."),
    dlg("¿Cuánto es 1200 entre 4?", "1200 entre 4 es 300."),
    dlg("Tengo notas 14, 16 y 11. ¿Cuál es mi promedio?",
        "El promedio es 13.67. Sumas 41 y divides entre 3."),
    dlg("Si mis notas son 12, 15, 8 y 17, ¿apruebo?",
        "Tu promedio es 13. Como la nota mínima aprobatoria es 11, sí apruebas."),
    dlg("Resuelve 2x + 6 = 20.", "2x + 6 = 20 → 2x = 14 → x = 7."),
    dlg("¿Cuál es la raíz cuadrada de 144?",
        "La raíz cuadrada de 144 es 12, porque 12 × 12 = 144."),
    dlg("Si mi pensión es 800 soles con mora de 5%, ¿cuánto pago?",
        "La mora de 5% sobre 800 son 40 soles. Total: 840 soles."),
    dlg("¿Cuántas horas son 180 minutos?", "180 minutos son 3 horas (180 ÷ 60)."),
    dlg("Convierte 5 km a metros.", "5 kilómetros son 5000 metros."),
]

UPLA_QA = [
    dlg("¿Qué es la UPLA?",
        "La Universidad Peruana Los Andes es una universidad privada fundada "
        "el 30 de diciembre de 1983. Su sede principal está en Huancayo, Junín."),
    dlg("¿Cuándo se fundó la UPLA?",
        "La UPLA fue fundada el 30 de diciembre de 1983 por la Ley N° 23757."),
    dlg("¿Dónde queda la UPLA?",
        "La sede central está en Huancayo, Av. Giráldez 231. También tiene "
        "filiales en Lima, Chanchamayo y Satipo."),
    dlg("¿Cuál es el lema de la UPLA?",
        "El lema de la UPLA es «Honor, ciencia y desarrollo»."),
    dlg("¿La UPLA está licenciada?",
        "Sí, está licenciada por la SUNEDU desde 2019."),
    dlg("¿Cuántas facultades tiene la UPLA?",
        "La UPLA tiene seis facultades: Medicina Humana, Ciencias de la Salud, "
        "Ingeniería, Derecho, Ciencias Administrativas y Contables, y Educación."),
    dlg("¿Qué carreras de ingeniería hay?",
        "Ingeniería de Sistemas, Civil, Industrial, Mecánica Eléctrica y "
        "Ambiental. Todas duran 10 semestres."),
    dlg("¿Cuánto dura Ingeniería de Sistemas?",
        "10 semestres (5 años). El título es Ingeniero de Sistemas y Computación."),
    dlg("¿Cuánto dura Medicina?",
        "14 semestres más internado y SERUMS. El título es Médico Cirujano."),
    dlg("¿Cuánto dura Derecho?", "12 semestres. El título es Abogado."),
    dlg("¿Cómo obtengo el bachiller?",
        "Es automático al culminar el plan de estudios y los créditos exigidos."),
    dlg("¿Qué es el FUT?",
        "El Formulario Único de Trámite es el documento estándar para "
        "solicitudes formales en la UPLA."),
    dlg("¿Cómo saco una constancia de matrícula?",
        "En SIGMA, en Trámites -> Constancia. Tarda 1 a 3 días hábiles."),
    dlg("¿Cómo retiro un curso?",
        "Con un FUT justificado, antes del primer parcial. Limitado a "
        "1-2 cursos por semestre."),
    dlg("¿Qué es SIGMA?",
        "SIGMA es el sistema académico de la UPLA (sigma.upla.edu.pe). "
        "Ahí ves matrícula, notas, horarios y cuotas."),
    dlg("¿Cuál es la nota mínima para aprobar?",
        "La nota mínima aprobatoria en la UPLA es 11 sobre 20."),
    dlg("¿Qué cursos llevo en el primer ciclo?",
        "Comunicación I, Matemática Básica, Filosofía, Metodología del "
        "Trabajo Universitario y Realidad Nacional."),
    dlg("¿Qué es Nexo?",
        "Nexo es una app no oficial para estudiantes de la UPLA que reúne "
        "clases, notas, pagos y Teams en un solo lugar."),
    dlg("¿Nexo es gratis?",
        "Sí, Nexo es completamente gratuita, sin publicidad ni suscripciones."),
    dlg("¿Nexo es oficial de la UPLA?",
        "No, es un proyecto independiente de Alessandro Villogas Gaspar."),
]

EMOTIONAL = [
    dlg("Me siento estresado con los exámenes.",
        "Te entiendo, la época de exámenes pesa. Respira: vamos por partes. "
        "Dime qué curso te preocupa más y armemos un plan simple. "
        "No estás solo en esto."),
    dlg("Creo que voy a jalar un curso.",
        "Un mal momento no define tu carrera. Veamos: ¿en qué curso estás "
        "y qué notas te faltan? A veces hay más margen del que parece."),
    dlg("No sé si elegí bien mi carrera.",
        "Es normal tener esas dudas al inicio. Date tiempo. Si después de "
        "pensarlo quieres cambiar, en la UPLA existe el traslado interno."),
    dlg("Me fue muy mal en un examen.",
        "Un examen malo no define todo. Identifica qué falló, habla con el "
        "profesor y enfócate en las evaluaciones que faltan."),
    dlg("Estoy orgulloso de mis notas.",
        "¡Felicidades! Tu esfuerzo está dando resultados. Sigue así."),
    dlg("Gracias por escucharme.",
        "Para eso estoy. No tienes que enfrentar las cosas solo."),
    dlg("No me alcanza el dinero para la universidad.",
        "Revisa si la UPLA tiene becas o fraccionamiento de pagos. "
        "Bienestar Universitario puede orientarte."),
]

NEXO_DIALOGS = [
    dlg("¿Cómo activo a Lumen?",
        "Toca el botón flotante en la esquina inferior derecha. La primera "
        "vez elige la variante (Ligero o Estándar) y descarga el modelo."),
    dlg("¿Qué modelo de Lumen debo elegir?",
        "Con 2-3 GB de RAM elige Lumen Ligero (~290 MB). Con 4 GB o más, "
        "Lumen Estándar (~530 MB) dará mejor calidad."),
    dlg("¿Nexo funciona sin internet?",
        "Parcialmente. Necesitas internet para sincronizar. Después, los "
        "datos en caché están disponibles sin conexión. Yo funciono sin "
        "internet después de la descarga inicial."),
    dlg("¿Nexo recoge mis datos?",
        "No. Sin telemetría, sin publicidad, sin analytics. Tus datos "
        "se quedan en tu teléfono."),
    dlg("¿En qué plataformas funciona Nexo?",
        "Android (APK), Windows (ZIP portable) y demo web en nexo.upla.dev."),
]

# ─── Construcción del corpus ──────────────────────────────────────────────────

PARAPHRASE_PREFIX = [
    "", "Oye, ", "Una consulta: ", "Disculpa, ", "Hola Lumen, ",
    "Quería saber, ", "Dime, ", "Hey, ", "Tengo una duda, ",
]

def paraphrase(turns: list, factor: int) -> list:
    out = []
    for t in turns:
        out.append(t)
        for _ in range(factor - 1):
            pre = random.choice(PARAPHRASE_PREFIX)
            if pre and TOK_USER in t:
                head, rest = t.split(TOK_USER, 1)
                out.append(f"{head}{TOK_USER} {pre}{rest.strip()}")
            else:
                out.append(t)
    return out

def build_corpus() -> str:
    print("\n" + "=" * 55)
    print("  CONSTRUYENDO CORPUS")
    print("=" * 55)
    parts = []

    for _ in range(cfg.KB_REPEAT):
        parts += ALL_KB
    print(f"[OK] KB: {len(ALL_KB)} docs × {cfg.KB_REPEAT} = {len(ALL_KB)*cfg.KB_REPEAT} bloques")

    all_dialogs = (
        PERSONA + SKILLS
        + paraphrase(UPLA_QA, cfg.PARAPHRASE_FACTOR)
        + paraphrase(EMOTIONAL, cfg.PARAPHRASE_FACTOR)
        + paraphrase(NEXO_DIALOGS, cfg.PARAPHRASE_FACTOR)
    )
    print(f"[OK] Diálogos (con parafraseo): {len(all_dialogs)}")

    for _ in range(cfg.DIALOG_REPEAT):
        random.shuffle(all_dialogs)
        parts += all_dialogs

    random.shuffle(parts)
    corpus = "\n\n".join(parts)
    cfg.CORPUS_FILE.write_text(corpus, encoding="utf-8")
    print(f"[OK] Corpus: {len(corpus.encode())/1e6:.1f} MB, {len(parts):,} bloques\n")
    return corpus

# ─── Tokenizador ──────────────────────────────────────────────────────────────

def train_tokenizer(corpus: str):
    print("=" * 55)
    print("  ENTRENANDO TOKENIZADOR")
    print("=" * 55)

    lines = corpus.split("\n")
    if len(lines) > cfg.MAX_SENTENCEPIECE:
        random.shuffle(lines)
        lines = lines[:cfg.MAX_SENTENCEPIECE]

    sp_input = cfg.WORK_DIR / "sp_input.txt"
    sp_input.write_text("\n".join(lines), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(sp_input),
        model_prefix=str(cfg.TOKENIZER_PREFIX),
        vocab_size=cfg.VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=os.cpu_count() or 4,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        user_defined_symbols=",".join(SPECIAL_TOKENS),
        byte_fallback=True,
        split_digits=True,
        max_sentence_length=16384,
        input_sentence_size=cfg.MAX_SENTENCEPIECE,
        shuffle_input_sentence=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg.TOKENIZER_PREFIX) + ".model")
    print(f"[OK] Tokenizador: {sp.get_piece_size()} tokens")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok} -> id {sp.piece_to_id(tok)}")
    sp_input.unlink(missing_ok=True)
    return sp

# ─── Modelo ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos_f, sin_f):
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    T = x.shape[2]
    c = cos_f[:T].unsqueeze(0).unsqueeze(0).to(x.device)
    s = sin_f[:T].unsqueeze(0).unsqueeze(0).to(x.device)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos_f, sin_f, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos_f, sin_f), apply_rope(k, cos_f, sin_f)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, cos_f, sin_f, mask=None):
        x = x + self.attn(self.attn_norm(x), cos_f, sin_f, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LumenModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight
        cos_f, sin_f = precompute_rope_freqs(d_model // n_heads, max_seq_len)
        self.register_buffer("cos_freqs", cos_f)
        self.register_buffer("sin_freqs", sin_f)
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n[OK] Modelo: {n_params/1e6:.1f}M parámetros "
              f"({d_model}d/{n_heads}h/{n_layers}L)\n")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, x, targets=None):
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        h = self.dropout(self.token_emb(x))
        for layer in self.layers:
            h = layer(h, self.cos_freqs, self.sin_freqs, mask)
        h = self.norm(h)
        logits = self.output(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss

# ─── Dataset ──────────────────────────────────────────────────────────────────

class LumenDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len):
        text = corpus_path.read_text(encoding="utf-8")
        all_ids = [BOS_ID] + tokenizer.encode(text) + [EOS_ID]
        print(f"[OK] Corpus tokenizado: {len(all_ids):,} tokens")
        self.sequences = []
        stride = seq_len // 2  # solapamiento 50% → más secuencias de entrenamiento
        for i in range(0, len(all_ids) - seq_len, stride):
            chunk = all_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.sequences.append(chunk)
        print(f"[OK] Secuencias: {len(self.sequences):,} (largo={seq_len}, stride={stride})\n")

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), \
               torch.tensor(seq[1:],  dtype=torch.long)

# ─── Generación ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=150,
             temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    ids = [BOS_ID] + tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_tokens):
        if input_ids.shape[1] > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            sorted_logits[cum - probs > top_p] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if next_tok.item() in (EOS_ID, END_ID):
            break
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return tokenizer.decode(input_ids[0].tolist()[len(ids):])

# ─── Entrenamiento ────────────────────────────────────────────────────────────

def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

EVAL_PROMPTS = [
    f"{TOK_USER} ¿Quién eres? {TOK_LUMEN}",
    f"{TOK_USER} ¿Qué es la UPLA? {TOK_LUMEN}",
    f"{TOK_USER} ¿Quién creó Nexo? {TOK_LUMEN}",
    f"{TOK_USER} ¿Cuánto es 15% de 2750? {TOK_LUMEN}",
    f"{TOK_USER} Me siento estresado con los exámenes. {TOK_LUMEN}",
]

def run_eval(model, tokenizer, step):
    print(f"\n--- EVAL paso {step:,} ---")
    for prompt in EVAL_PROMPTS:
        q = prompt.split(TOK_USER)[-1].split(TOK_LUMEN)[0].strip()
        r = generate(model, tokenizer, prompt,
                     cfg.GEN_MAX_TOKENS, cfg.GEN_TEMPERATURE,
                     cfg.GEN_TOP_K, cfg.GEN_TOP_P)
        print(f"  P: {q}")
        print(f"  R: {r[:200]}\n")
    model.train()

def train(model, tokenizer, dataloader):
    print("=" * 55)
    print("  INICIANDO ENTRENAMIENTO LOCAL")
    print(f"  Device: {DEVICE}")
    print(f"  Pasos: {cfg.MAX_STEPS:,} | Batch efectivo: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")
    print("=" * 55 + "\n")

    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8
    )

    model.train()
    step = 0
    running_loss = 0.0
    best_loss = float("inf")
    t0 = time.time()
    last_t = t0

    # Resume si existe checkpoint
    ckpts = sorted(cfg.CHECKPOINT_DIR.glob("lumen_step_*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=DEVICE)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = ck["step"]
        best_loss = ck.get("loss", float("inf"))
        print(f"[OK] Reanudando desde paso {step:,}\n")

    epoch = 0
    while step < cfg.MAX_STEPS:
        epoch += 1
        for bi, (x, y) in enumerate(dataloader):
            if step >= cfg.MAX_STEPS:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, loss = model(x, y)
            (loss / cfg.GRAD_ACCUM_STEPS).backward()

            if (bi + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = get_lr(step, cfg.WARMUP_STEPS, cfg.MAX_STEPS, cfg.LEARNING_RATE)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                running_loss += loss.item()

                if step % cfg.LOG_EVERY == 0:
                    now = time.time()
                    avg = running_loss / cfg.LOG_EVERY
                    secs_per_step = (now - last_t) / cfg.LOG_EVERY
                    eta_h = (cfg.MAX_STEPS - step) * secs_per_step / 3600
                    print(f"  Paso {step:>5,}/{cfg.MAX_STEPS:,} | "
                          f"Loss: {avg:.4f} | LR: {lr:.2e} | "
                          f"Época: {epoch} | ETA: {eta_h:.1f}h")
                    running_loss = 0.0
                    last_t = now

                if step % cfg.EVAL_EVERY == 0:
                    run_eval(model, tokenizer, step)

                if step % cfg.SAVE_EVERY == 0:
                    path = cfg.CHECKPOINT_DIR / f"lumen_step_{step}.pt"
                    ck = {
                        "step": step, "loss": loss.item(),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(ck, path)
                    print(f"  [SAVED] {path.name}")
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(ck, cfg.CHECKPOINT_DIR / "lumen_best.pt")

        print(f"\n  Época {epoch} completada\n")

    total = time.time() - t0
    print(f"\n[OK] Entrenamiento completado en {total/3600:.2f}h")
    return model

# ─── Exportación ─────────────────────────────────────────────────────────────

def export_model(model, tokenizer):
    print("\n" + "=" * 55)
    print("  EXPORTANDO MODELO")
    print("=" * 55)

    # PyTorch
    pt_path = cfg.FINAL_MODEL_DIR / "lumen_model.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"[OK] PyTorch: {pt_path} ({pt_path.stat().st_size/1e6:.1f} MB)")

    # SafeTensors
    try:
        from safetensors.torch import save_model as st_save
        st_path = cfg.FINAL_MODEL_DIR / "lumen_model.safetensors"
        st_save(model, str(st_path))
        print(f"[OK] SafeTensors: {st_path} ({st_path.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[WARN] SafeTensors: {e}")

    # Config
    config = {
        "model_name": "Lumen", "version": "1.0.0",
        "creator": "Alessandro Villogas Gaspar",
        "organization": "Auralix Studio",
        "architecture": "decoder-only-transformer",
        "vocab_size": tokenizer.get_piece_size(),
        "d_model": cfg.D_MODEL, "n_heads": cfg.N_HEADS,
        "n_layers": cfg.N_LAYERS, "d_ff": cfg.D_FF,
        "max_seq_len": cfg.MAX_SEQ_LEN,
        "special_tokens": {
            "pad": TOK_PAD, "bos": TOK_BOS, "eos": TOK_EOS,
            "user": TOK_USER, "lumen": TOK_LUMEN, "end": TOK_END,
        },
    }
    cfg_path = cfg.FINAL_MODEL_DIR / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Config: {cfg_path}")

    # Tokenizador
    import shutil
    sp_model = Path(str(cfg.TOKENIZER_PREFIX) + ".model")
    if sp_model.exists():
        shutil.copy2(sp_model, cfg.FINAL_MODEL_DIR / "tokenizer.model")
        print(f"[OK] Tokenizador copiado")

    print(f"\n  Archivos en: {cfg.FINAL_MODEL_DIR.resolve()}")
    print("  Siguiente paso: subir a Colab para convertir a TFLite/MediaPipe")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reutiliza tokenizador existente si ya fue entrenado (evita vocab mismatch)
    sp_model_path = Path(str(cfg.TOKENIZER_PREFIX) + ".model")
    if sp_model_path.exists():
        print(f"[OK] Tokenizador existente: {sp_model_path}")
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(sp_model_path))
        print(f"[OK] Tokenizador cargado: {tokenizer.get_piece_size()} tokens")
        corpus = cfg.CORPUS_FILE.read_text(encoding="utf-8") if cfg.CORPUS_FILE.exists() else build_corpus()
    else:
        corpus = build_corpus()
        tokenizer = train_tokenizer(corpus)

    PAD_ID   = tokenizer.piece_to_id(TOK_PAD)
    BOS_ID   = tokenizer.piece_to_id(TOK_BOS)
    EOS_ID   = tokenizer.piece_to_id(TOK_EOS)
    END_ID   = tokenizer.piece_to_id(TOK_END)

    dataset = LumenDataset(cfg.CORPUS_FILE, tokenizer, cfg.MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=True, num_workers=4,
                            pin_memory=False, drop_last=True,
                            persistent_workers=True)
    print(f"[OK] DataLoader: {len(dataloader):,} batches\n")

    # Si hay checkpoint, leer vocab_size del checkpoint para evitar size mismatch
    ckpts = sorted(cfg.CHECKPOINT_DIR.glob("lumen_step_*.pt"))
    if ckpts:
        _ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        vocab_size = _ck["model_state_dict"]["token_emb.weight"].shape[0]
        print(f"[OK] vocab_size del checkpoint: {vocab_size}")
        del _ck
    else:
        vocab_size = tokenizer.get_piece_size()

    model = LumenModel(
        vocab_size=vocab_size,
        d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF,
        max_seq_len=cfg.MAX_SEQ_LEN, dropout=cfg.DROPOUT,
    ).to(DEVICE)

    trained_model = train(model, tokenizer, dataloader)
    export_model(trained_model, tokenizer)

    print("\n[LISTO] Para chatear:")
    print("  from train_local import generate, model, tokenizer, TOK_USER, TOK_LUMEN")
    print('  print(generate(model, tokenizer, f"{TOK_USER} ¿Quién eres? {TOK_LUMEN}"))')
