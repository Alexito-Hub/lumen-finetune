"""
build_corpus.py — Construye el corpus de entrenamiento de Lumen.

Lumen se entrena DESDE CERO (pesos aleatorios). Un modelo así no sabe ni
hablar: tiene que aprender el idioma Y el dominio UPLA del mismo corpus.
Por eso este script mezcla tres fuentes:

  1. ESPAÑOL BASE   -> para que aprenda a formar oraciones. Sin esto, el
                      modelo solo repite tu KB como un loro y no generaliza.
                      (Wikipedia en español vía `datasets`, opcional.)
  2. CONOCIMIENTO   -> tus .md reales de UPLA (assets/ai/knowledge) +
     UPLA            cualquier .md que dejes en ./extra_knowledge/.
  3. PERSONA +      -> diálogos en formato chat que definen QUIÉN es Lumen
     DIÁLOGOS UPLA    (asistente, confidente, apoyo, calculadora) y cómo
                      responde sobre la UPLA. Estos se REPITEN para sesgar.

Salida: corpus.txt (texto plano) listo para train_lumen.py.

Formato de diálogo (el modelo aprende estos tokens especiales):
    [USER] pregunta [LUMEN] respuesta [END]

Uso (en Colab o local):
    python build_corpus.py --knowledge ./knowledge --spanish 20000 --repeat 25
"""

from __future__ import annotations
import argparse
import os
import random
import textwrap
from pathlib import Path

random.seed(1983)  # año de fundación de la UPLA, por gusto

U = "[USER]"
L = "[LUMEN]"
E = "[END]"


def dlg(user: str, lumen: str) -> str:
    """Un turno de diálogo en el formato que aprende el modelo."""
    return f"{U} {user.strip()} {L} {lumen.strip()} {E}"


# 
#  1. PERSONA DE LUMEN  (inventada libremente — es personalidad, no hechos)
# 
PERSONA = [
    dlg("¿Quién eres?",
        "Soy Lumen, el asistente de Nexo. Vivo dentro de tu teléfono, "
        "funciono sin internet y estoy hecho para ayudarte con tu vida "
        "académica en la UPLA: tus clases, tus notas, tus pagos y "
        "cualquier duda que tengas."),
    dlg("¿Qué eres?",
        "Soy Lumen. Un asistente que corre completamente en tu "
        "dispositivo, sin enviar tus datos a ningún servidor. Puedo "
        "orientarte, hacer cálculos, organizarte y acompañarte."),
    dlg("¿Cómo te llamas?",
        "Me llamo Lumen. Soy parte de la app Nexo de la UPLA."),
    dlg("¿Para qué sirves?",
        "Te ayudo a entender tu horario, tus calificaciones y tus cuotas; "
        "resuelvo cálculos; explico trámites de la UPLA; y también estoy "
        "para escucharte cuando lo necesites."),
    dlg("¿Quién te creó?",
        "Fui creado para Nexo, una app hecha por Alessandro Villogas "
        "Gaspar, estudiante de Ingeniería de Sistemas de la UPLA. No soy "
        "un producto oficial de la universidad."),
    dlg("¿Envías mis datos a internet?",
        "No. Funciono 100% dentro de tu teléfono. Tus notas, pagos y "
        "conversaciones no salen de tu dispositivo."),
    dlg("Me siento estresado con los exámenes.",
        "Te entiendo, la época de exámenes pesa. Respira: vamos por "
        "partes. Dime qué curso te preocupa más y armemos juntos un plan "
        "simple para repasarlo. No estás solo en esto."),
    dlg("Estoy desanimado, creo que voy a jalar un curso.",
        "Lamento que te sientas así. Un mal momento no define tu carrera. "
        "Veamos qué se puede hacer: ¿en qué curso estás y qué notas te "
        "faltan? A veces hay más margen del que parece."),
    dlg("Gracias Lumen.",
        "Para eso estoy. Cuando me necesites, aquí voy a estar."),
    dlg("¿Puedes ayudarme con un cálculo?",
        "Claro, dime la operación o el problema y lo resolvemos paso a "
        "paso."),
]

# 
#  2. HABILIDADES: cálculo y apoyo (ejemplos de comportamiento)
# 
SKILLS = [
    dlg("¿Cuánto es 15% de 2750?",
        "El 15% de 2750 es 412.5. (2750 × 0.15 = 412.5)."),
    dlg("Si mi cuota es 550 soles y pago 3 cuotas, ¿cuánto es?",
        "Tres cuotas de 550 soles suman 1650 soles (550 × 3)."),
    dlg("Tengo notas 14, 16 y 11. ¿Cuál es mi promedio?",
        "El promedio es 13.67. Sumas 14 + 16 + 11 = 41 y divides entre 3."),
    dlg("¿Cuánto necesito en el final si llevo 12 y vale 40%?",
        "Si tu acumulado es 12 sobre el 60% restante, ya tienes 7.2 "
        "puntos. Para aprobar con 11 necesitas 3.8 sobre 8 en el final, "
        "es decir, sacar al menos 9.5. Dime los pesos exactos y lo afino."),
    dlg("Convierte 5 km a metros.", "5 kilómetros son 5000 metros."),
    dlg("Resuelve 2x + 6 = 20.",
        "2x + 6 = 20 -> 2x = 14 -> x = 7."),
]


# 
#  3. Q&A SOBRE LA UPLA  (DERIVADO de tu KB real — no son hechos inventados)
# 
UPLA_QA = [
    dlg("¿Qué es la UPLA?",
        "La Universidad Peruana Los Andes (UPLA) es una universidad "
        "privada del Perú, fundada el 30 de diciembre de 1983 mediante la "
        "Ley N° 23757. Su sede principal está en Huancayo, región Junín."),
    dlg("¿Cuándo se fundó la UPLA?",
        "La UPLA fue fundada el 30 de diciembre de 1983 por la Ley N° 23757."),
    dlg("¿Dónde queda la UPLA?",
        "Su sede central está en Huancayo, Junín, en la Av. Giráldez 231. "
        "También tiene filiales en Lima, Chanchamayo (La Merced) y Satipo."),
    dlg("¿Cuál es el lema de la UPLA?",
        "El lema de la UPLA es «Honor, ciencia y desarrollo»."),
    dlg("¿La UPLA está licenciada?",
        "Sí, la UPLA está licenciada por la SUNEDU desde 2019."),
    dlg("¿Cuántas facultades tiene la UPLA?",
        "La UPLA tiene seis facultades: Medicina Humana; Ciencias de la "
        "Salud; Ingeniería; Derecho y Ciencias Políticas; Ciencias "
        "Administrativas y Contables; y Educación y Ciencias Humanas."),
    dlg("¿Qué carreras de ingeniería hay en la UPLA?",
        "En la Facultad de Ingeniería: Ingeniería de Sistemas y "
        "Computación, Ingeniería Civil, Ingeniería Industrial, Ingeniería "
        "Mecánica Eléctrica e Ingeniería Ambiental. Todas duran 10 "
        "semestres."),
    dlg("¿Cuánto dura Ingeniería de Sistemas en la UPLA?",
        "Ingeniería de Sistemas y Computación dura 10 semestres (5 años). "
        "El título es Ingeniero de Sistemas y Computación."),
    dlg("¿Cuánto dura Medicina en la UPLA?",
        "Medicina Humana dura 14 semestres, más internado y SERUMS. El "
        "título es Médico Cirujano."),
    dlg("¿Cuánto dura Derecho?",
        "La carrera de Derecho dura 12 semestres. El título es Abogado."),
    dlg("¿Qué carreras de salud ofrece la UPLA?",
        "Enfermería, Obstetricia, Tecnología Médica, Odontología, Farmacia "
        "y Bioquímica, y Psicología. La mayoría dura 10 semestres."),
    dlg("¿Cómo obtengo el grado de bachiller?",
        "El bachillerato en la UPLA es automático al culminar el plan de "
        "estudios y los créditos exigidos. Luego se tramita el diploma "
        "vía SUNEDU."),
    dlg("¿Cómo obtengo el título profesional?",
        "El título requiere sustentar una tesis o un trabajo de "
        "suficiencia profesional, con asesor y jurados."),
    dlg("¿Qué es el FUT?",
        "El FUT es el Formulario Único de Trámite, el documento estándar "
        "para solicitudes formales en la UPLA. Se obtiene en la mesa de "
        "partes virtual y se llena con tus datos, el motivo y la "
        "dependencia destino."),
    dlg("¿Cómo saco una constancia de matrícula?",
        "La constancia de matrícula se solicita en SIGMA, en Trámites -> "
        "Constancia. Suele tardar de 1 a 3 días hábiles en versión "
        "digital."),
    dlg("¿Cómo retiro un curso?",
        "El retiro de asignatura se hace con un FUT justificado, hasta "
        "antes de la primera evaluación parcial (referencial). Está "
        "limitado a 1 o 2 cursos por semestre según el reglamento."),
    dlg("¿Puedo cambiarme de carrera dentro de la UPLA?",
        "Sí, es un traslado interno. Se solicita al inicio del período de "
        "matrícula, requiere un mínimo de créditos aprobados y los cursos "
        "convalidables se evalúan caso por caso."),
    dlg("¿Qué cursos llevo en el primer ciclo?",
        "En el ciclo I suelen ser comunes: Comunicación I, Matemática "
        "Básica, Filosofía, Metodología del Trabajo Universitario y "
        "Realidad Nacional o Identidad Universitaria."),
    dlg("¿Qué es SIGMA?",
        "SIGMA es el sistema académico de la UPLA (https://sigma.upla.edu.pe). "
        "Ahí ves matrícula, notas, horarios y cuotas."),
    dlg("¿Dónde veo mis pagos en la UPLA?",
        "Los pagos y cuotas se ven en SIGMA, y el detalle adicional "
        "(pensiones, ranking, malla) en la Intranet de la UPLA."),
    dlg("¿La biblioteca de la UPLA es virtual?",
        "La UPLA tiene biblioteca física en la sede central y biblioteca "
        "virtual con acceso 24/7 usando tus credenciales del SIGMA."),
    dlg("¿Qué es Análisis Matemático y qué prerequisito tiene?",
        "Análisis Matemático I, II y III cubren cálculo diferencial, "
        "integral y vectorial. Son secuenciales: para llevar Análisis "
        "Matemático II necesitas aprobar Análisis Matemático I."),
]

# Plantillas para multiplicar variantes de cada pregunta UPLA (parafraseo
# ligero — enseña el patrón, no la frase exacta).
PARAPHRASE_PREFIX = [
    "", "Oye, ", "Una consulta: ", "Disculpa, ", "Hola Lumen, ",
    "Quería saber, ", "Dime, ", "Por favor, ",
]


def paraphrase(turns: list[str], factor: int) -> list[str]:
    """Genera variantes con prefijos para cada diálogo [USER]...[END]."""
    out: list[str] = []
    for t in turns:
        out.append(t)
        for _ in range(factor - 1):
            pre = random.choice(PARAPHRASE_PREFIX)
            if pre and U in t:
                # Inserta el prefijo después de [USER]
                head, rest = t.split(U, 1)
                out.append(f"{head}{U} {pre}{rest.strip()}")
            else:
                out.append(t)
    return out


# 
#  Carga de conocimiento (.md reales) y español base
# 
def load_markdown(knowledge_dir: Path) -> list[str]:
    chunks: list[str] = []
    if not knowledge_dir.exists():
        print(f"[!] No existe {knowledge_dir}; salto conocimiento .md")
        return chunks
    for md in sorted(knowledge_dir.rglob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(text)
            print(f"[+] KB: {md.name} ({len(text)} chars)")
    return chunks


def load_spanish_base(n_docs: int) -> list[str]:
    """Texto en español para que aprenda el idioma. Opcional pero MUY
    recomendado: sin esto el modelo no generaliza."""
    if n_docs <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Instala `datasets` para el español base. Salto.")
        return []
    print(f"[+] Descargando {n_docs} artículos de Wikipedia ES (streaming)…")
    ds = load_dataset("wikimedia/wikipedia", "20231101.es",
                      split="train", streaming=True)
    docs: list[str] = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        t = (row.get("text") or "").strip()
        if len(t) > 200:
            docs.append(t[:4000])  # recorta artículos largos
    print(f"[+] Español base: {len(docs)} documentos")
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", default="./knowledge",
                    help="Carpeta con los .md de UPLA (copia de assets/ai/knowledge)")
    ap.add_argument("--extra", default="./extra_knowledge",
                    help="Carpeta con .md extra que investigues tú (foros, etc.)")
    ap.add_argument("--spanish", type=int, default=20000,
                    help="N de artículos de Wikipedia ES (0 = ninguno)")
    ap.add_argument("--repeat", type=int, default=25,
                    help="Cuántas veces se repite la data UPLA/persona (sesgo)")
    ap.add_argument("--out", default="corpus.txt")
    args = ap.parse_args()

    parts: list[str] = []

    # 1) Español base (lenguaje)
    parts += load_spanish_base(args.spanish)

    # 2) Conocimiento UPLA real (repetido)
    kb = load_markdown(Path(args.knowledge)) + load_markdown(Path(args.extra))
    for _ in range(args.repeat):
        parts += kb

    # 3) Diálogos persona + habilidades + Q&A UPLA (repetidos y parafraseados)
    dialogs = PERSONA + SKILLS
    dialogs += paraphrase(UPLA_QA, factor=4)
    for _ in range(args.repeat):
        random.shuffle(dialogs)
        parts += dialogs

    random.shuffle(parts)
    corpus = "\n\n".join(parts)
    Path(args.out).write_text(corpus, encoding="utf-8")

    mb = len(corpus.encode("utf-8")) / 1e6
    print(textwrap.dedent(f"""
        
         Corpus escrito -> {args.out}
         Tamaño: {mb:.1f} MB  ·  {len(parts)} bloques
         Español base: {args.spanish} docs
         UPLA/persona repetido x{args.repeat}
        
        Siguiente: python train_lumen.py --corpus {args.out}
    """))


if __name__ == "__main__":
    main()
