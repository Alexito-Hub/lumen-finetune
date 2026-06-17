"""
              LUMEN - Entrenamiento v2.0 (Google Colab)

  Modelo:  Lumen - asistente IA on-device para Nexo / UPLA
  Autor:   Alessandro Villogas Gaspar - Auralix Studio
  Fecha:   Junio 2026

  Mejoras v2.0
  1. TOKENS DE FUNCION  [CALL:X] / [RESULT]
     El modelo aprende cuando pedir datos del estudiante.
     El runtime de Nexo intercepta [CALL:X] e inyecta el
     resultado real. El modelo completa la respuesta.

  2. MODO THINKING  [THINK] ... [/THINK]
     Para calculos complejos el modelo emite un razonamiento
     breve antes de la respuesta. Mejora aritmetica y la
     deteccion de intencion.

  3. LOSS SOLO SOBRE RESPUESTAS DE LUMEN (response mask)
     No se propagan gradientes sobre la pregunta del usuario.
     El modelo aprende a responder, no a memorizar preguntas.

  4. LABEL SMOOTHING (0.1)
     Regulariza el vocabulario pequeno. Mejora generalizacion.

  5. ESPANOL + UPLA PRIMERO
     Wikipedia reducida a 3000 docs.
     KB UPLA/Nexo/Lumen repetida 40 veces.

  6. CORPUS EXPANDIDO
     +40 dialogos de function-call
     +30 dialogos con thinking
     +25 dialogos de clarificacion
"""

import subprocess, sys

def install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("sentencepiece")
install("datasets")
install("safetensors")

import os, math, json, time, random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import sentencepiece as spm

print(f"PyTorch {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class Config:
    """Hiperparametros y directorios centralizados."""

    WORK_DIR         = Path("./lumen_training")
    CORPUS_FILE      = WORK_DIR / "corpus.txt"
    TOKENIZER_PREFIX = WORK_DIR / "lumen_tokenizer"
    CHECKPOINT_DIR   = WORK_DIR / "checkpoints"
    FINAL_MODEL_DIR  = WORK_DIR / "lumen_final"

    GDRIVE_MOUNT     = Path("/content/drive")
    GDRIVE_SAVE_DIR  = GDRIVE_MOUNT / "MyDrive" / "lumen_models"

    # Corpus - UPLA tiene prioridad absoluta sobre Wikipedia
    N_WIKIPEDIA_DOCS  = 3_000
    KB_REPEAT         = 40
    DIALOG_REPEAT     = 40
    PARAPHRASE_FACTOR = 5

    # Tokenizador
    VOCAB_SIZE        = 12_000
    MAX_SENTENCEPIECE = 300_000

    # Arquitectura
    D_MODEL     = 512
    N_HEADS     = 8
    N_LAYERS    = 8
    D_FF        = 2048
    MAX_SEQ_LEN = 512
    DROPOUT     = 0.1

    # Entrenamiento
    BATCH_SIZE        = 16
    GRAD_ACCUM_STEPS  = 4
    LEARNING_RATE     = 3e-4
    WEIGHT_DECAY      = 0.1
    WARMUP_STEPS      = 3_000
    MAX_STEPS         = 60_000
    LABEL_SMOOTHING   = 0.1
    GRAD_CLIP         = 1.0

    LOG_EVERY  = 100
    SAVE_EVERY = 2_000
    EVAL_EVERY = 2_000

    # Generacion
    GEN_MAX_TOKENS  = 256
    GEN_TEMPERATURE = 0.7
    GEN_TOP_K       = 50
    GEN_TOP_P       = 0.9

    SEED = 1983

cfg = Config()

for d in [cfg.WORK_DIR, cfg.CHECKPOINT_DIR, cfg.FINAL_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {DEVICE}")
print(f"Modelo: {cfg.D_MODEL}d / {cfg.N_HEADS}h / {cfg.N_LAYERS}L")
print(f"Vocab: {cfg.VOCAB_SIZE} | Seq: {cfg.MAX_SEQ_LEN}")
print(f"Wikipedia: {cfg.N_WIKIPEDIA_DOCS} docs | KB x{cfg.KB_REPEAT} | Dialogs x{cfg.DIALOG_REPEAT}")

try:
    from google.colab import drive
    drive.mount(str(cfg.GDRIVE_MOUNT))
    cfg.GDRIVE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    USING_GDRIVE = True
    print(f"[OK] Google Drive montado -> {cfg.GDRIVE_SAVE_DIR}")
except Exception:
    USING_GDRIVE = False
    print("[WARN] Google Drive no disponible. Solo guardado local.")


# ═══════════════════════════════════════════════════════════
#   TOKENS ESPECIALES
# ═══════════════════════════════════════════════════════════

TOK_PAD      = "[PAD]"
TOK_BOS      = "[BOS]"
TOK_EOS      = "[EOS]"
TOK_USER     = "[USER]"
TOK_LUMEN    = "[LUMEN]"
TOK_END      = "[END]"
TOK_THINK    = "[THINK]"
TOK_ENDTHINK = "[/THINK]"
TOK_CALL     = "[CALL]"
TOK_RESULT   = "[RESULT]"

SPECIAL_TOKENS = [
    TOK_PAD, TOK_BOS, TOK_EOS,
    TOK_USER, TOK_LUMEN, TOK_END,
    TOK_THINK, TOK_ENDTHINK,
    TOK_CALL, TOK_RESULT,
]


def dlg(user: str, lumen: str) -> str:
    """Turno de dialogo simple."""
    return f"{TOK_USER} {user.strip()} {TOK_LUMEN} {lumen.strip()} {TOK_END}"


def dlg_think(user: str, think: str, response: str) -> str:
    """Turno con paso de razonamiento explicito."""
    return (
        f"{TOK_USER} {user.strip()} {TOK_LUMEN} "
        f"{TOK_THINK} {think.strip()} {TOK_ENDTHINK} "
        f"{response.strip()} {TOK_END}"
    )


def dlg_call(user: str, func: str, result: str, response: str,
             think: str = "") -> str:
    """
    Turno con function call.
    El modelo emite [CALL]:func, el runtime inyecta [RESULT]...[RESULT],
    y el modelo completa la respuesta.
    """
    think_part = (
        f"{TOK_THINK} {think.strip()} {TOK_ENDTHINK} " if think else ""
    )
    return (
        f"{TOK_USER} {user.strip()} {TOK_LUMEN} "
        f"{think_part}"
        f"{TOK_CALL}:{func} "
        f"{TOK_RESULT} {result.strip()} {TOK_RESULT} "
        f"{response.strip()} {TOK_END}"
    )


# ═══════════════════════════════════════════════════════════
#   BASE DE CONOCIMIENTO
# ═══════════════════════════════════════════════════════════

KB_UPLA_GENERAL = """\
# Universidad Peruana Los Andes (UPLA)

## Informacion general
- Fundacion: 30 de diciembre de 1983 (Ley N 23757).
- Tipo: Universidad privada.
- Acreditacion: Licenciada por SUNEDU desde 2019.
- Sede principal: Huancayo, region Junin, Peru.
- Web oficial: https://www.upla.edu.pe
- Lema: Honor, ciencia y desarrollo.

## Sedes
- Huancayo (central): Av. Giraldez 231.
- Filial Lima: San Juan de Lurigancho y Jesus Maria.
- Filial Chanchamayo: La Merced, Junin.
- Filial Satipo: Satipo, Junin.

## Facultades
1. Medicina Humana.
2. Ciencias de la Salud (Enfermeria, Obstetricia, Tecnologia Medica,
   Odontologia, Farmacia y Bioquimica, Psicologia).
3. Ingenieria (Sistemas, Civil, Industrial, Mecanica Electrica, Ambiental).
4. Derecho y Ciencias Politicas.
5. Ciencias Administrativas y Contables.
6. Educacion y Ciencias Humanas.

## Biblioteca
- Fisica (sede central) + virtual 24/7 con credenciales SIGMA.
- Horario: lun-vie 8:00-20:00, sab 8:00-13:00.
- Bases de datos: e-libro, ProQuest, Scopus.

## Sistemas digitales
- SIGMA: https://sigma.upla.edu.pe
- Intranet: pensiones detalladas, ranking, malla curricular.
- Mesa de partes virtual: FUT y constancias.
"""

KB_CARRERAS = """\
# Carreras profesionales UPLA

## Medicina Humana
- Medicina Humana: 14 semestres + internado + SERUMS. Titulo: Medico Cirujano.

## Ciencias de la Salud
- Enfermeria: 10 semestres.
- Obstetricia: 10 semestres.
- Tecnologia Medica: 10 semestres.
- Odontologia: 10 semestres. Titulo: Cirujano Dentista.
- Farmacia y Bioquimica: 10 semestres.
- Psicologia: 10 semestres.

## Ingenieria
- Ingenieria de Sistemas y Computacion: 10 semestres.
- Ingenieria Civil: 10 semestres.
- Ingenieria Industrial: 10 semestres.
- Ingenieria Mecanica Electrica: 10 semestres.
- Ingenieria Ambiental: 10 semestres.

## Derecho
- Derecho: 12 semestres. Titulo: Abogado/a.

## Ciencias Administrativas
- Administracion: 10 semestres.
- Contabilidad: 10 semestres. Titulo: Contador Publico.

## Educacion
- Educacion Inicial, Primaria, Secundaria: 10 semestres.

## Grado y titulo
- Bachiller: automatico al completar el plan de estudios.
- Titulo: tesis o suficiencia profesional.
"""

KB_ASIGNATURAS = """\
# Asignaturas por facultad (referencial)

## Ciclo I - todas las facultades
- Comunicacion I, Matematica Basica, Filosofia,
  Metodologia del Trabajo Universitario, Realidad Nacional.

## Ingenieria
- Analisis Matematico I-II-III (prerequisitos secuenciales).
- Algebra Lineal, Fisica I-II-III, Estadistica y Probabilidades.
- Algoritmos y Programacion I-II.

## Salud
- Anatomia Humana I y II, Bioquimica, Biologia Celular, Histologia.

## Derecho
- Introduccion al Derecho, Derecho Romano, Teoria General del Derecho.

## Administracion
- Contabilidad General I y II, Microeconomia, Matematica Financiera.

## Prerequisitos clave
- Analisis Matematico II requiere Analisis Matematico I.
- Algoritmos II requiere Algoritmos I.
- Fisica II requiere Analisis Matematico I + Fisica I.

## Carga academica tipica
- Ciclos 1-3: 22-26 creditos (5-7 cursos).
- Ciclos 4-7: 18-22 creditos.
- Ciclos 8-10: 16-18 creditos + practicas.
"""

KB_TRAMITES = """\
# Tramites administrativos UPLA

## FUT (Formulario Unico de Tramite)
Documento estandar para solicitudes. Mesa de partes virtual.

## Constancia de matricula
SIGMA -> Tramites -> Constancia. 1-3 dias habiles.

## Constancia de notas
FUT o via SIGMA. Para becas, intercambios, practicas.

## Retiro de asignatura
FUT justificado, antes de la primera evaluacion parcial.
Limite: 1-2 cursos por semestre.

## Retiro de semestre
FUT + entrevista en Bienestar Universitario. Sin devolucion total.

## Traslado interno
Inicio del periodo. Minimo de creditos. Convalidacion caso por caso.

## Bachiller
Automatico al completar el plan de estudios. Diploma via SUNEDU.

## Titulo profesional
Tesis (con asesor y jurados) o suficiencia profesional.
"""

KB_NEXO = """\
# Nexo - Aplicacion academica UPLA

## Que es
Nexo es una app multiplataforma para estudiantes de la UPLA.
Reimagina SIGMA: horario, notas, cuotas, pagos, Teams.
NO es oficial de la UPLA. Es un proyecto independiente.

## Creador
Alessandro Villogas Gaspar, estudiante de Ingenieria de Sistemas,
UPLA sede Huancayo, codigo U01025B. Fundador de Auralix Studio.

## Privacidad
- Sin servidores intermediarios. Comunicacion directa con SIGMA.
- Sin telemetria, sin publicidad, sin compras. Gratuita.

## Lumen
- Variantes: Ligero (~290 MB) y Estandar (~530 MB).
- 100% on-device. MediaPipe LLM Inference.
- Sin llamadas a APIs externas.
- Lumen viene del latin: luz.

## Plataformas
- Android (APK), Windows x64, Web demo: nexo.upla.dev.

## Versiones
- v1.1.1: Lumen integrado + Teams.
- v1.2.0: Session Intranet resiliente + actualizaciones in-app.
- v1.3: Persistencia historial (en desarrollo).

## Bugs
- GitHub: https://github.com/auralix-studio/nexo/issues
"""

KB_VIDA = """\
# Vida universitaria UPLA

## Calendario academico
- Semestre I: marzo/abril a julio.
- Semestre II: agosto/septiembre a diciembre.
- Parciales: semanas 8-9. Finales: semanas 16-17.
- Matricula: 1-2 semanas antes del inicio.

## Evaluacion
- Sistema vigesimal (0-20). Aprobatorio: 11.
- Continua + parcial + final. Pesos segun silabo.

## Servicios
- Bienestar Universitario: psicologia, tutoria, becas.
- Comedor universitario. Laboratorios. Wi-Fi campus.

## Consejos para nuevos
- Revisar el silabo al inicio del ciclo.
- Usar SIGMA para notas y cuotas.
- Aprovechar tutorias y biblioteca virtual.
"""

KB_FUNCIONES = """\
# Funciones de Lumen para consultar datos del estudiante

Cuando el usuario pregunta datos personales, Lumen emite [CALL]:funcion.
El runtime de Nexo ejecuta la funcion y devuelve [RESULT]datos[RESULT].
Lumen completa la respuesta con los datos reales.

## obtener_horario
Devuelve clases del estudiante: proxima, hoy, semana.
Usar para: clases, aulas, docentes, horario, dias.

## obtener_pagos
Devuelve cuotas y pagos: pendientes, vencidas, totales.
Usar para: deudas, pensiones, montos, mora, vencimiento.

## obtener_promedio
Devuelve promedio ciclo actual y acumulado, creditos.
Usar para: rendimiento academico, promedio, creditos.

## obtener_notas
Devuelve notas por curso del periodo activo (boleta).
Usar para: calificaciones de cursos especificos.

## obtener_perfil
Devuelve nombre, carrera, ciclo, creditos del estudiante.
Usar para: perfil academico, carrera, ciclo actual.
"""

ALL_KB = [
    KB_UPLA_GENERAL,
    KB_CARRERAS,
    KB_ASIGNATURAS,
    KB_TRAMITES,
    KB_NEXO,
    KB_VIDA,
    KB_FUNCIONES,
]

print(f"\n[OK] KB: {len(ALL_KB)} docs, {sum(len(k) for k in ALL_KB):,} chars")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - PERSONA
# ═══════════════════════════════════════════════════════════

PERSONA = [
    dlg("Quien eres?",
        "Soy Lumen, el asistente de Nexo. Vivo dentro de tu telefono, "
        "funciono sin internet y estoy hecho para ayudarte con tu vida "
        "academica en la UPLA."),
    dlg("Que eres?",
        "Soy Lumen. Un asistente que corre completamente en tu "
        "dispositivo, sin enviar tus datos a ningun servidor."),
    dlg("Como te llamas?",
        "Me llamo Lumen. Soy parte de la app Nexo para la UPLA."),
    dlg("Para que sirves?",
        "Te ayudo con tu horario, notas, cuotas y tramites de la UPLA. "
        "Tambien resuelvo calculos y te apoyo cuando lo necesites."),
    dlg("Que significa tu nombre?",
        "Lumen viene del latin y significa luz. Fui disenado para "
        "iluminar tu experiencia como estudiante de la UPLA."),
    dlg("Quien te creo?",
        "Fui creado para Nexo, una app hecha por Alessandro Villogas "
        "Gaspar, estudiante de Ingenieria de Sistemas de la UPLA, "
        "sede Huancayo. Fundador de Auralix Studio."),
    dlg("Quien hizo Nexo?",
        "Nexo fue creado por Alessandro Villogas Gaspar, codigo U01025B. "
        "Es un proyecto personal bajo Auralix Studio."),
    dlg("Envias mis datos a internet?",
        "No. Funciono 100% dentro de tu telefono. Tus datos no salen "
        "de tu dispositivo."),
    dlg("Mis datos estan seguros?",
        "Si. Nexo no tiene servidores propios. Tu informacion se queda "
        "en tu telefono."),
    dlg("Usas IA de la nube?",
        "No. Funciono completamente en tu dispositivo. Sin internet, "
        "sin enviar datos a nadie."),
    dlg("Que puedes hacer?",
        "Puedo consultar tu horario, cuotas, notas y perfil academico. "
        "Explico tramites, resuelvo calculos y te apoyo emocionalmente."),
    dlg("Puedes modificar mis datos en SIGMA?",
        "No. Soy un asistente de solo lectura. Para cambios usa el "
        "portal oficial de la UPLA."),
    dlg("Por que a veces te equivocas?",
        "Soy un modelo pequeno que corre en tu telefono. Si algo no "
        "parece correcto, verifica en el portal oficial de la UPLA."),
    dlg("Hola",
        "Hola! Soy Lumen, tu asistente en Nexo. En que puedo ayudarte?"),
    dlg("Buenos dias",
        "Buenos dias! Que necesitas? Estoy aqui para ayudarte."),
    dlg("Buenas tardes",
        "Buenas tardes! En que te ayudo hoy?"),
    dlg("Gracias Lumen.",
        "Para eso estoy. Cuando me necesites, aqui voy a estar."),
    dlg("Chau",
        "Hasta luego! Exitos con tus estudios."),
    dlg("Eres genial",
        "Gracias! Me alegra poder ayudarte."),
    dlg("Nexo es oficial de la UPLA?",
        "No, Nexo es independiente. No esta respaldado oficialmente "
        "por la Universidad Peruana Los Andes."),
    dlg("Nexo es gratis?",
        "Si, completamente gratis. Sin publicidad, sin compras."),
]

print(f"[OK] PERSONA: {len(PERSONA)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - HABILIDADES (con thinking para matematicas)
# ═══════════════════════════════════════════════════════════

SKILLS = [
    dlg("Cuanto es 15% de 2750?",
        "El 15% de 2750 es 412.5. (2750 x 0.15 = 412.5)."),
    dlg("Cuanto es 1200 entre 4?",
        "1200 / 4 = 300."),
    dlg("Cuanto es 3.14 por 7?",
        "3.14 x 7 = 21.98."),
    dlg("Si mi cuota es 550 soles y pago 3 cuotas, cuanto es?",
        "Tres cuotas de 550 soles suman 1650 soles (550 x 3)."),
    dlg("Tengo que pagar 5 cuotas de 650, cuanto en total?",
        "5 cuotas de 650 soles = 3250 soles."),
    dlg("Cuantos centimetros hay en 2.5 metros?",
        "2.5 metros = 250 centimetros (2.5 x 100)."),
    dlg("Cuantas horas son 180 minutos?",
        "180 minutos = 3 horas (180 / 60 = 3)."),
    dlg("Convierte 5 km a metros.",
        "5 kilometros = 5000 metros."),
    dlg("Cual es la raiz cuadrada de 144?",
        "La raiz cuadrada de 144 es 12 (12 x 12 = 144)."),
    dlg("Puedes ayudarme con un calculo?",
        "Claro, dime la operacion y lo resolvemos paso a paso."),

    dlg_think(
        "Tengo notas 14, 16 y 11. Cual es mi promedio?",
        think="Sumo: 14+16+11=41. Divido entre 3: 41/3=13.67.",
        response="Tu promedio es 13.67. Apruebas (minimo 11 en la UPLA)."),

    dlg_think(
        "Si mis notas son 12, 15, 08 y 17, apruebo?",
        think="Sumo: 12+15+8+17=52. Promedio: 52/4=13. Minimo: 11.",
        response="Tu promedio es 13. Si apruebas."),

    dlg_think(
        "Cuanto necesito en el final si llevo 14 en el parcial que vale 70%?",
        think="parcial=14, peso=0.70. Aporte=14x0.70=9.8. "
              "Necesito 11 total. Final=(11-9.8)/0.30=4.",
        response="Con 14 de parcial (70%) ya tienes 9.8 puntos. "
                 "Solo necesitas 4 en el final. Vas muy bien!"),

    dlg_think(
        "Cuanto necesito en el final si llevo 10 y vale 40%?",
        think="parcial=10, peso_parcial=0.60. Aporte=10x0.60=6.0. "
              "Final_minimo=(11-6.0)/0.40=12.5.",
        response="Con 10 de parcial (60%) tienes 6.0 puntos. "
                 "Necesitas al menos 12.5 en el final. Se puede!"),

    dlg_think(
        "Tengo 13 de parcial (60%) y 15 de final (40%). Cual es mi nota?",
        think="Aporte_parcial=13x0.60=7.8. Aporte_final=15x0.40=6.0. Total=13.8.",
        response="Tu nota del curso es 13.8. Apruebas con holgura."),

    dlg_think(
        "Mi pension es 800 soles y tengo mora de 5%. Cuanto pago?",
        think="Mora=800x0.05=40. Total=800+40=840.",
        response="Con 5% de mora sobre 800 soles, pagas 840 soles en total."),

    dlg_think(
        "Resuelve 3x - 9 = 0.",
        think="3x=9. x=3.",
        response="3x - 9 = 0 -> 3x = 9 -> x = 3."),

    dlg_think(
        "Resuelve 2x + 6 = 20.",
        think="2x=20-6=14. x=7.",
        response="2x + 6 = 20 -> 2x = 14 -> x = 7."),
]

print(f"[OK] SKILLS: {len(SKILLS)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - FUNCTION CALLING
# ═══════════════════════════════════════════════════════════

FUNCTION_DIALOGS = [
    # obtener_horario
    dlg_call(
        user="Cuando tengo clases hoy?",
        think="El usuario pregunta su horario de hoy. Debo consultarlo.",
        func="obtener_horario",
        result="hoy:[{curso:Algoritmos,tipo:LAB,inicio:08:00,fin:10:00,aula:B-201}," +
               "{curso:Fisica I,tipo:TEO,inicio:10:00,fin:12:00,aula:A-105}]",
        response="Hoy tienes dos clases: Algoritmos (lab, 08:00-10:00, aula B-201) "
                 "y Fisica I (teoria, 10:00-12:00, aula A-105)."),

    dlg_call(
        user="Cual es mi proxima clase?",
        think="Necesito el horario para saber la proxima clase.",
        func="obtener_horario",
        result="proxima:{curso:Calculo II,tipo:TEO,dia:martes,inicio:14:00,fin:16:00,aula:C-301}",
        response="Tu proxima clase es Calculo II el martes de 14:00 a 16:00 en el aula C-301."),

    dlg_call(
        user="Tengo clases el viernes?",
        think="El usuario pregunta el horario del viernes.",
        func="obtener_horario",
        result="dia:viernes,clases:[{curso:Estadistica,tipo:LAB,inicio:08:00,aula:Lab-02}]",
        response="El viernes tienes Estadistica en laboratorio de 08:00 en el Lab-02."),

    dlg_call(
        user="Cuantos cursos llevo este ciclo?",
        think="Necesito el horario para contar cursos unicos.",
        func="obtener_horario",
        result="total_cursos:5,semana:[Algoritmos,Fisica I,Calculo II,Estadistica,Comunicacion]",
        response="Este ciclo llevas 5 cursos: Algoritmos, Fisica I, Calculo II, "
                 "Estadistica y Comunicacion."),

    dlg_call(
        user="Que aula tengo para Algoritmos?",
        think="El usuario pregunta sobre un curso especifico en su horario.",
        func="obtener_horario",
        result="curso:Algoritmos,aula:B-201,docente:Ing. Lopez,dias:[lunes,miercoles]",
        response="Algoritmos es en el aula B-201 con el Ing. Lopez, lunes y miercoles."),

    dlg_call(
        user="A que hora es mi primera clase manana?",
        think="Necesito el horario del dia siguiente.",
        func="obtener_horario",
        result="manana:martes,primera:{curso:Algebra Lineal,inicio:07:00,aula:A-201}",
        response="Tu primera clase manana (martes) es Algebra Lineal a las 7:00 en el aula A-201."),

    # obtener_pagos
    dlg_call(
        user="Cuanto debo de cuotas?",
        think="El usuario pregunta sus pagos pendientes.",
        func="obtener_pagos",
        result="pendientes:2,total_pendiente:1300.00," +
               "detalle:[{concepto:Cuota 5,monto:650.00,vence:2026-06-20}," +
               "{concepto:Cuota 6,monto:650.00,vence:2026-07-20}]",
        response="Tienes 2 cuotas pendientes por S/. 1,300.00 en total. "
                 "La proxima vence el 20 de junio (S/. 650)."),

    dlg_call(
        user="Tengo cuotas vencidas?",
        think="Necesito verificar si hay cuotas vencidas.",
        func="obtener_pagos",
        result="vencidas:1,detalle:[{concepto:Cuota 4,monto:650.00,mora:32.50}]",
        response="Si, tienes 1 cuota vencida: Cuota 4 (S/. 650 + S/. 32.50 mora). "
                 "Te recomiendo pagarla pronto para evitar mas intereses."),

    dlg_call(
        user="Cuando vence mi proxima cuota?",
        think="El usuario pregunta la fecha de vencimiento.",
        func="obtener_pagos",
        result="proxima:{concepto:Cuota 6,monto:650.00,vence:2026-07-20}",
        response="Tu proxima cuota es la Cuota 6 (S/. 650) y vence el 20 de julio."),

    dlg_call(
        user="Cuanto tengo que pagar en total?",
        think="El usuario quiere saber el monto total pendiente.",
        func="obtener_pagos",
        result="pendientes:3,total_pendiente:1950.00",
        response="Tienes 3 cuotas pendientes por un total de S/. 1,950.00."),

    dlg_call(
        user="Tengo mora en alguna cuota?",
        think="Debo revisar si hay mora en las cuotas.",
        func="obtener_pagos",
        result="vencidas:0,pendientes:2,total_pendiente:1300.00",
        response="No tienes cuotas vencidas. Tus 2 cuotas estan al dia."),

    # obtener_promedio
    dlg_call(
        user="Cual es mi promedio?",
        think="El usuario pregunta su promedio academico.",
        func="obtener_promedio",
        result="promedio_ciclo_actual:14.20,promedio_acumulado:13.85,creditos_aprobados:72",
        response="Tu promedio del ciclo actual es 14.20 y tu acumulado es 13.85. "
                 "Llevas 72 creditos aprobados."),

    dlg_call(
        user="Como voy en mis notas este ciclo?",
        think="El usuario pregunta su rendimiento actual.",
        func="obtener_promedio",
        result="promedio_ciclo_actual:12.50,promedio_acumulado:13.10",
        response="Este ciclo vas con 12.50 de promedio. Tu acumulado es 13.10. "
                 "Apruebas, pero hay margen para mejorar."),

    dlg_call(
        user="Cuantos creditos me faltan para graduarme?",
        think="Necesito los creditos para responder.",
        func="obtener_promedio",
        result="creditos_aprobados:155,creditos_totales:200",
        response="Llevas 155 creditos de 200. Te faltan 45 para completar el plan."),

    dlg_call(
        user="Subio mi promedio este ciclo?",
        think="El usuario quiere comparar su promedio actual con el acumulado.",
        func="obtener_promedio",
        result="promedio_ciclo_actual:14.00,promedio_acumulado:13.50",
        response="Tu promedio de este ciclo (14.00) esta por encima del acumulado (13.50). "
                 "Vas mejorando!"),

    # obtener_notas
    dlg_call(
        user="Cuales son mis notas de este ciclo?",
        think="El usuario quiere ver sus calificaciones del periodo activo.",
        func="obtener_notas",
        result="periodo:2026-I,cursos:[{curso:Algoritmos,promedio:16.00,estado:Aprobado}," +
               "{curso:Fisica I,promedio:12.00,estado:Aprobado}," +
               "{curso:Calculo II,promedio:09.00,estado:En curso}]",
        response="En 2026-I tienes: Algoritmos 16 (aprobado), Fisica I 12 (aprobado), "
                 "Calculo II 9 (aun en curso)."),

    dlg_call(
        user="Que nota tengo en Algoritmos?",
        think="El usuario pregunta la nota de un curso especifico.",
        func="obtener_notas",
        result="cursos:[{curso:Algoritmos,promedio:16.00,estado:Aprobado}]",
        response="En Algoritmos tienes 16. Muy bien!"),

    dlg_call(
        user="Estoy jalando algun curso?",
        think="El usuario quiere saber si algun curso esta en riesgo.",
        func="obtener_notas",
        result="cursos:[{curso:Algoritmos,promedio:16,estado:Aprobado}," +
               "{curso:Fisica I,promedio:10,estado:En riesgo}]",
        response="Fisica I esta en riesgo con 10 (minimo 11). Aun queda el final. "
                 "Quieres calcular cuanto necesitas?"),

    # obtener_perfil
    dlg_call(
        user="En que ciclo estoy?",
        think="El usuario pregunta su nivel academico.",
        func="obtener_perfil",
        result="nombre:Carlos Mendoza,carrera:Ingenieria de Sistemas,ciclo:VI,creditos_aprobados:120",
        response="Estas en el VI ciclo de Ingenieria de Sistemas con 120 creditos aprobados."),

    dlg_call(
        user="Cual es mi carrera?",
        think="El usuario quiere ver su perfil academico.",
        func="obtener_perfil",
        result="carrera:Ingenieria Civil,ciclo:IV",
        response="Estudias Ingenieria Civil y estas en el IV ciclo."),

    dlg_call(
        user="Como me llamo en el sistema?",
        think="El usuario quiere ver su nombre en SIGMA.",
        func="obtener_perfil",
        result="nombre:Maria Rodriguez,carrera:Enfermeria,ciclo:III",
        response="En SIGMA tu nombre es Maria Rodriguez. Estudias Enfermeria, III ciclo."),
]

print(f"[OK] FUNCTION_DIALOGS: {len(FUNCTION_DIALOGS)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - CLARIFICACION (sin datos disponibles)
# ═══════════════════════════════════════════════════════════

CLARIFICATION_DIALOGS = [
    dlg("Cuando tengo clases hoy?",
        "No tengo tu horario cargado. Abre la pestana Horario en Nexo "
        "y toca Actualizar. Luego te doy el detalle."),
    dlg("Cual es mi proxima clase?",
        "Aun no tengo tu horario sincronizado. Ve a Horario, actualizalo "
        "y preguntame de nuevo."),
    dlg("Que aula tengo?",
        "No veo tu horario disponible. Sincronizalo en Nexo y te digo el aula."),
    dlg("Cuanto debo?",
        "No tengo tus cuotas cargadas. Abre Pagos en Nexo y sincroniza. "
        "Despues te digo el monto exacto."),
    dlg("Tengo cuotas vencidas?",
        "Aun no tengo tu informacion de pagos. Ve a Pagos -> Actualizar "
        "y vuelve a preguntarme."),
    dlg("Cuales son mis notas?",
        "No tengo tus notas del ciclo cargadas. Ve a la pestana de Notas "
        "en Nexo, sincroniza y te doy el detalle."),
    dlg("Que nota tengo en Fisica?",
        "No tengo tus notas disponibles. Sincroniza en Notas y te digo "
        "la nota de Fisica."),
    dlg("Cual es mi promedio?",
        "No tengo tu promedio cargado. Sincroniza en Notas o Perfil "
        "y te digo el promedio actualizado."),
    dlg_think(
        "Voy a aprobar el curso?",
        think="El usuario no dio notas ni pesos. No puedo calcular sin esos datos.",
        response="Para saberlo necesito: tu nota parcial y que porcentaje "
                 "vale el examen final. Dimelo y calculo al instante."),
    dlg_think(
        "Cuanto necesito en el final?",
        think="Falta la nota parcial y el peso del final para calcular.",
        response="Dime tu nota parcial y el porcentaje del final "
                 "(ejemplo: tengo 13 y el final vale 40%) y calculo ya."),
    dlg("Cuantos creditos me faltan?",
        "No tengo tu perfil cargado. Sincroniza en Nexo y te digo "
        "cuantos creditos te faltan."),
    dlg("Tengo clase manana?",
        "No tengo tu horario. Sincronizalo en Nexo. Si quieres, dime "
        "tu carrera y ciclo y te cuento como suelen ser los horarios."),
    dlg("Cuando vence mi pension?",
        "No tengo tus datos de pago. Abre Pagos en Nexo y actualizalo "
        "para ver la fecha exacta."),
]

print(f"[OK] CLARIFICATION_DIALOGS: {len(CLARIFICATION_DIALOGS)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - Q&A UPLA
# ═══════════════════════════════════════════════════════════

UPLA_QA = [
    dlg("Que es la UPLA?",
        "La Universidad Peruana Los Andes (UPLA) es una universidad "
        "privada fundada el 30 de diciembre de 1983 (Ley N 23757). "
        "Sede principal en Huancayo, Junin."),
    dlg("Cuando se fundo la UPLA?",
        "La UPLA fue fundada el 30 de diciembre de 1983 por la Ley N 23757."),
    dlg("Donde queda la UPLA?",
        "Sede central: Av. Giraldez 231, Huancayo, Junin. Filiales en "
        "Lima, Chanchamayo (La Merced) y Satipo."),
    dlg("Cual es el lema de la UPLA?",
        "El lema es Honor, ciencia y desarrollo."),
    dlg("La UPLA esta licenciada?",
        "Si, la UPLA esta licenciada por la SUNEDU desde 2019."),
    dlg("Cuantas facultades tiene la UPLA?",
        "Seis: Medicina Humana, Ciencias de la Salud, Ingenieria, "
        "Derecho y Ciencias Politicas, Ciencias Administrativas y "
        "Contables, y Educacion y Ciencias Humanas."),
    dlg("Que carreras de ingenieria hay en la UPLA?",
        "Ingenieria de Sistemas y Computacion, Civil, Industrial, "
        "Mecanica Electrica y Ambiental. Todas 10 semestres."),
    dlg("Cuanto dura Ingenieria de Sistemas?",
        "10 semestres (5 anos). Titulo: Ingeniero de Sistemas y Computacion."),
    dlg("Cuanto dura Medicina?",
        "14 semestres mas internado y SERUMS. Titulo: Medico Cirujano."),
    dlg("Cuanto dura Derecho?",
        "12 semestres. Titulo: Abogado/a."),
    dlg("Que carreras de salud ofrece la UPLA?",
        "Enfermeria, Obstetricia, Tecnologia Medica, Odontologia, "
        "Farmacia y Bioquimica, y Psicologia. La mayoria 10 semestres."),
    dlg("Hay carreras de Administracion en la UPLA?",
        "Si: Administracion y Contabilidad. Ambas 10 semestres."),
    dlg("Como obtengo el grado de bachiller?",
        "Es automatico al culminar el plan de estudios y los creditos. "
        "Luego se tramita el diploma via SUNEDU."),
    dlg("Como obtengo el titulo profesional?",
        "Debes sustentar una tesis o trabajo de suficiencia profesional."),
    dlg("Que es el FUT?",
        "El Formulario Unico de Tramite es el documento estandar para "
        "solicitudes formales en la UPLA."),
    dlg("Como saco una constancia de matricula?",
        "SIGMA -> Tramites -> Constancia. Tarda 1-3 dias habiles."),
    dlg("Como retiro un curso?",
        "FUT justificado antes de la primera evaluacion parcial. "
        "Limite de 1-2 cursos por semestre."),
    dlg("Puedo cambiarme de carrera dentro de la UPLA?",
        "Si, es un traslado interno. Se solicita al inicio del periodo. "
        "Minimo de creditos y convalidacion caso por caso."),
    dlg("Que cursos llevo en el primer ciclo?",
        "Comunicacion I, Matematica Basica, Filosofia, Metodologia del "
        "Trabajo Universitario y Realidad Nacional."),
    dlg("Que es Analisis Matematico y que prerequisito tiene?",
        "Analisis Matematico I, II y III cubren calculo diferencial, "
        "integral y vectorial. Son secuenciales: apruebas I para llevar II."),
    dlg("Cuantos cursos se llevan por ciclo?",
        "Ciclos 1-3: 5-7 cursos. Ciclos 4-7: 18-22 creditos. "
        "Ciclos 8-10: 16-18 creditos mas practicas."),
    dlg("Que es SIGMA?",
        "SIGMA (sigma.upla.edu.pe) es el sistema academico de la UPLA. "
        "Ahi ves matricula, notas, horarios y cuotas."),
    dlg("Que es la Intranet de la UPLA?",
        "La Intranet complementa SIGMA con pensiones, ranking y malla curricular."),
    dlg("La biblioteca de la UPLA es virtual?",
        "Tiene fisica en sede central y virtual 24/7 con credenciales SIGMA."),
    dlg("Cual es la nota minima para aprobar en la UPLA?",
        "11 sobre 20 (sistema vigesimal)."),
    dlg("Como se evaluan los cursos?",
        "Evaluacion continua, parcial y final. Pesos segun silabo del docente."),
    dlg("Cuantas sedes tiene la UPLA?",
        "Sede central en Huancayo mas filiales en Lima, Chanchamayo y Satipo."),
    dlg("Que es Bienestar Universitario?",
        "La oficina que brinda apoyo psicologico, tutoria y orientacion "
        "sobre becas y servicios al estudiante."),
    dlg("Que es Nexo?",
        "Una app no oficial para estudiantes de la UPLA. Reune horario, "
        "notas, pagos y Teams. Creada por Alessandro Villogas Gaspar."),
    dlg("Como descargo Nexo?",
        "Desde GitHub: github.com/auralix-studio/nexo/releases. "
        "Disponible para Android y Windows."),
    dlg("Puedo usar Nexo sin Lumen?",
        "Si. Lumen es opcional. Puedes usar Nexo para horario, notas "
        "y pagos sin activar el asistente."),
]

print(f"[OK] UPLA_QA: {len(UPLA_QA)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - APOYO EMOCIONAL
# ═══════════════════════════════════════════════════════════

EMOTIONAL_SUPPORT = [
    dlg("Me siento estresado con los examenes.",
        "Te entiendo, la epoca de examenes pesa. Respira: vamos por "
        "partes. Dime que curso te preocupa mas. No estas solo."),
    dlg("Estoy desanimado, creo que voy a jalar un curso.",
        "Un mal momento no define tu carrera. Dime en que curso estas "
        "y que notas te faltan. A veces hay mas margen del que parece."),
    dlg("No se si elegi bien mi carrera.",
        "Es normal tener esas dudas. Date tiempo para conocer las "
        "materias. Si quieres cambiar, en la UPLA existe el traslado interno."),
    dlg("Me siento solo en la universidad.",
        "La universidad puede ser abrumadora. Bienestar Universitario "
        "ofrece apoyo. Y aqui estoy para lo que necesites."),
    dlg("Estoy pensando en dejar la universidad.",
        "Es una decision grande. Habla con Bienestar Universitario. "
        "A veces un retiro de semestre puede ser mejor que dejarlo del todo."),
    dlg("No me alcanza el dinero para la universidad.",
        "Consulta en Bienestar Universitario sobre becas y fraccionamiento. "
        "Hay opciones de apoyo economico."),
    dlg("Tengo ansiedad por las exposiciones.",
        "Es comun. Prepara el tema, practica con alguien y lleva notas. "
        "La ansiedad baja con la practica. Tu puedes."),
    dlg("Me fue muy mal en un examen.",
        "Un examen malo no define todo. Identifica que fallo y enfocate "
        "en las evaluaciones que faltan. Todavia hay margen."),
    dlg("Estoy orgulloso de mis notas.",
        "Felicidades! Tu esfuerzo esta dando resultados. Sigue asi."),
    dlg("Gracias por escucharme.",
        "Para eso estoy. No tienes que enfrentar las cosas solo."),
    dlg("Me cuesta concentrarme para estudiar.",
        "Prueba estudiar en bloques de 25 minutos con descansos de 5 "
        "(tecnica Pomodoro). Elimina distracciones. Quieres que "
        "te ayude a organizar el tema?"),
    dlg("Gracias Lumen.",
        "Para eso estoy. Cuando me necesites, aqui voy a estar."),
]

print(f"[OK] EMOTIONAL_SUPPORT: {len(EMOTIONAL_SUPPORT)} dialogos")


# ═══════════════════════════════════════════════════════════
#   DIALOGOS - NEXO
# ═══════════════════════════════════════════════════════════

NEXO_DIALOGS = [
    dlg("Como funciona Nexo?",
        "Nexo se conecta directamente a SIGMA e Intranet con tus "
        "credenciales. Sin servidores intermediarios. Todo local."),
    dlg("Nexo recoge mis datos?",
        "No. Sin telemetria, sin publicidad, sin estadisticas. "
        "Tus datos se quedan en tu telefono."),
    dlg("Como activo a Lumen?",
        "Toca el boton flotante en la esquina inferior derecha. "
        "La primera vez elige Ligero o Estandar y descarga el modelo."),
    dlg("Que modelo de Lumen debo elegir?",
        "Con 2-3 GB de RAM elige Lumen Ligero (~290 MB). "
        "Con 4 GB o mas, Lumen Estandar (~530 MB) da mejores respuestas."),
    dlg("Nexo funciona sin internet?",
        "Parcialmente. Necesitas internet para sincronizar. Los datos "
        "en cache se usan sin conexion. Lumen funciona sin internet."),
    dlg("Como reporto un error en Nexo?",
        "GitHub: github.com/auralix-studio/nexo/issues. "
        "Incluye pasos, captura y version de Nexo."),
    dlg("En que plataformas funciona Nexo?",
        "Android (APK), Windows (ZIP portable) y web demo. "
        "Tambien compila para iOS, macOS y Linux."),
    dlg("Nexo tiene notificaciones?",
        "Si. Recordatorios de clases (15 min antes), avisos de cuotas "
        "(1 dia antes) y deteccion de notas nuevas."),
    dlg("Que es el patron Resolver de Nexo?",
        "Si SIGMA no responde, Nexo usa automaticamente la Intranet. "
        "Asi la app sigue funcionando."),
    dlg("Puedo cambiar el modelo de Lumen?",
        "Si. Lumen -> Configuracion -> Cambiar modelo."),
]

print(f"[OK] NEXO_DIALOGS: {len(NEXO_DIALOGS)} dialogos")


# ═══════════════════════════════════════════════════════════
#   PARAFRASEO Y CORPUS
# ═══════════════════════════════════════════════════════════

PARAPHRASE_PREFIX = [
    "", "Oye, ", "Una consulta: ", "Disculpa, ", "Hola Lumen, ",
    "Queria saber, ", "Dime, ", "Por favor, ", "Hey, ",
    "Tengo una duda, ", "Me puedes decir, ", "Necesito saber, ",
    "A ver, ", "Cuentame, ",
]


def paraphrase(turns: list, factor: int) -> list:
    """Genera variantes con prefijos."""
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


def load_spanish_base(n_docs: int) -> list:
    """Wikipedia en espanol - base linguistica minima."""
    if n_docs <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] datasets no instalado. Saltando Wikipedia.")
        return []
    print(f"  Descargando {n_docs:,} articulos de Wikipedia ES...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.es",
                      split="train", streaming=True)
    docs = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        t = (row.get("text") or "").strip()
        if len(t) > 200:
            docs.append(t[:3000])
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1:,} articulos")
    print(f"[OK] Wikipedia ES: {len(docs):,} docs")
    return docs


def build_corpus() -> str:
    """Construye el corpus de entrenamiento."""
    print("\n" + "=" * 60)
    print("  CONSTRUYENDO CORPUS")
    print("=" * 60)

    parts = []

    # 1) Base linguistica minima
    parts += load_spanish_base(cfg.N_WIKIPEDIA_DOCS)
    print(f"  Wikipedia: {len(parts):,} bloques")

    # 2) KB UPLA/Nexo/Lumen (alta prioridad, muchas repeticiones)
    for _ in range(cfg.KB_REPEAT):
        parts += ALL_KB
    print(f"[OK] KB UPLA: {len(ALL_KB)} x {cfg.KB_REPEAT} = {len(ALL_KB) * cfg.KB_REPEAT} bloques")

    # 3) Dialogos
    all_dialogs = list(PERSONA) + list(SKILLS)
    all_dialogs += paraphrase(FUNCTION_DIALOGS, factor=3)
    all_dialogs += paraphrase(CLARIFICATION_DIALOGS, factor=3)
    all_dialogs += paraphrase(UPLA_QA, factor=cfg.PARAPHRASE_FACTOR)
    all_dialogs += paraphrase(EMOTIONAL_SUPPORT, factor=cfg.PARAPHRASE_FACTOR)
    all_dialogs += paraphrase(NEXO_DIALOGS, factor=cfg.PARAPHRASE_FACTOR)

    print(f"[OK] Dialogos con parafraseo: {len(all_dialogs):,}")

    for _ in range(cfg.DIALOG_REPEAT):
        random.shuffle(all_dialogs)
        parts += all_dialogs

    print(f"[OK] Dialogos x {cfg.DIALOG_REPEAT} = {len(all_dialogs) * cfg.DIALOG_REPEAT:,} bloques")

    random.shuffle(parts)
    corpus = "\n\n".join(parts)
    cfg.CORPUS_FILE.write_text(corpus, encoding="utf-8")
    mb = len(corpus.encode("utf-8")) / 1e6
    print(f"\n  Corpus -> {cfg.CORPUS_FILE}")
    print(f"  Tamano: {mb:.1f} MB, {len(parts):,} bloques")
    print("-" * 60 + "\n")
    return corpus


corpus_text = build_corpus()


# ═══════════════════════════════════════════════════════════
#   TOKENIZADOR
# ═══════════════════════════════════════════════════════════

def train_tokenizer(corpus: str):
    print("\n" + "=" * 60)
    print("  ENTRENANDO TOKENIZADOR")
    print("=" * 60)

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
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
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

    test = "Hola Lumen, cuando es mi proxima clase?"
    print(f"\n  Test: '{test}'")
    enc = sp.encode(test)
    print(f"  Tokens: {enc[:15]} ({len(enc)} tokens)")
    print(f"  Decoded: '{sp.decode(enc)}'")

    sp_input.unlink(missing_ok=True)
    return sp


tokenizer = train_tokenizer(corpus_text)

PAD_ID      = tokenizer.piece_to_id(TOK_PAD)
BOS_ID      = tokenizer.piece_to_id(TOK_BOS)
EOS_ID      = tokenizer.piece_to_id(TOK_EOS)
USER_ID     = tokenizer.piece_to_id(TOK_USER)
LUMEN_ID    = tokenizer.piece_to_id(TOK_LUMEN)
END_ID      = tokenizer.piece_to_id(TOK_END)
THINK_ID    = tokenizer.piece_to_id(TOK_THINK)
ENDTHINK_ID = tokenizer.piece_to_id(TOK_ENDTHINK)
CALL_ID     = tokenizer.piece_to_id(TOK_CALL)
RESULT_ID   = tokenizer.piece_to_id(TOK_RESULT)

print(f"\nIDs: PAD={PAD_ID} BOS={BOS_ID} EOS={EOS_ID}")
print(f"     USER={USER_ID} LUMEN={LUMEN_ID} END={END_ID}")
print(f"     THINK={THINK_ID} /THINK={ENDTHINK_ID}")
print(f"     CALL={CALL_ID} RESULT={RESULT_ID}")


# ═══════════════════════════════════════════════════════════
#   DATASET con mascara de respuesta (response-only loss)
# ═══════════════════════════════════════════════════════════

class LumenDataset(Dataset):
    """
    Mejoras sobre v1.x:
    1. Divide por bloques logicos (no arbitrariamente).
    2. Mascara de respuesta: solo los tokens de Lumen contribuyen a la loss.
    """

    def __init__(self, corpus_path: Path, tokenizer, seq_len: int):
        print("  Tokenizando corpus por bloques logicos...")
        text   = corpus_path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

        self.seq_len   = seq_len
        self.sequences = []
        self.masks     = []

        for block in blocks:
            block_ids = tokenizer.encode(block)
            if not block_ids:
                continue
            full = [BOS_ID] + block_ids + [EOS_ID]

            if len(full) <= seq_len + 1:
                pad_len = (seq_len + 1) - len(full)
                padded  = full + [PAD_ID] * pad_len
                self.sequences.append(padded)
                self.masks.append(self._make_mask(padded))
            else:
                stride = seq_len // 2
                i = 0
                while i < len(full):
                    chunk = full[i:i + seq_len + 1]
                    if len(chunk) < seq_len + 1:
                        chunk = chunk + [PAD_ID] * ((seq_len + 1) - len(chunk))
                    self.sequences.append(chunk)
                    self.masks.append(self._make_mask(chunk))
                    if i + seq_len + 1 >= len(full):
                        break
                    i += stride

        print(f"  Secuencias: {len(self.sequences):,} (largo={seq_len})")

    def _make_mask(self, seq: list) -> list:
        """
        True para tokens de la respuesta de Lumen: [LUMEN]...[END]
        Excluye los tokens de entrada del usuario y el bloque de resultado [RESULT]...[RESULT].
        """
        mask        = [False] * len(seq)
        in_response = False
        in_result   = False
        for i, tok in enumerate(seq):
            if tok == LUMEN_ID:
                in_response = True
                continue
            if tok == RESULT_ID:
                if not in_result:
                    in_result   = True
                    in_response = False
                else:
                    in_result   = False
                    in_response = True
                continue
            if tok == END_ID:
                mask[i]     = True
                in_response = False
                continue
            if in_response and not in_result:
                mask[i] = True
        return mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq  = self.sequences[idx]
        mask = self.masks[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:],  dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.bool)
        return x, y, m


dataset = LumenDataset(cfg.CORPUS_FILE, tokenizer, cfg.MAX_SEQ_LEN)
dataloader = DataLoader(
    dataset,
    batch_size  = cfg.BATCH_SIZE,
    shuffle     = True,
    num_workers = 0,
    pin_memory  = True,
    drop_last   = True,
)

print(f"[OK] DataLoader: {len(dataloader):,} batches de {cfg.BATCH_SIZE}")
print(f"  Tokens por epoca: {len(dataset) * cfg.MAX_SEQ_LEN:,}")


# ═══════════════════════════════════════════════════════════
#   ARQUITECTURA DEL MODELO
# ═══════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos_f, sin_f):
    d    = x.shape[-1]
    x1   = x[..., :d // 2]
    x2   = x[..., d // 2:]
    T    = x.shape[2]
    cos  = cos_f[:T].unsqueeze(0).unsqueeze(0).to(x.device)
    sin  = sin_f[:T].unsqueeze(0).unsqueeze(0).to(x.device)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos_f, sin_f, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos_f, sin_f)
        k = apply_rope(k, cos_f, sin_f)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff, bias=False)
        self.w2   = nn.Linear(d_ff, d_model, bias=False)
        self.w3   = nn.Linear(d_model, d_ff, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, cos_f, sin_f, mask=None):
        x = x + self.attn(self.attn_norm(x), cos_f, sin_f, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    response_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy con:
    - Ignorar tokens de padding.
    - Ignorar tokens fuera de la respuesta de Lumen (response_mask).
    - Label smoothing.
    """
    B, T, V = logits.shape
    if label_smoothing > 0.0:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, label_smoothing / (V - 1))
            smooth.scatter_(-1, targets.unsqueeze(-1).clamp(min=0),
                            1.0 - label_smoothing)
        per_token = -(smooth * log_probs).sum(-1)
    else:
        per_token = F.cross_entropy(
            logits.view(-1, V), targets.view(-1),
            ignore_index=pad_id, reduction="none",
        ).view(B, T)

    ignore = targets == pad_id
    if response_mask is not None:
        ignore = ignore | ~response_mask
    per_token = per_token.masked_fill(ignore, 0.0)
    return per_token.sum() / (~ignore).sum().clamp(min=1)


class LumenModel(nn.Module):
    """Lumen v2.0 - GPT decoder-only con RoPE, RMSNorm y SwiGLU."""

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.token_emb   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout     = nn.Dropout(dropout)
        self.layers      = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm   = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight  # weight tying

        cos_f, sin_f = precompute_rope_freqs(d_model // n_heads, max_seq_len)
        self.register_buffer("cos_freqs", cos_f)
        self.register_buffer("sin_freqs", sin_f)
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'=' * 60}")
        print(f"  LUMEN v2.0 - {n_params/1e6:.1f}M parametros")
        print(f"  d_model={d_model} n_heads={n_heads} n_layers={n_layers}")
        print(f"  d_ff={d_ff} max_seq={max_seq_len} vocab={vocab_size}")
        print(f"{'=' * 60}\n")

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

    def forward(self, x, targets=None, response_mask=None):
        B, T = x.shape
        assert T <= self.max_seq_len
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        h    = self.dropout(self.token_emb(x))
        for layer in self.layers:
            h = layer(h, self.cos_freqs, self.sin_freqs, mask)
        logits = self.output(self.norm(h))
        loss   = None
        if targets is not None:
            loss = masked_cross_entropy(
                logits, targets,
                pad_id=PAD_ID,
                response_mask=response_mask,
                label_smoothing=cfg.LABEL_SMOOTHING,
            )
        return logits, loss


model = LumenModel(
    vocab_size  = tokenizer.get_piece_size(),
    d_model     = cfg.D_MODEL,
    n_heads     = cfg.N_HEADS,
    n_layers    = cfg.N_LAYERS,
    d_ff        = cfg.D_FF,
    max_seq_len = cfg.MAX_SEQ_LEN,
    dropout     = cfg.DROPOUT,
).to(DEVICE)


# ═══════════════════════════════════════════════════════════
#   OPTIMIZADOR + SCHEDULER
# ═══════════════════════════════════════════════════════════

def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


decay_p, no_decay_p = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        (decay_p if p.dim() >= 2 else no_decay_p).append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay_p, "weight_decay": cfg.WEIGHT_DECAY},
     {"params": no_decay_p, "weight_decay": 0.0}],
    lr=cfg.LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8,
)
scaler = GradScaler()

print(f"[OK] AdamW | lr={cfg.LEARNING_RATE} | wd={cfg.WEIGHT_DECAY}")
print(f"  Warmup: {cfg.WARMUP_STEPS} | Total: {cfg.MAX_STEPS}")
print(f"  Batch efectivo: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")
print(f"  Label smoothing: {cfg.LABEL_SMOOTHING} | Response-only loss: ON")


# ═══════════════════════════════════════════════════════════
#   GENERACION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=200,
             temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    ids       = [BOS_ID] + tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_tokens):
        if input_ids.shape[1] > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]
        logits, _ = model(input_ids)
        logits    = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            probs  = F.softmax(sl, dim=-1)
            cumul  = torch.cumsum(probs, dim=-1)
            sl[(cumul - probs) > top_p] = float("-inf")
            logits = sl.scatter(1, si, sl)

        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if next_tok.item() in (EOS_ID, END_ID):
            break
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    return tokenizer.decode(input_ids[0].tolist()[len(ids):])


EVAL_PROMPTS = [
    f"{TOK_USER} Quien eres? {TOK_LUMEN}",
    f"{TOK_USER} Que es la UPLA? {TOK_LUMEN}",
    f"{TOK_USER} Cuando tengo clases hoy? {TOK_LUMEN}",
    f"{TOK_USER} Cuanto debo de cuotas? {TOK_LUMEN}",
    f"{TOK_USER} Cual es mi promedio? {TOK_LUMEN}",
    f"{TOK_USER} Tengo notas 14, 16 y 11. Cual es mi promedio? {TOK_LUMEN}",
    f"{TOK_USER} Cuanto necesito en el final si llevo 14 y vale 70%? {TOK_LUMEN}",
    f"{TOK_USER} Que nota tengo en Fisica? {TOK_LUMEN}",
    f"{TOK_USER} Cuanto dura Ingenieria de Sistemas? {TOK_LUMEN}",
    f"{TOK_USER} Como saco una constancia de matricula? {TOK_LUMEN}",
]


def run_evaluation(model, tokenizer, step):
    print(f"\n{'=' * 60}")
    print(f"  EVALUACION - Paso {step:,}")
    print("=" * 60)
    for prompt in EVAL_PROMPTS[:6]:
        resp = generate(model, tokenizer, prompt,
                        max_tokens=cfg.GEN_MAX_TOKENS,
                        temperature=cfg.GEN_TEMPERATURE,
                        top_k=cfg.GEN_TOP_K, top_p=cfg.GEN_TOP_P)
        q = prompt.split(TOK_USER)[-1].split(TOK_LUMEN)[0].strip()
        print(f"\n  Q: {q}")
        print(f"  A: {resp[:300]}")
    print(f"\n{'=' * 60}\n")
    model.train()


# ═══════════════════════════════════════════════════════════
#   CHECKPOINTS
# ═══════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scaler, step, loss, path):
    ckpt = {
        "step": step, "loss": loss,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "config": {
            "vocab_size":  tokenizer.get_piece_size(),
            "d_model":     cfg.D_MODEL, "n_heads": cfg.N_HEADS,
            "n_layers":    cfg.N_LAYERS, "d_ff": cfg.D_FF,
            "max_seq_len": cfg.MAX_SEQ_LEN,
            "special_tokens": dict(zip(
                ["pad","bos","eos","user","lumen","end","think","endthink","call","result"],
                SPECIAL_TOKENS
            )),
        },
    }
    torch.save(ckpt, path)
    print(f"  [CKPT] {path.name} - paso {step:,}, loss {loss:.4f}")
    if USING_GDRIVE:
        torch.save(ckpt, cfg.GDRIVE_SAVE_DIR / path.name)
        print(f"  [Drive] Guardado")


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  [CKPT] Reanudando desde {path.name} - paso {ckpt['step']:,}")
    return ckpt["step"], ckpt.get("loss", float("inf"))


# ═══════════════════════════════════════════════════════════
#   LOOP DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════

def train():
    print("\n" + "=" * 60)
    print("  INICIANDO ENTRENAMIENTO LUMEN v2.0")
    print("=" * 60)
    print(f"  Device: {DEVICE} | Pasos: {cfg.MAX_STEPS:,}")
    print(f"  Batch efectivo: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")
    print(f"  Response-only loss + Label Smoothing {cfg.LABEL_SMOOTHING}")
    print("=" * 60 + "\n")

    model.train()
    global_step   = 0
    running_loss  = 0.0
    best_loss     = float("inf")
    start_time    = time.time()
    tok_processed = 0
    last_log_time = start_time
    last_tok      = 0

    # Resume desde checkpoint
    existing = sorted(cfg.CHECKPOINT_DIR.glob("lumen_step_*.pt"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    if not existing and USING_GDRIVE:
        gdrive_ckpts = sorted(cfg.GDRIVE_SAVE_DIR.glob("lumen_step_*.pt"),
                              key=lambda p: int(p.stem.split("_")[-1]))
        if gdrive_ckpts:
            import shutil
            local = cfg.CHECKPOINT_DIR / gdrive_ckpts[-1].name
            shutil.copy(str(gdrive_ckpts[-1]), str(local))
            existing = [local]
            print(f"  [Drive] Restaurado: {gdrive_ckpts[-1].name}")

    if existing:
        global_step, best_loss = load_checkpoint(
            existing[-1], model, optimizer, scaler)
        print(f"  Reanudando desde paso {global_step:,}\n")

    epoch = 0
    while global_step < cfg.MAX_STEPS:
        epoch += 1
        for batch_idx, (x, y, resp_mask) in enumerate(dataloader):
            if global_step >= cfg.MAX_STEPS:
                break

            x         = x.to(DEVICE, non_blocking=True)
            y         = y.to(DEVICE, non_blocking=True)
            resp_mask = resp_mask.to(DEVICE, non_blocking=True)

            with autocast(dtype=torch.float16):
                _, loss = model(x, targets=y, response_mask=resp_mask)
                loss    = loss / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            tok_processed += x.numel()

            if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)

                lr = get_lr(global_step, cfg.WARMUP_STEPS,
                            cfg.MAX_STEPS, cfg.LEARNING_RATE)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step  += 1
                running_loss += loss.item() * cfg.GRAD_ACCUM_STEPS

                if global_step % cfg.LOG_EVERY == 0:
                    avg_loss    = running_loss / cfg.LOG_EVERY
                    now         = time.time()
                    step_time   = (now - last_log_time) / cfg.LOG_EVERY
                    tok_per_sec = (tok_processed - last_tok) / (now - last_log_time)
                    eta_h       = (cfg.MAX_STEPS - global_step) * step_time / 3600
                    print(
                        f"  Paso {global_step:>6,}/{cfg.MAX_STEPS:,} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Tok/s: {tok_per_sec:,.0f} | Epoca: {epoch} | ETA: {eta_h:.1f}h"
                    )
                    running_loss  = 0.0
                    last_log_time = now
                    last_tok      = tok_processed

                if global_step % cfg.EVAL_EVERY == 0:
                    run_evaluation(model, tokenizer, global_step)
                    model.train()

                if global_step % cfg.SAVE_EVERY == 0:
                    ckpt_path    = cfg.CHECKPOINT_DIR / f"lumen_step_{global_step}.pt"
                    current_loss = loss.item() * cfg.GRAD_ACCUM_STEPS
                    save_checkpoint(model, optimizer, scaler,
                                    global_step, current_loss, ckpt_path)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        save_checkpoint(model, optimizer, scaler, global_step,
                                        current_loss,
                                        cfg.CHECKPOINT_DIR / "lumen_best.pt")

        print(f"\n  Epoca {epoch} completada.\n")

    total = time.time() - start_time
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"  Pasos: {global_step:,} | Tiempo: {total/3600:.2f}h")
    print(f"  Mejor loss: {best_loss:.4f} | Tokens: {tok_processed:,}")
    return model


trained_model = train()


# ═══════════════════════════════════════════════════════════
#   EVALUACION FINAL
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  EVALUACION FINAL")
print("=" * 60)

for prompt in EVAL_PROMPTS:
    resp = generate(trained_model, tokenizer, prompt,
                    max_tokens=cfg.GEN_MAX_TOKENS,
                    temperature=cfg.GEN_TEMPERATURE,
                    top_k=cfg.GEN_TOP_K, top_p=cfg.GEN_TOP_P)
    q = prompt.split(TOK_USER)[-1].split(TOK_LUMEN)[0].strip()
    print(f"\n  Q: {q}")
    print(f"  A: {resp[:500]}")


# ═══════════════════════════════════════════════════════════
#   EXPORTACION
# ═══════════════════════════════════════════════════════════

def export_model(model, tokenizer):
    print("\n" + "=" * 60)
    print("  EXPORTANDO MODELO FINAL")
    print("=" * 60)

    pt_path = cfg.FINAL_MODEL_DIR / "lumen_model.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"  [OK] PyTorch: {pt_path} ({pt_path.stat().st_size/1e6:.1f} MB)")

    config = {
        "model_name": "Lumen", "version": "2.0.0",
        "creator": "Alessandro Villogas Gaspar",
        "organization": "Auralix Studio",
        "description": "Asistente IA on-device para Nexo / UPLA",
        "architecture": "decoder-only-transformer",
        "vocab_size": tokenizer.get_piece_size(),
        "d_model": cfg.D_MODEL, "n_heads": cfg.N_HEADS,
        "n_layers": cfg.N_LAYERS, "d_ff": cfg.D_FF,
        "max_seq_len": cfg.MAX_SEQ_LEN,
        "special_tokens": dict(zip(
            ["pad","bos","eos","user","lumen","end","think","endthink","call","result"],
            SPECIAL_TOKENS
        )),
        "features": {
            "function_calling": True,
            "thinking_mode": True,
            "response_only_loss": True,
            "label_smoothing": cfg.LABEL_SMOOTHING,
        },
        "training": {
            "steps": cfg.MAX_STEPS,
            "batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
            "learning_rate": cfg.LEARNING_RATE,
            "warmup_steps": cfg.WARMUP_STEPS,
            "wikipedia_docs": cfg.N_WIKIPEDIA_DOCS,
            "kb_repeat": cfg.KB_REPEAT,
        },
    }
    cfg_path = cfg.FINAL_MODEL_DIR / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Config: {cfg_path}")

    import shutil
    for ext in [".model", ".vocab"]:
        src = Path(str(cfg.TOKENIZER_PREFIX) + ext)
        if src.exists():
            dst = cfg.FINAL_MODEL_DIR / f"tokenizer{ext}"
            shutil.copy2(src, dst)
            print(f"  [OK] Tokenizador: {dst}")

    try:
        from safetensors.torch import save_model
        st_path = cfg.FINAL_MODEL_DIR / "lumen_model.safetensors"
        save_model(model, str(st_path))
        print(f"  [OK] SafeTensors: {st_path} ({st_path.stat().st_size/1e6:.1f} MB)")
    except ImportError:
        print("  [WARN] safetensors no disponible - solo se guardo .pt")

    if USING_GDRIVE:
        dest = cfg.GDRIVE_SAVE_DIR / "lumen_final"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cfg.FINAL_MODEL_DIR, dest)
        print(f"  [Drive] Copia completa en {dest}")

    print("\n" + "=" * 60)
    print("  MODELO LUMEN v2.0 EXPORTADO")
    print("=" * 60)
    print(f"  -> lumen_model.safetensors")
    print(f"  -> config.json")
    print(f"  -> tokenizer.model")
    print("  Siguiente: convertir a TFLite/MediaPipe con export_lumen.py")


export_model(trained_model, tokenizer)


# ═══════════════════════════════════════════════════════════
#   CHAT INTERACTIVO (opcional)
# ═══════════════════════════════════════════════════════════

def chat(model, tokenizer):
    print("\n" + "=" * 60)
    print("  CHAT CON LUMEN v2.0")
    print("=" * 60 + "\n")
    while True:
        try:
            user_input = input("  Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("salir", "exit"):
            print("  Hasta luego!")
            break
        resp = generate(model, tokenizer,
                        f"{TOK_USER} {user_input} {TOK_LUMEN}",
                        max_tokens=cfg.GEN_MAX_TOKENS,
                        temperature=cfg.GEN_TEMPERATURE,
                        top_k=cfg.GEN_TOP_K, top_p=cfg.GEN_TOP_P)
        print(f"  Lumen: {resp}\n")


print("\n  Script completado. Para chatear: chat(trained_model, tokenizer)")
