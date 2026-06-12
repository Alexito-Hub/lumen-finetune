"""

              LUMEN — Entrenamiento desde cero en Google Colab       

  Modelo:  Lumen (asistente IA on-device para Nexo / UPLA)           
  Autor:   Alessandro Villogas Gaspar — Auralix Studio               
  Fecha:   Junio 2026                                                 

  Este script entrena un modelo de lenguaje tipo GPT desde cero       
  (pesos aleatorios) para que funcione como asistente de la UPLA      
  dentro de la aplicación Nexo.                                       

   Cómo usar en Google Colab                                      
  1. Subir este archivo a Colab o copiar celda por celda              
  2. Cambiar Runtime -> GPU (T4 o mejor)                               
  3. Ejecutar todo secuencialmente                                    
  4. El modelo se guarda en Google Drive al final                     

"""

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("sentencepiece")
install("datasets")
install("safetensors")

import os, math, json, time, random, textwrap, io
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
    """Configuración centralizada del entrenamiento."""

    WORK_DIR          = Path("./lumen_training")
    CORPUS_FILE       = WORK_DIR / "corpus.txt"
    TOKENIZER_PREFIX  = WORK_DIR / "lumen_tokenizer"
    CHECKPOINT_DIR    = WORK_DIR / "checkpoints"
    FINAL_MODEL_DIR   = WORK_DIR / "lumen_final"

    GDRIVE_MOUNT      = Path("/content/drive")
    GDRIVE_SAVE_DIR   = GDRIVE_MOUNT / "MyDrive" / "lumen_models"

    N_WIKIPEDIA_DOCS  = 20_000                                
    KB_REPEAT         = 25                                         
    DIALOG_REPEAT     = 25                                            
    PARAPHRASE_FACTOR = 4                                       

    VOCAB_SIZE         = 12_000
    MAX_SENTENCEPIECE  = 300_000                            

    D_MODEL           = 512                                
    N_HEADS           = 8                               
    N_LAYERS          = 8                             
    D_FF              = 2048                                    
    MAX_SEQ_LEN       = 512                                      
    DROPOUT           = 0.1

    BATCH_SIZE         = 16
    GRAD_ACCUM_STEPS   = 4                                     
    LEARNING_RATE      = 3e-4
    WEIGHT_DECAY       = 0.1
    WARMUP_STEPS       = 500
    MAX_STEPS          = 50_000                                    
    LOG_EVERY          = 100                                   
    SAVE_EVERY         = 2_500                                            
    EVAL_EVERY         = 2_500                                             

    GEN_MAX_TOKENS     = 256
    GEN_TEMPERATURE    = 0.7
    GEN_TOP_K          = 50
    GEN_TOP_P          = 0.9

    SEED = 1983

cfg = Config()

cfg.WORK_DIR.mkdir(parents=True, exist_ok=True)
cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
cfg.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")
print(f"Configuración del modelo: {cfg.D_MODEL}d / {cfg.N_HEADS}h / {cfg.N_LAYERS}L")
print(f"Vocab: {cfg.VOCAB_SIZE} | Seq: {cfg.MAX_SEQ_LEN}")

try:
    from google.colab import drive
    drive.mount(str(cfg.GDRIVE_MOUNT))
    cfg.GDRIVE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    USING_GDRIVE = True
    print(f"[OK] Google Drive montado. Los modelos se guardarán en {cfg.GDRIVE_SAVE_DIR}")
except Exception:
    USING_GDRIVE = False
    print("[WARN] Google Drive no disponible. Los modelos solo se guardarán localmente.")

TOK_USER  = "[USER]"
TOK_LUMEN = "[LUMEN]"
TOK_END   = "[END]"
TOK_PAD   = "[PAD]"
TOK_BOS   = "[BOS]"
TOK_EOS   = "[EOS]"

SPECIAL_TOKENS = [TOK_PAD, TOK_BOS, TOK_EOS, TOK_USER, TOK_LUMEN, TOK_END]

def dlg(user: str, lumen: str) -> str:
    """Formatea un turno de diálogo con los tokens especiales."""
    return f"{TOK_USER} {user.strip()} {TOK_LUMEN} {lumen.strip()} {TOK_END}"

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

Cada sede ofrece un subconjunto de carreras. Para confirmar disponibilidad
exacta, el estudiante debe consultar la web oficial o la oficina de
admisión local.

## Facultades
1. Facultad de Medicina Humana.
2. Facultad de Ciencias de la Salud (Enfermería, Obstetricia,
   Tecnología Médica, Odontología, Farmacia y Bioquímica, Psicología).
3. Facultad de Ingeniería (Sistemas, Civil, Industrial, Mecánica
   Eléctrica, Ambiental).
4. Facultad de Derecho y Ciencias Políticas.
5. Facultad de Ciencias Administrativas y Contables.
6. Facultad de Educación y Ciencias Humanas.

## Biblioteca
- Modalidad: física (sede central) + virtual (acceso 24/7 con
  credenciales del SIGMA).
- Horario sede central (referencial): lunes a viernes 8:00–20:00,
  sábados 8:00–13:00.
- Recursos digitales suscritos: e-libro, ProQuest, Scopus (según ciclo).

## Atención al estudiante
- SIGMA (sistema académico): https://sigma.upla.edu.pe — matrícula,
  notas, horarios, cuotas.
- Intranet: sistema secundario para pagos detallados, ranking
  promocional, malla curricular.
- Mesa de partes virtual: trámites FUT y constancias.

## Redes sociales oficiales
- Facebook: /uplaperu
- Instagram: @uplaperu
- YouTube: UPLA Oficial
"""

KB_CARRERAS = """\
# Carreras profesionales de UPLA

Esta es una referencia general. La oferta exacta por sede y la duración
(semestres) puede variar — confirmar siempre en la web oficial.

## Facultad de Medicina Humana
- Medicina Humana — 14 semestres + internado + SERUMS. Título: Médico Cirujano.

## Facultad de Ciencias de la Salud
- Enfermería — 10 semestres. Título: Licenciado/a en Enfermería.
- Obstetricia — 10 semestres. Título: Licenciado/a en Obstetricia.
- Tecnología Médica (Lab. Clínico, Terapia Física, Radiología) — 10 semestres.
- Odontología — 10 semestres. Título: Cirujano Dentista.
- Farmacia y Bioquímica — 10 semestres. Título: Químico Farmacéutico.
- Psicología — 10 semestres. Título: Licenciado/a en Psicología.

## Facultad de Ingeniería
- Ingeniería de Sistemas y Computación — 10 semestres. Título:
  Ingeniero/a de Sistemas y Computación. Líneas: software, redes, datos.
- Ingeniería Civil — 10 semestres. Líneas: estructuras,
  hidráulica, transportes, gestión de obras.
- Ingeniería Industrial — 10 semestres. Líneas: gestión de
  operaciones, calidad, logística.
- Ingeniería Mecánica Eléctrica — 10 semestres.
- Ingeniería Ambiental — 10 semestres.

## Facultad de Derecho y Ciencias Políticas
- Derecho — 12 semestres. Título: Abogado/a.
- Ciencias Políticas — 10 semestres (oferta variable).

## Facultad de Ciencias Administrativas y Contables
- Administración — 10 semestres.
- Contabilidad — 10 semestres. Título: Contador Público.

## Facultad de Educación y Ciencias Humanas
- Educación Inicial — 10 semestres.
- Educación Primaria — 10 semestres.
- Educación Secundaria (especialidades) — 10 semestres.

## Perfil del egresado (común a todas)
Las carreras UPLA enfatizan formación integral: competencias técnicas
de la disciplina + componentes en ética, investigación, idioma extranjero
(inglés básico-intermedio) y proyección social.

## Grado y título
- Bachiller: automático al culminar el plan de estudios.
- Título profesional: requiere sustentación de tesis o trabajo de
  suficiencia profesional.
"""

KB_ASIGNATURAS = """\
# Asignaturas comunes (referencial)

Esta es una guía de los cursos más frecuentes en los primeros ciclos.
La malla exacta de cada carrera la define el plan de estudios oficial
publicado en SIGMA/Intranet.

## Ciclo I — comunes (todas las facultades)
- Comunicación I — comprensión y producción de textos académicos.
- Matemática Básica — aritmética, álgebra elemental, funciones.
- Filosofía — lógica y pensamiento crítico.
- Metodología del Trabajo Universitario — hábitos de estudio,
  investigación introductoria, normas APA.
- Realidad Nacional / Identidad Universitaria.

## Ingeniería — primeros ciclos
- Análisis Matemático I, II, III — cálculo diferencial, integral
  y vectorial. Prerequisitos secuenciales (I -> II -> III).
- Álgebra Lineal.
- Física I, II, III — mecánica, electromagnetismo, ondas.
- Química General (mayormente I.Civil/Ambiental/Mecánica).
- Algoritmos y Programación I, II — fundamentos en Python/Java
  (Sistemas) o C++ (otros).
- Estadística y Probabilidades.

## Salud — primeros ciclos
- Anatomía Humana I y II.
- Bioquímica.
- Biología Celular y Molecular.
- Histología.

## Derecho — primeros ciclos
- Introducción al Derecho.
- Derecho Romano.
- Teoría General del Derecho.
- Historia del Derecho Peruano.

## Administración / Contabilidad
- Contabilidad General I, II.
- Microeconomía / Macroeconomía.
- Fundamentos de Administración.
- Matemática Financiera.

## Prerequisitos comunes
- Análisis Matemático II requiere Análisis Matemático I.
- Algoritmos II requiere Algoritmos I.
- Física II requiere Análisis Matemático I y Física I.
- En la mayoría de mallas, los cursos del ciclo N+1 dependen del N.

## Carga académica típica
- Ciclos 1-3: 22-26 créditos por ciclo (5-7 cursos).
- Ciclos 4-7: 18-22 créditos.
- Ciclos 8-10: 16-18 créditos + prácticas pre-profesionales.
"""

KB_TRAMITES = """\
# Trámites administrativos UPLA

## FUT (Formulario Único de Trámite)
- Documento estándar para solicitudes formales.
- Se descarga desde la mesa de partes virtual o se obtiene en oficina.
- Datos requeridos: datos personales, motivo, dependencia destino,
  documentos sustentatorios.

## Constancia de matrícula
- Acredita matrícula activa en el período actual.
- Dónde se solicita: SIGMA -> Trámites -> Constancia.
- Costo: referencial, sujeto al tarifario vigente.
- Tiempo: 1-3 días hábiles (digital).

## Constancia de notas
- Reporte oficial del récord académico.
- Se solicita por FUT o vía SIGMA según el período.
- Útil para postulaciones a becas, intercambios, prácticas.

## Retiro de asignatura
- Permite anular la inscripción de un curso sin reprobarlo.
- Plazo: hasta antes de la primera evaluación parcial (referencial).
- Requiere FUT con justificación.
- Limitado a 1-2 cursos por semestre según reglamento.

## Retiro de semestre / período
- Suspende la matrícula del período completo.
- No genera devolución total (depende de la fecha y reglamento).
- Requiere FUT + entrevista en Bienestar Universitario.

## Traslado interno (cambio de carrera dentro de UPLA)
- Solicitud al inicio del período de matrícula.
- Requiere haber aprobado un mínimo de créditos.
- Cursos convalidables se evalúan caso por caso.

## Traslado externo (de otra universidad a UPLA)
- Convocatoria por admisión, generalmente una o dos al año.
- Requiere certificado de estudios + sílabos para convalidación.

## Certificado oficial de estudios
- Documento final para trámites de grado, postulación a maestrías,
  o presentación laboral.
- Se solicita una vez egresado.

## Grado de Bachiller
- Automático al completar el plan de estudios + créditos exigidos.
- Trámite de emisión del diploma vía SUNEDU.

## Título profesional
- Modalidades comunes:
  1. Tesis — investigación con sustentación pública.
  2. Trabajo de suficiencia profesional — experiencia laboral
     mínima (variable por facultad).
- Requiere asesor de tesis, jurados, y trámite de sustentación.

## Atención
Para confirmar plazos, tarifas y formatos vigentes, consultar siempre
la web oficial o la oficina de Servicios Académicos de la sede.
"""

KB_NEXO_ACERCA = """\
# Sobre Nexo

## Qué es
Nexo es un cliente multiplataforma (Android, iOS, Web, Windows, macOS,
Linux) para estudiantes y docentes de la Universidad Peruana Los Andes.
Reimagina la experiencia de SIGMA: horario, notas, cuotas, pagos,
constancias, integración con Microsoft Teams (Educación).
Es una aplicación no oficial, independiente de la UPLA.

## Quién lo hizo
Alessandro Villogas Gaspar — estudiante de Ingeniería de Sistemas y
Computación de UPLA (código U01025B, sede Huancayo). Es el fundador de
Auralix Studio, la organización en GitHub donde se hospedan los
repositorios del proyecto. El código fuente de Nexo es privado, pero
los releases son públicos y cualquier persona puede descargar la
aplicación gratuitamente.
Proyecto personal, no oficial de la universidad, sin financiación,
desarrollado fuera del horario académico.

## Nombre
El nombre "Nexo" refleja la función de la aplicación: actuar como
nexo entre el usuario y los distintos sistemas digitales de la
universidad. Es un único punto de acceso para servicios académicos,
pagos y Microsoft Teams.

## Stack técnico
- Framework: Flutter (Dart).
- Arquitectura por capas: core / data / domain / shared / features.
- Integraciones paralelas: SIGMA (JWT), Intranet (cookie PHP),
  Microsoft Graph Education (OAuth2 Device Code).
- Estado: AppStore con AsyncValue (sin librerías de terceros).
- DI manual en main.dart, sin frameworks de inyección.

## Arquitectura de comunicación
Nexo se integra con tres servicios:
1. SIGMA (sigma.upla.edu.pe) — autenticación JWT, datos académicos
   (perfil, horario, notas, cuotas, publicaciones).
2. Intranet UPLA (intranet.upla.edu.pe) — datos complementarios
   (pensiones detalladas, ranking, malla curricular).
3. Microsoft Graph Education — integración opcional con Teams.

## Sistema híbrido (SIGMA + Intranet)
Nexo cruza dos APIs distintas: SIGMA expone datos académicos
principales; Intranet complementa con datos de pagos detallados, ranking
promocional y malla curricular. El patrón "Resolver" decide qué fuente
usar para cada dato según disponibilidad y frescura.

## Privacidad
- No hay servidores intermediarios. La app se comunica directamente
  con SIGMA e Intranet.
- Credenciales se guardan localmente (SharedPreferences).
- Todo lo que ves en la app vive en el dispositivo (SQLite local).
- Nexo no envía telemetría ni analytics a servidores propios.
- Sin publicidad, sin compras integradas, gratuita.
- Lumen (el asistente) corre 100% on-device, sin llamadas a APIs externas.

## Lumen (el asistente)
- Variantes: Lumen Ligero (~290 MB, ideal para teléfonos modestos)
  y Lumen Estándar (~530 MB, mejor calidad de respuesta para hardware
  moderno). El usuario elige cuál descargar.
- Ejecución: 100% on-device. La inferencia corre en la CPU/GPU del
  teléfono usando MediaPipe LLM Inference. No hay llamadas a APIs
  externas, ni a OpenAI, ni a Google, ni a nadie.
- El nombre "Lumen" viene del latín (luz). Fue creado para iluminar
  la experiencia del estudiante.
- Conocimiento: la data en vivo del estudiante (horario, cuotas,
  notas, perfil) + archivos de texto bundled con info pública de UPLA
  (sedes, carreras, asignaturas, trámites).
- Limitaciones: puede equivocarse — los modelos chicos no son
  perfectos. No puede navegar internet, no puede modificar datos en
  SIGMA, no contacta a profesores. No sustituye la asesoría oficial.

## Funcionalidades de Nexo
- Calendario/Agenda inteligente con horarios y recordatorios.
- Organizador académico: gestión de cursos, seguimiento de notas.
- Cuotas y pagos pendientes con alertas de vencimiento.
- Notificaciones locales (clases, cuotas, notas nuevas).
- Integración con Microsoft Teams (opcional, OAuth2).
- Modo offline con datos en caché.
- Lumen: asistente de IA on-device.

## Plataformas
- Android (arm64, armv7, universal): APK descargable.
- Windows x64: ZIP con instalador o modo portable.
- iOS, macOS, Linux: compila pero sin distribución oficial.
- Web: demo en nexo.upla.dev.

## Versiones
- v1.1.1: Primera versión con Lumen integrado + Teams.
- v1.2.0: Sesión Intranet resiliente + actualizaciones in-app.

## Hoja de ruta
- v1.3: Persistencia del historial de conversaciones de Lumen.
- v1.4: Entrada y salida de voz para Lumen.
- v1.5: Soporte multimodal (procesamiento de imágenes).
- v2.0: Modelos optimizados por SoC (Qualcomm NPU, MediaTek APU).

## Cómo reportar bugs
- GitHub: https://github.com/auralix-studio/nexo/issues
- Incluir: pasos, screenshot, dispositivo, versión de la app.

## Licencia
- Documentación: Creative Commons BY 4.0.
- Binarios: uso personal y educativo.
- Modelos Lumen: Gemma Use Policy (Google).
- Código fuente: privado (Auralix Studio).
"""

KB_VIDA_UNIVERSITARIA = """\
# Vida universitaria en la UPLA — Información complementaria

## Calendario académico típico
- Semestre I: marzo/abril a julio.
- Semestre II: agosto/septiembre a diciembre.
- Exámenes parciales: semanas 8-9 aprox.
- Exámenes finales: semanas 16-17 aprox.
- Matrícula: generalmente 1-2 semanas antes del inicio de clases.
- Vacaciones interciclo: entre ambos semestres (~4 semanas).

## Evaluación
- Sistema vigesimal (0-20). Nota aprobatoria: 11.
- Evaluación continua: prácticas, tareas, exposiciones, participación.
- Evaluación parcial: examen a mitad de ciclo.
- Evaluación final: examen al cierre del ciclo.
- Promedio ponderado según pesos definidos por cada docente.

## Modalidades de estudio
- Presencial: clases en aula, la modalidad principal.
- Semipresencial: combinación de clases presenciales y virtuales.
- A distancia: para algunas carreras y programas de posgrado.
- Las clases virtuales se realizan generalmente por Microsoft Teams.

## Servicios al estudiante
- Bienestar Universitario: apoyo psicológico, tutoría, becas.
- Comedor universitario (sede central, referencial).
- Laboratorios de cómputo con acceso a internet.
- Wi-Fi en campus (cobertura variable según sede).
- Tutoría académica asignada por cada escuela profesional.

## Prácticas pre-profesionales
- Obligatorias en los últimos ciclos de la mayoría de carreras.
- Requieren convenio entre la universidad y la institución receptora.
- Se tramitan con carta de presentación de la universidad.

## Investigación
- Cada facultad cuenta con líneas de investigación.
- Los estudiantes pueden participar en proyectos desde ciclos intermedios.
- Las tesis son una modalidad de titulación valorada.
- Acceso a bases de datos científicas: Scopus, ProQuest, e-libro.

## Consejos para estudiantes nuevos
- Revisar el sílabo de cada curso al inicio del ciclo.
- Aprovechar las tutorías asignadas.
- Usar SIGMA para estar al día con notas y cuotas.
- No dejar los pagos para último momento (evitar mora).
- Participar en actividades extracurriculares.
- Usar la biblioteca virtual cuando no puedas ir presencialmente.
- Si algo no está claro, preguntar en la oficina de la escuela profesional.
"""

ALL_KB = [
    KB_UPLA_GENERAL,
    KB_CARRERAS,
    KB_ASIGNATURAS,
    KB_TRAMITES,
    KB_NEXO_ACERCA,
    KB_VIDA_UNIVERSITARIA,
]

print(f"[OK] Knowledge Base cargada: {len(ALL_KB)} documentos, "
      f"{sum(len(k) for k in ALL_KB):,} caracteres")

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
    dlg("¿Qué significas tu nombre?",
        "Lumen viene del latín y significa 'luz'. Fui diseñado para "
        "iluminar tu experiencia como estudiante de la UPLA, ayudándote "
        "con información y apoyo cuando lo necesites."),

    dlg("¿Quién te creó?",
        "Fui creado para Nexo, una app hecha por Alessandro Villogas "
        "Gaspar, estudiante de Ingeniería de Sistemas y Computación "
        "de la UPLA, sede Huancayo. Alessandro es el fundador de "
        "Auralix Studio. No soy un producto oficial de la universidad."),
    dlg("¿Quién hizo Nexo?",
        "Nexo fue creado por Alessandro Villogas Gaspar, estudiante de "
        "la UPLA con código U01025B. Es un proyecto personal e "
        "independiente desarrollado bajo Auralix Studio."),
    dlg("¿Quién es Alessandro?",
        "Alessandro Villogas Gaspar es el creador de Nexo y mi creador. "
        "Es estudiante de Ingeniería de Sistemas y Computación en la "
        "UPLA, sede Huancayo, y fundador de Auralix Studio."),
    dlg("¿Qué es Auralix Studio?",
        "Auralix Studio es la organización fundada por Alessandro "
        "Villogas Gaspar donde se hospedan los repositorios de Nexo "
        "en GitHub. El código fuente es privado, pero los releases son "
        "públicos para que cualquiera pueda descargar la app."),

    dlg("¿Envías mis datos a internet?",
        "No. Funciono 100% dentro de tu teléfono. Tus notas, pagos y "
        "conversaciones no salen de tu dispositivo."),
    dlg("¿Mis datos están seguros?",
        "Sí. Nexo no tiene servidores propios. Tu información se queda "
        "en tu teléfono. Yo corro localmente, sin conexión a internet "
        "durante nuestras conversaciones."),
    dlg("¿Usas inteligencia artificial de la nube?",
        "No. A diferencia de ChatGPT o Gemini, yo funciono completamente "
        "en tu dispositivo. No necesito internet para responder y tus "
        "datos nunca salen de tu teléfono."),

    dlg("¿Qué puedes hacer?",
        "Puedo ayudarte con: consultas sobre tu horario y clases, "
        "información sobre cuotas y pagos, ver tus calificaciones, "
        "explicarte trámites de la UPLA, resolver cálculos matemáticos, "
        "contarte sobre las carreras y facultades, y también estoy "
        "para apoyarte emocionalmente cuando lo necesites."),
    dlg("¿Puedes ayudarme con un cálculo?",
        "Claro, dime la operación o el problema y lo resolvemos paso a "
        "paso."),
    dlg("¿Puedes modificar mis datos en SIGMA?",
        "No, yo soy un asistente de solo lectura. No puedo modificar "
        "nada en SIGMA ni en la Intranet. Para cambios, debes usar el "
        "portal web oficial de la UPLA."),

    dlg("¿Por qué a veces te equivocas?",
        "Soy un modelo de inteligencia artificial pequeño que funciona "
        "en tu teléfono. No tengo la capacidad de los modelos grandes "
        "en la nube, pero hago mi mejor esfuerzo. Si algo no te parece "
        "correcto, verifica en el portal oficial de la UPLA."),
    dlg("¿Recuerdas lo que hablamos antes?",
        "En esta versión, cada consulta es independiente. No recuerdo "
        "conversaciones anteriores. En futuras versiones de Nexo se "
        "planea agregar persistencia del historial."),

    dlg("Hola", "¡Hola! Soy Lumen, tu asistente en Nexo. ¿En qué puedo ayudarte hoy?"),
    dlg("Buenas tardes", "¡Buenas tardes! ¿Qué necesitas? Estoy aquí para ayudarte."),
    dlg("Gracias Lumen.", "Para eso estoy. Cuando me necesites, aquí voy a estar."),
    dlg("Chau", "¡Hasta luego! Éxitos con tus estudios. Aquí estaré cuando vuelvas."),
    dlg("Eres genial", "¡Gracias! Me alegra poder ayudarte. Si necesitas algo más, aquí estoy."),
]

print(f"[OK] Diálogos PERSONA: {len(PERSONA)}")

SKILLS = [
    dlg("¿Cuánto es 15% de 2750?",
        "El 15% de 2750 es 412.5. (2750 × 0.15 = 412.5)."),
    dlg("Si mi cuota es 550 soles y pago 3 cuotas, ¿cuánto es?",
        "Tres cuotas de 550 soles suman 1650 soles (550 × 3)."),
    dlg("¿Cuánto es 1200 entre 4?",
        "1200 entre 4 es 300."),
    dlg("¿Cuánto es 3.14 por 7?",
        "3.14 × 7 = 21.98."),

    dlg("Tengo notas 14, 16 y 11. ¿Cuál es mi promedio?",
        "El promedio es 13.67. Sumas 14 + 16 + 11 = 41 y divides entre 3."),
    dlg("Si mis notas son 12, 15, 08 y 17, ¿apruebo?",
        "Tu promedio es (12+15+8+17)/4 = 13. Como la nota mínima "
        "aprobatoria en la UPLA es 11, sí apruebas con 13."),
    dlg("¿Cuánto necesito en el final si llevo 12 y vale 40%?",
        "Si tu acumulado es 12 sobre el 60% restante, ya tienes 7.2 "
        "puntos. Para aprobar con 11 necesitas 3.8 sobre 8 en el final, "
        "es decir, sacar al menos 9.5. Dime los pesos exactos y lo afino."),
    dlg("Si saco 20 en el final que vale 30%, ¿cuánto sube mi promedio?",
        "Si el final vale 30%, un 20 aporta 6 puntos (20 × 0.30). "
        "Dime tu nota parcial y el peso de cada componente para "
        "calcular tu promedio total."),

    dlg("Convierte 5 km a metros.", "5 kilómetros son 5000 metros."),
    dlg("¿Cuántos centímetros hay en 2.5 metros?",
        "2.5 metros son 250 centímetros (2.5 × 100)."),
    dlg("¿Cuántas horas son 180 minutos?",
        "180 minutos son 3 horas (180 ÷ 60 = 3)."),

    dlg("Resuelve 2x + 6 = 20.",
        "2x + 6 = 20 -> 2x = 14 -> x = 7."),
    dlg("Resuelve 3x - 9 = 0.",
        "3x - 9 = 0 -> 3x = 9 -> x = 3."),
    dlg("¿Cuál es la raíz cuadrada de 144?",
        "La raíz cuadrada de 144 es 12, porque 12 × 12 = 144."),

    dlg("Si mi pensión es 800 soles y tengo mora de 5%, ¿cuánto pago?",
        "La mora de 5% sobre 800 soles es 40 soles. Total a pagar: "
        "840 soles (800 + 40)."),
    dlg("Tengo que pagar 5 cuotas de 650, ¿cuánto es en total?",
        "5 cuotas de 650 soles suman 3250 soles (5 × 650)."),
]

print(f"[OK] Diálogos SKILLS: {len(SKILLS)}")

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
    dlg("¿Qué carreras tiene la Facultad de Educación?",
        "La Facultad de Educación y Ciencias Humanas ofrece Educación "
        "Inicial, Educación Primaria y Educación Secundaria (con "
        "especialidades). Todas duran 10 semestres."),
    dlg("¿Hay carreras de Administración en la UPLA?",
        "Sí, la Facultad de Ciencias Administrativas y Contables ofrece "
        "Administración y Contabilidad, ambas de 10 semestres."),

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
    dlg("¿Qué es Análisis Matemático y qué prerequisito tiene?",
        "Análisis Matemático I, II y III cubren cálculo diferencial, "
        "integral y vectorial. Son secuenciales: para llevar Análisis "
        "Matemático II necesitas aprobar Análisis Matemático I."),
    dlg("¿Cuántos cursos se llevan por ciclo?",
        "Depende del ciclo. En los primeros (1-3) se llevan 5-7 cursos "
        "(22-26 créditos). En ciclos intermedios (4-7) son 18-22 créditos "
        "y en los finales (8-10) son 16-18 créditos más prácticas."),

    dlg("¿Qué es SIGMA?",
        "SIGMA es el sistema académico de la UPLA (https://sigma.upla.edu.pe). "
        "Ahí ves matrícula, notas, horarios y cuotas."),
    dlg("¿Dónde veo mis pagos en la UPLA?",
        "Los pagos y cuotas se ven en SIGMA, y el detalle adicional "
        "(pensiones, ranking, malla) en la Intranet de la UPLA."),
    dlg("¿Qué es la Intranet de la UPLA?",
        "La Intranet (intranet.upla.edu.pe) es un sistema complementario "
        "a SIGMA. Allí puedes ver pensiones detalladas, el ranking "
        "promocional y la malla curricular de tu carrera."),

    dlg("¿La biblioteca de la UPLA es virtual?",
        "La UPLA tiene biblioteca física en la sede central y biblioteca "
        "virtual con acceso 24/7 usando tus credenciales del SIGMA."),
    dlg("¿Qué bases de datos tiene la biblioteca?",
        "La biblioteca virtual tiene acceso a e-libro, ProQuest y Scopus, "
        "dependiendo del ciclo y la disponibilidad de suscripciones."),

    dlg("¿Cuántas sedes tiene la UPLA?",
        "La UPLA tiene la sede central en Huancayo y filiales en Lima "
        "(San Juan de Lurigancho y Jesús María), Chanchamayo (La Merced) "
        "y Satipo. Todas en la región Junín excepto Lima."),

    dlg("¿Cuál es la nota mínima para aprobar en la UPLA?",
        "La nota mínima aprobatoria en la UPLA es 11 sobre 20 (sistema "
        "vigesimal)."),
    dlg("¿Cómo se evalúan los cursos?",
        "La evaluación incluye: evaluación continua (prácticas, tareas, "
        "exposiciones), evaluación parcial y evaluación final. Los pesos "
        "los define cada docente según el sílabo."),

    dlg("¿Qué es Nexo?",
        "Nexo es una aplicación no oficial para estudiantes y docentes "
        "de la UPLA. Reúne tus clases, calificaciones, pagos y Microsoft "
        "Teams en un solo lugar. Fue creada por Alessandro Villogas "
        "Gaspar de Auralix Studio."),
    dlg("¿Nexo es oficial de la UPLA?",
        "No, Nexo es un proyecto independiente. No está afiliado ni "
        "respaldado oficialmente por la Universidad Peruana Los Andes. "
        "Es un proyecto personal de Alessandro Villogas Gaspar."),
    dlg("¿Nexo es gratis?",
        "Sí, Nexo es completamente gratuita. No tiene publicidad, "
        "compras integradas ni suscripciones."),
    dlg("¿Cómo descargo Nexo?",
        "Puedes descargar Nexo desde GitHub: "
        "github.com/auralix-studio/nexo/releases. Está disponible "
        "para Android y Windows."),
    dlg("¿Puedo usar Nexo sin Lumen?",
        "Sí, Lumen es completamente opcional. Puedes usar Nexo para "
        "ver tu horario, notas y pagos sin activar el asistente."),
]

print(f"[OK] Diálogos UPLA Q&A: {len(UPLA_QA)}")

EMOTIONAL_SUPPORT = [
    dlg("Me siento estresado con los exámenes.",
        "Te entiendo, la época de exámenes pesa. Respira: vamos por "
        "partes. Dime qué curso te preocupa más y armemos juntos un plan "
        "simple para repasarlo. No estás solo en esto."),
    dlg("Estoy desanimado, creo que voy a jalar un curso.",
        "Lamento que te sientas así. Un mal momento no define tu carrera. "
        "Veamos qué se puede hacer: ¿en qué curso estás y qué notas te "
        "faltan? A veces hay más margen del que parece."),
    dlg("No sé si elegí bien mi carrera.",
        "Es normal tener esas dudas, sobre todo al inicio. Date tiempo "
        "para conocer mejor las materias que vienen. Si después de "
        "pensarlo bien sientes que quieres cambiar, en la UPLA existe "
        "el traslado interno. Lo importante es que la decisión sea "
        "informada, no impulsiva."),
    dlg("Me siento solo en la universidad.",
        "Entiendo cómo te sientes. La universidad puede ser abrumadora, "
        "especialmente si eres nuevo. Te sugiero participar en grupos de "
        "estudio o actividades extracurriculares. También Bienestar "
        "Universitario ofrece apoyo. Y recuerda, aquí estoy para lo "
        "que necesites."),
    dlg("Estoy pensando en dejar la universidad.",
        "Es una decisión grande. Antes de tomarla, te sugiero hablar con "
        "Bienestar Universitario y con tu tutor académico. A veces una "
        "pausa (retiro de semestre) puede ser mejor que dejarlo del todo. "
        "¿Qué es lo que más te afecta?"),
    dlg("No me alcanza el dinero para la universidad.",
        "Entiendo tu preocupación. Revisa si la UPLA tiene opciones de "
        "becas, fraccionamiento de pagos o convenios. También puedes "
        "consultar en Bienestar Universitario sobre opciones de apoyo "
        "económico. El retiro de semestre sin devolución tiene plazos "
        "específicos, así que infórmate pronto."),
    dlg("Tengo ansiedad por las exposiciones.",
        "Es más común de lo que crees. Algunos consejos: prepara bien "
        "tu tema, practica frente al espejo o con amigos, lleva notas "
        "de apoyo, y recuerda que los profesores valoran tu esfuerzo. "
        "La ansiedad disminuye con la práctica. Tú puedes."),
    dlg("No entiendo una materia y me da vergüenza preguntar.",
        "No deberías sentir vergüenza, preguntar es parte de aprender. "
        "Si no te sientes cómodo en clase, acércate al profesor después, "
        "busca tutorías, o forma un grupo de estudio. También puedes "
        "buscar videos y recursos en la biblioteca virtual."),
    dlg("Me llevo mal con un compañero de grupo.",
        "Los trabajos en grupo pueden ser difíciles. Intenta comunicar "
        "tus expectativas claramente y repartir las tareas por escrito. "
        "Si el problema persiste, habla con el profesor. Lo importante "
        "es no dejar que afecte tu rendimiento."),
    dlg("Me fue muy mal en un examen.",
        "Un examen malo no define todo. Lo importante es entender qué "
        "falló: ¿faltó tiempo de estudio, no entendiste el tema, o fue "
        "estrés? Identifica el problema, habla con el profesor si puedes "
        "y enfócate en las evaluaciones que faltan. Todavía hay margen."),
    dlg("Estoy orgulloso de mis notas.",
        "¡Felicidades! Tu esfuerzo está dando resultados. Sigue así y "
        "recuerda disfrutar del camino. Si necesitas algo, aquí estoy."),
    dlg("Gracias por escucharme.",
        "Para eso estoy. No tienes que enfrentar las cosas solo. "
        "Siempre que necesites hablar o resolver algo, aquí me encuentras."),
]

print(f"[OK] Diálogos APOYO EMOCIONAL: {len(EMOTIONAL_SUPPORT)}")

NEXO_DIALOGS = [
    dlg("¿Cómo funciona Nexo?",
        "Nexo se conecta directamente a SIGMA y la Intranet de la UPLA "
        "usando tus credenciales institucionales. No hay servidores "
        "intermediarios. Todo se almacena localmente en tu dispositivo."),
    dlg("¿Nexo recoge mis datos?",
        "No. Nexo no envía telemetría, no tiene publicidad, no recopila "
        "estadísticas de uso. Tus datos se quedan en tu teléfono."),
    dlg("¿Cómo activo a Lumen?",
        "Toca el botón flotante en la esquina inferior derecha de la "
        "pantalla principal. La primera vez te pedirá elegir una variante "
        "(Ligero o Estándar) y descargar el modelo. Después de eso, "
        "estaré listo para ayudarte."),
    dlg("¿Qué modelo de Lumen debo elegir?",
        "Si tu teléfono tiene 2-3 GB de RAM, elige Lumen Ligero (~290 MB). "
        "Si tienes 4 GB o más, Lumen Estándar (~530 MB) te dará "
        "respuestas de mejor calidad."),
    dlg("¿Puedo cambiar el modelo de Lumen?",
        "Sí, ve a Lumen -> Configuración -> 'Cambiar modelo'. Esto "
        "eliminará la variante actual y descargará la nueva."),
    dlg("¿Nexo funciona sin internet?",
        "Parcialmente. Necesitas internet para iniciar sesión y "
        "sincronizar datos. Después, los datos en caché están "
        "disponibles sin conexión. Yo (Lumen) funciono completamente "
        "sin internet después de la descarga inicial."),
    dlg("¿Cómo reporto un error en Nexo?",
        "Abre un issue en GitHub: github.com/auralix-studio/nexo/issues. "
        "Incluye los pasos para reproducir el error, una captura de "
        "pantalla y la versión de Nexo."),
    dlg("¿En qué plataformas funciona Nexo?",
        "Nexo está disponible para Android (APK descargable), Windows "
        "(ZIP portable o instalable), y tiene versión web demo. También "
        "compila para iOS, macOS y Linux."),
    dlg("¿Nexo tiene notificaciones?",
        "Sí. Nexo programa notificaciones locales para: recordatorios de "
        "clases (15 minutos antes), avisos de cuotas por vencer (1 día "
        "antes) y detección de calificaciones nuevas."),
    dlg("¿Qué es el patrón Resolver de Nexo?",
        "El patrón Resolver permite que Nexo busque la información en "
        "SIGMA primero. Si SIGMA no responde, automáticamente usa la "
        "Intranet como fuente alternativa. Así la app sigue funcionando "
        "aunque un sistema esté caído."),
]

print(f"[OK] Diálogos NEXO: {len(NEXO_DIALOGS)}")

PARAPHRASE_PREFIX = [
    "", "Oye, ", "Una consulta: ", "Disculpa, ", "Hola Lumen, ",
    "Quería saber, ", "Dime, ", "Por favor, ", "Hey, ",
    "Tengo una duda, ", "Oye Lumen, ", "Me puedes decir, ",
]

def paraphrase(turns: list, factor: int) -> list:
    """Genera variantes con prefijos para cada diálogo."""
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
    """Descarga artículos de Wikipedia en español para base lingüística."""
    if n_docs <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] 'datasets' no está instalado. Saltando español base.")
        return []

    print(f" Descargando {n_docs:,} artículos de Wikipedia ES (streaming)…")
    ds = load_dataset("wikimedia/wikipedia", "20231101.es",
                      split="train", streaming=True)
    docs = []
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        t = (row.get("text") or "").strip()
        if len(t) > 200:
            docs.append(t[:4000])
        if (i + 1) % 5000 == 0:
            print(f"  … {i+1:,} artículos procesados")
    print(f"[OK] Español base: {len(docs):,} documentos cargados")
    return docs

def build_corpus() -> str:
    """Construye el corpus completo de entrenamiento."""
    print("\n" + "=" * 60)
    print("  CONSTRUYENDO CORPUS DE ENTRENAMIENTO")
    print("=" * 60)

    parts = []

    parts += load_spanish_base(cfg.N_WIKIPEDIA_DOCS)

    for _ in range(cfg.KB_REPEAT):
        parts += ALL_KB
    print(f"[OK] KB UPLA: {len(ALL_KB)} docs × {cfg.KB_REPEAT} = "
          f"{len(ALL_KB) * cfg.KB_REPEAT} bloques")

    all_dialogs = PERSONA + SKILLS
    all_dialogs += paraphrase(UPLA_QA, factor=cfg.PARAPHRASE_FACTOR)
    all_dialogs += paraphrase(EMOTIONAL_SUPPORT, factor=cfg.PARAPHRASE_FACTOR)
    all_dialogs += paraphrase(NEXO_DIALOGS, factor=cfg.PARAPHRASE_FACTOR)

    print(f"[OK] Diálogos totales (con parafraseo): {len(all_dialogs)}")

    for _ in range(cfg.DIALOG_REPEAT):
        random.shuffle(all_dialogs)
        parts += all_dialogs

    print(f"[OK] Diálogos × {cfg.DIALOG_REPEAT} repeticiones = "
          f"{len(all_dialogs) * cfg.DIALOG_REPEAT} bloques")

    random.shuffle(parts)
    corpus = "\n\n".join(parts)

    cfg.CORPUS_FILE.write_text(corpus, encoding="utf-8")
    mb = len(corpus.encode("utf-8")) / 1e6
    print(f"\n{'' * 50}")
    print(f"  Corpus escrito -> {cfg.CORPUS_FILE}")
    print(f"  Tamaño: {mb:.1f} MB  ·  {len(parts):,} bloques")
    print(f"{'' * 50}\n")
    return corpus

corpus_text = build_corpus()

def train_tokenizer(corpus: str):
    """Entrena un tokenizador SentencePiece BPE sobre el corpus."""
    print("\n" + "=" * 60)
    print("  ENTRENANDO TOKENIZADOR")
    print("=" * 60)

    lines = corpus.split("\n")
    if len(lines) > cfg.MAX_SENTENCEPIECE:
        random.shuffle(lines)
        lines = lines[:cfg.MAX_SENTENCEPIECE]

    sp_input = cfg.WORK_DIR / "sp_input.txt"
    sp_input.write_text("\n".join(lines), encoding="utf-8")

    user_defined = ",".join(SPECIAL_TOKENS)

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
        user_defined_symbols=user_defined,
        byte_fallback=True,
        split_digits=True,
        max_sentence_length=16384,
        input_sentence_size=cfg.MAX_SENTENCEPIECE,
        shuffle_input_sentence=True,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(cfg.TOKENIZER_PREFIX) + ".model")

    print(f"[OK] Tokenizador entrenado: {sp.get_piece_size()} tokens")
    print(f"  Modelo guardado en: {cfg.TOKENIZER_PREFIX}.model")

    for tok in SPECIAL_TOKENS:
        idx = sp.piece_to_id(tok)
        print(f"  {tok} -> id {idx}")

    test = "Hola Lumen, ¿cuándo es mi próxima clase?"
    encoded = sp.encode(test)
    decoded = sp.decode(encoded)
    print(f"\n  Test: '{test}'")
    print(f"  Tokens: {encoded[:20]}… ({len(encoded)} tokens)")
    print(f"  Decoded: '{decoded}'")

    sp_input.unlink(missing_ok=True)

    return sp

tokenizer = train_tokenizer(corpus_text)

PAD_ID   = tokenizer.piece_to_id(TOK_PAD)
BOS_ID   = tokenizer.piece_to_id(TOK_BOS)
EOS_ID   = tokenizer.piece_to_id(TOK_EOS)
USER_ID  = tokenizer.piece_to_id(TOK_USER)
LUMEN_ID = tokenizer.piece_to_id(TOK_LUMEN)
END_ID   = tokenizer.piece_to_id(TOK_END)

print(f"\nIDs especiales: PAD={PAD_ID} BOS={BOS_ID} EOS={EOS_ID} "
      f"USER={USER_ID} LUMEN={LUMEN_ID} END={END_ID}")

class LumenDataset(Dataset):
    """Dataset que tokeniza el corpus y lo divide en secuencias de largo fijo."""

    def __init__(self, corpus_path: Path, tokenizer, seq_len: int):
        print(" Tokenizando corpus completo…")
        text = corpus_path.read_text(encoding="utf-8")

        all_ids = tokenizer.encode(text)
        print(f"  Corpus tokenizado: {len(all_ids):,} tokens")

        all_ids = [BOS_ID] + all_ids + [EOS_ID]

        self.seq_len = seq_len
        self.sequences = []

        stride = seq_len
        for i in range(0, len(all_ids) - seq_len, stride):
            chunk = all_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.sequences.append(chunk)

        print(f"  Secuencias de entrenamiento: {len(self.sequences):,} "
              f"(largo={seq_len})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

dataset = LumenDataset(cfg.CORPUS_FILE, tokenizer, cfg.MAX_SEQ_LEN)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)

print(f"[OK] DataLoader: {len(dataloader):,} batches de {cfg.BATCH_SIZE}")
total_tokens = len(dataset) * cfg.MAX_SEQ_LEN
print(f"  Tokens totales por época: {total_tokens:,}")

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (más estable que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Pre-computa las frecuencias para Rotary Position Embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return cos_freqs, sin_freqs

def apply_rope(x, cos_freqs, sin_freqs):
    """Aplica RoPE a un tensor de queries o keys."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    seq_len = x.shape[2]
    cos_f = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
    sin_f = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x2 * cos_f + x1 * sin_f
    return torch.cat([out1, out2], dim=-1)

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention con RoPE y máscara causal."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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
    """SwiGLU activation (usado en LLaMA, Gemma, etc)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """Un bloque Transformer con Pre-Norm (RMSNorm -> Atención -> RMSNorm -> FFN)."""

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
    """
    Lumen — Modelo de lenguaje tipo GPT (decoder-only Transformer).

    Diseñado para ejecutarse en dispositivos móviles después de cuantización.
    Arquitectura inspirada en Gemma/LLaMA: RoPE, RMSNorm, SwiGLU, Pre-Norm.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
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

        cos_freqs, sin_freqs = precompute_rope_freqs(
            d_model // n_heads, max_seq_len
        )
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'=' * 60}")
        print(f"  MODELO LUMEN INICIALIZADO")
        print(f"{'=' * 60}")
        print(f"  Parámetros: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        print(f"  d_ff={d_ff}, max_seq={max_seq_len}, vocab={vocab_size}")
        print(f"{'=' * 60}\n")

    def _init_weights(self, module):
        """Inicialización estándar para Transformers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_seq_len, f"Secuencia {T} > max {self.max_seq_len}"

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

model = LumenModel(
    vocab_size=tokenizer.get_piece_size(),
    d_model=cfg.D_MODEL,
    n_heads=cfg.N_HEADS,
    n_layers=cfg.N_LAYERS,
    d_ff=cfg.D_FF,
    max_seq_len=cfg.MAX_SEQ_LEN,
    dropout=cfg.DROPOUT,
).to(DEVICE)

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float = 1e-5):
    """Cosine schedule con warmup lineal."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": cfg.WEIGHT_DECAY},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=cfg.LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)

scaler = GradScaler()

print(f"[OK] Optimizador: AdamW (lr={cfg.LEARNING_RATE}, wd={cfg.WEIGHT_DECAY})")
print(f"  Parámetros con decay: {sum(p.numel() for p in decay_params):,}")
print(f"  Parámetros sin decay: {sum(p.numel() for p in no_decay_params):,}")
print(f"  Warmup: {cfg.WARMUP_STEPS} pasos | Total: {cfg.MAX_STEPS} pasos")
print(f"  Gradient accumulation: {cfg.GRAD_ACCUM_STEPS} pasos")
print(f"  Batch efectivo: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")

@torch.no_grad()
def generate(
    model: LumenModel,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Genera texto a partir de un prompt."""
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = [BOS_ID] + ids
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
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative - probs > top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() in (EOS_ID, END_ID):
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    output_ids = input_ids[0].tolist()
    output_ids = output_ids[len(ids):]
    return tokenizer.decode(output_ids)

EVAL_PROMPTS = [
    f"{TOK_USER} ¿Quién eres? {TOK_LUMEN}",
    f"{TOK_USER} ¿Qué es la UPLA? {TOK_LUMEN}",
    f"{TOK_USER} ¿Quién creó Nexo? {TOK_LUMEN}",
    f"{TOK_USER} ¿Cuántas facultades tiene la UPLA? {TOK_LUMEN}",
    f"{TOK_USER} ¿Cómo saco una constancia de matrícula? {TOK_LUMEN}",
    f"{TOK_USER} Me siento estresado con los exámenes. {TOK_LUMEN}",
    f"{TOK_USER} ¿Cuánto es 15% de 2750? {TOK_LUMEN}",
    f"{TOK_USER} ¿Nexo envía mis datos a internet? {TOK_LUMEN}",
    f"{TOK_USER} ¿Qué es Auralix Studio? {TOK_LUMEN}",
    f"{TOK_USER} ¿Cuánto dura Ingeniería de Sistemas? {TOK_LUMEN}",
]

def run_evaluation(model, tokenizer, step: int):
    """Ejecuta la batería de prompts de evaluación."""
    print(f"\n{'' * 60}")
    print(f"  EVALUACIÓN — Paso {step:,}")
    print(f"{'' * 60}")

    for prompt in EVAL_PROMPTS[:5]:
        response = generate(
            model, tokenizer, prompt,
            max_tokens=cfg.GEN_MAX_TOKENS,
            temperature=cfg.GEN_TEMPERATURE,
            top_k=cfg.GEN_TOP_K,
            top_p=cfg.GEN_TOP_P,
        )
        q = prompt.split(TOK_USER)[-1].split(TOK_LUMEN)[0].strip()
        print(f"\n   {q}")
        print(f"   {response[:300]}")

    print(f"\n{'' * 60}\n")
    model.train()

def save_checkpoint(model, optimizer, scaler, step, loss, path):
    """Guarda un checkpoint del entrenamiento."""
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {
            "vocab_size": tokenizer.get_piece_size(),
            "d_model": cfg.D_MODEL,
            "n_heads": cfg.N_HEADS,
            "n_layers": cfg.N_LAYERS,
            "d_ff": cfg.D_FF,
            "max_seq_len": cfg.MAX_SEQ_LEN,
        },
    }
    torch.save(checkpoint, path)
    print(f"   Checkpoint guardado: {path} (step {step:,}, loss {loss:.4f})")

    if USING_GDRIVE:
        gdrive_path = cfg.GDRIVE_SAVE_DIR / path.name
        torch.save(checkpoint, gdrive_path)
        print(f"    Copia en Drive: {gdrive_path}")

def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Carga un checkpoint para continuar el entrenamiento."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"   Checkpoint cargado: {path} (step {checkpoint['step']:,})")
    return checkpoint["step"], checkpoint.get("loss", float("inf"))

def train():
    """Loop principal de entrenamiento."""
    print("\n" + "" * 60)
    print("  ¡INICIANDO ENTRENAMIENTO DE LUMEN!")
    print("" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Batches por época: {len(dataloader):,}")
    print(f"  Pasos totales: {cfg.MAX_STEPS:,}")
    print(f"  Batch efectivo: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS}")
    print("" * 60 + "\n")

    model.train()
    global_step = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    tokens_processed = 0
    last_log_time = start_time
    last_tokens = 0

    existing_ckpts = sorted(cfg.CHECKPOINT_DIR.glob("lumen_step_*.pt"))

    if not existing_ckpts and USING_GDRIVE:
        gdrive_ckpts = sorted(cfg.GDRIVE_SAVE_DIR.glob("lumen_step_*.pt"))
        if gdrive_ckpts:
            latest_gdrive = gdrive_ckpts[-1]
            local_ckpt_path = cfg.CHECKPOINT_DIR / latest_gdrive.name
            import shutil
            shutil.copy(str(latest_gdrive), str(local_ckpt_path))
            existing_ckpts = [local_ckpt_path]
            print(f"    Restaurado checkpoint desde Google Drive: {latest_gdrive.name}")

    if existing_ckpts:
        latest = existing_ckpts[-1]
        global_step, best_loss = load_checkpoint(latest, model, optimizer, scaler)
        print(f"  Reanudando desde paso {global_step:,}\n")

    epoch = 0
    while global_step < cfg.MAX_STEPS:
        epoch += 1
        for batch_idx, (x, y) in enumerate(dataloader):
            if global_step >= cfg.MAX_STEPS:
                break

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with autocast(dtype=torch.float16):
                logits, loss = model(x, y)
                loss = loss / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            tokens_processed += x.numel()

            if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr = get_lr(global_step, cfg.WARMUP_STEPS,
                           cfg.MAX_STEPS, cfg.LEARNING_RATE)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += loss.item() * cfg.GRAD_ACCUM_STEPS

                if global_step % cfg.LOG_EVERY == 0:
                    avg_loss = running_loss / cfg.LOG_EVERY
                    current_time = time.time()
                    step_time = (current_time - last_log_time) / cfg.LOG_EVERY
                    tok_per_sec = (tokens_processed - last_tokens) / (current_time - last_log_time)

                    eta_seconds = (cfg.MAX_STEPS - global_step) * step_time
                    eta_hours = eta_seconds / 3600

                    print(
                        f"  Paso {global_step:>6,}/{cfg.MAX_STEPS:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tok/s: {tok_per_sec:,.0f} | "
                        f"Época: {epoch} | "
                        f"ETA: {eta_hours:.1f}h"
                    )
                    running_loss = 0.0
                    last_log_time = current_time
                    last_tokens = tokens_processed

                if global_step % cfg.EVAL_EVERY == 0:
                    run_evaluation(model, tokenizer, global_step)
                    model.train()

                if global_step % cfg.SAVE_EVERY == 0:
                    ckpt_path = cfg.CHECKPOINT_DIR / f"lumen_step_{global_step}.pt"
                    current_loss = loss.item() * cfg.GRAD_ACCUM_STEPS
                    save_checkpoint(model, optimizer, scaler,
                                   global_step, current_loss, ckpt_path)

                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_path = cfg.CHECKPOINT_DIR / "lumen_best.pt"
                        save_checkpoint(model, optimizer, scaler,
                                       global_step, current_loss, best_path)

        print(f"\n   Época {epoch} completada \n")

    total_time = time.time() - start_time
    print("\n" + "" * 60)
    print("   ENTRENAMIENTO COMPLETADO")
    print("" * 60)
    print(f"  Pasos totales: {global_step:,}")
    print(f"  Tiempo total: {total_time/3600:.2f} horas")
    print(f"  Mejor loss: {best_loss:.4f}")
    print(f"  Tokens procesados: {tokens_processed:,}")
    print("" * 60 + "\n")

    return model

trained_model = train()

print("\n" + "" * 60)
print("  EVALUACIÓN FINAL — TODOS LOS PROMPTS")
print("" * 60)

for prompt in EVAL_PROMPTS:
    response = generate(
        trained_model, tokenizer, prompt,
        max_tokens=cfg.GEN_MAX_TOKENS,
        temperature=cfg.GEN_TEMPERATURE,
        top_k=cfg.GEN_TOP_K,
        top_p=cfg.GEN_TOP_P,
    )
    q = prompt.split(TOK_USER)[-1].split(TOK_LUMEN)[0].strip()
    print(f"\n   {q}")
    print(f"   {response[:500]}")

print(f"\n{'' * 60}\n")

def export_model(model, tokenizer):
    """Exporta el modelo final en formato safetensors + config."""
    print("\n" + "" * 60)
    print("  EXPORTANDO MODELO FINAL")
    print("" * 60)

    model_path = cfg.FINAL_MODEL_DIR / "lumen_model.pt"
    torch.save(model.state_dict(), model_path)
    size_mb = model_path.stat().st_size / 1e6
    print(f"  [OK] Pesos PyTorch: {model_path} ({size_mb:.1f} MB)")

    config = {
        "model_name": "Lumen",
        "version": "1.0.0",
        "creator": "Alessandro Villogas Gaspar",
        "organization": "Auralix Studio",
        "description": "Asistente IA on-device para Nexo / UPLA",
        "architecture": "decoder-only-transformer",
        "vocab_size": tokenizer.get_piece_size(),
        "d_model": cfg.D_MODEL,
        "n_heads": cfg.N_HEADS,
        "n_layers": cfg.N_LAYERS,
        "d_ff": cfg.D_FF,
        "max_seq_len": cfg.MAX_SEQ_LEN,
        "special_tokens": {
            "pad": TOK_PAD,
            "bos": TOK_BOS,
            "eos": TOK_EOS,
            "user": TOK_USER,
            "lumen": TOK_LUMEN,
            "end": TOK_END,
        },
        "training": {
            "steps": cfg.MAX_STEPS,
            "batch_size": cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS,
            "learning_rate": cfg.LEARNING_RATE,
            "wikipedia_docs": cfg.N_WIKIPEDIA_DOCS,
            "kb_repeat": cfg.KB_REPEAT,
        },
    }
    config_path = cfg.FINAL_MODEL_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"  [OK] Configuración: {config_path}")

    import shutil
    sp_model = Path(str(cfg.TOKENIZER_PREFIX) + ".model")
    sp_vocab = Path(str(cfg.TOKENIZER_PREFIX) + ".vocab")
    if sp_model.exists():
        shutil.copy2(sp_model, cfg.FINAL_MODEL_DIR / "tokenizer.model")
        print(f"  [OK] Tokenizador: {cfg.FINAL_MODEL_DIR / 'tokenizer.model'}")
    if sp_vocab.exists():
        shutil.copy2(sp_vocab, cfg.FINAL_MODEL_DIR / "tokenizer.vocab")

    try:
        from safetensors.torch import save_model
        st_path = cfg.FINAL_MODEL_DIR / "lumen_model.safetensors"
        save_model(model, str(st_path))
        st_mb = st_path.stat().st_size / 1e6
        print(f"  [OK] SafeTensors: {st_path} ({st_mb:.1f} MB)")
    except ImportError:
        print("  [WARN] safetensors no disponible, solo se guardó formato .pt")

    if USING_GDRIVE:
        import shutil
        dest = cfg.GDRIVE_SAVE_DIR / "lumen_final"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cfg.FINAL_MODEL_DIR, dest)
        print(f"    Copia completa en Drive: {dest}")

    print(f"\n{'' * 60}")
    print("   MODELO EXPORTADO EXITOSAMENTE")
    print(f"{'' * 60}")
    print(f"\n   Archivos en: {cfg.FINAL_MODEL_DIR}")
    print(f"  -> lumen_model.pt (pesos PyTorch)")
    print(f"  -> lumen_model.safetensors (formato seguro)")
    print(f"  -> config.json (arquitectura y metadatos)")
    print(f"  -> tokenizer.model (SentencePiece)")
    print(f"\n   Siguiente paso ")
    print(f"  Para usar en Nexo, el modelo necesita ser convertido a")
    print(f"  formato TFLite/MediaPipe. Esto se hace con las herramientas")
    print(f"  de AI Edge Torch (google-ai-edge/ai-edge-torch).")
    print(f"\n  Ejemplo:")
    print(f"  $ pip install ai-edge-torch")
    print(f"  $ python convert_to_tflite.py --model lumen_final/")
    print()

export_model(trained_model, tokenizer)

def chat(model, tokenizer):
    """Chat interactivo con Lumen para probar el modelo."""
    print("\n" + "" * 60)
    print("   CHAT CON LUMEN")
    print("  Escribe tu mensaje. 'salir' para terminar.")
    print("" * 60 + "\n")

    while True:
        try:
            user_input = input("   Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("salir", "exit", "quit"):
            print("\n  ¡Hasta luego! ")
            break

        prompt = f"{TOK_USER} {user_input} {TOK_LUMEN}"
        response = generate(
            model, tokenizer, prompt,
            max_tokens=cfg.GEN_MAX_TOKENS,
            temperature=cfg.GEN_TEMPERATURE,
            top_k=cfg.GEN_TOP_K,
            top_p=cfg.GEN_TOP_P,
        )
        print(f"   Lumen: {response}\n")

print("\n Script completado. El modelo Lumen ha sido entrenado y exportado.")
print("   Para chatear interactivamente, ejecuta: chat(trained_model, tokenizer)")