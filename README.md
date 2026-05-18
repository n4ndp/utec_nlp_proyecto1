# Fine-tuning y Ablación de DistilBERT vs BERT para Clasificación de Texto

Proyecto 1 — Procesamiento de Lenguaje Natural — UTEC

Pipeline completo para comparar **DistilBERT** y **BERT-base** en tres
benchmarks de clasificación de texto (SST-2, AG News, Yelp Polarity), con un
estudio de ablación sobre el cabezal de clasificación de DistilBERT.

Todos los resultados (JSON por corrida, CSVs resumen y gráficos) están
almacenados en Google Drive:

**[Carpeta de resultados en Google Drive](https://drive.google.com/drive/folders/11OuT2VmzLhSaiwAVpIrL3xRRjzQ2eSJD?usp=sharing)**

---

## Requisitos

- Python 3.10+
- GPU recomendada (el pipeline se ejecutó en una **NVIDIA L4** en Google Colab Pro)

```bash
pip install transformers datasets evaluate accelerate scikit-learn matplotlib seaborn
```

---

## Ejecución

El notebook está pensado para correr en **Google Colab ([link del cuaderno](https://drive.google.com/file/d/1pdd1zZu2wUxKxHB3RVxFnZb8QIzJKajP/view?usp=sharing))** con GPU. Las celdas
se ejecutan en orden; cada bloque depende solo de los anteriores.

| Celdas | Contenido |
|--------|-----------|
| 1–3    | Instalación, configuración global y carga/tokenización de datasets |
| 4–7    | Métricas, *model factory*, callback de loss y función de entrenamiento (incluye *sanity check*) |
| 8      | **Fase A** — baseline DistilBERT vs BERT en los 3 datasets |
| 9      | **Fase B** — ablación (4 configuraciones, DistilBERT + SST-2) |
| 10     | **Fase C** — mejor configuración aplicada a DistilBERT y BERT |
| 11–12  | Gráficos y tabla consolidada |

Los resultados se persisten en `MyDrive/utec_nlp_proyecto1/results/`.

---

## Documentación del código

### Configuración global

- **`DATASETS`** — diccionario con la configuración de cada dataset: argumentos
  para `load_dataset`, nombre de columna de texto y de etiqueta, número de
  clases, partición de evaluación y tamaños de submuestreo. SST-2 se usa
  completo; AG News y Yelp se submuestrean (30k y 25k) por costo de cómputo,
  con semilla fija (`SEED = 42`).
- **`MODELS`** — mapea las claves `"distilbert"` / `"bert"` a sus checkpoints
  de Hugging Face.
- Hiperparámetros globales: `MAX_LEN=128`, `BATCH_SIZE=32`, `EPOCHS=2`,
  `LR=2e-5`, `WEIGHT_DECAY=0.01`, `WARMUP_RATIO=0.1`.

### `load_and_tokenize(ds_key, tokenizer)`

Carga un dataset de Hugging Face, lo submuestrea si la configuración lo indica,
lo tokeniza (truncando a `MAX_LEN`), renombra la columna de etiquetas a
`labels` y elimina columnas innecesarias. Devuelve `(train_ds, val_ds,
num_labels)` listos para el `Trainer`.

### `compute_metrics(eval_pred)`

Calcula accuracy, precision, recall y F1. Usa promedio `weighted` para
funcionar tanto en tareas binarias (SST-2, Yelp) como multiclase (AG News, 4
clases).

### `count_parameters(model)`

Devuelve el total de parámetros y los parámetros entrenables. Clave para el
ablation: cuando el transformer está congelado, los parámetros entrenables
caen a los del cabezal.

### `measure_latency_and_memory(model, tokenizer, sample_texts, ...)`

Mide la latencia de inferencia por muestra (batch 1) y la memoria GPU pico.
Usa `torch.cuda.Event` para temporización precisa y aplica *warmup* previo
para descartar el costo de inicialización de kernels CUDA.

### `class TransformerClassifier(nn.Module)`

Wrapper sobre DistilBERT/BERT con un cabezal de clasificación **configurable**,
diseñado específicamente para el ablation:

- `freeze_transformer` — congela todos los pesos del transformer.
- `hidden_layers` — lista con el número de neuronas por capa oculta del
  cabezal. `[]` = solo capa lineal final; `[768]` = una capa oculta;
  `[512, 256]` = dos capas ocultas.

En `forward` se toma la representación del token `[CLS]`
(`last_hidden_state[:, 0]`), consistente entre BERT y DistilBERT. Se filtra
`token_type_ids` cuando el modelo es DistilBERT (no lo acepta).

### `class LossHistoryCallback(TrainerCallback)`

Callback que captura `train_loss` y `eval_loss` (más métricas de validación)
en cada paso de logging. Es la fuente de las curvas *training loss &
validation loss vs iteraciones*.

### `class RunConfig` (dataclass)

Describe un experimento: nombre, modelo base, dataset, configuración del
cabezal (`hidden_layers`, `freeze_transformer`), épocas, batch y learning
rate.

### `run_experiment(cfg, save_dir=None)`

Función principal end-to-end:

1. Carga tokenizer y datos según `cfg`.
2. Construye el `TransformerClassifier` con la configuración del cabezal.
3. Entrena con `Trainer` (`bf16=True`, eval cada 200 pasos, sin checkpoints).
4. Evalúa y mide latencia + memoria con muestras reales del set de validación.
5. Serializa el resultado completo (config, métricas, eficiencia, historial de
   loss) a JSON en `save_dir`.
6. Libera memoria GPU.

Devuelve un diccionario con todas las métricas y los historiales de loss.

---

## Estructura de resultados en Drive

```
results/
├── table_final.csv              # Tabla consolidada baseline vs mejor config
├── table_final.tex              # Misma tabla en LaTeX
├── sanity_distilbert_sst2.json  # Corrida de validación del pipeline
├── phase_a_baseline/            # 6 JSON + summary.csv (DistilBERT/BERT × 3 datasets)
├── phase_b_ablation/            # 4 JSON + summary.csv (configuraciones del cabezal)
├── phase_c_best_config/         # 6 JSON + summary.csv (mejor config × 2 modelos × 3 datasets)
└── plots/
    ├── bubble_params_vs_accuracy.pdf
    ├── loss_curves_baseline.pdf
    └── ablation_comparison.pdf
```

Cada archivo JSON contiene la configuración del experimento, las métricas
finales, las mediciones de eficiencia (latencia, memoria, parámetros) y el
historial completo de *training/validation loss* por paso.

---

## Autores

- Fabian Alvarado — Universidad de Ingeniería y Tecnología (UTEC)
- Fernando Choqque — Universidad de Ingeniería y Tecnología (UTEC)
- Alexandro Chamochumbi — Universidad de Ingeniería y Tecnología (UTEC)
