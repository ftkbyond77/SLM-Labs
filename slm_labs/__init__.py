"""SLM Labs — Thai Sign Language → (open-vocabulary) sign extraction → LLM → Thai → TTS

modules
  config     : Config / DEVICE / seed
  vocab      : 52 classes, CSV schema, filename annotation
  features   : landmarks → resample ที่ cfg.TARGET_FPS → Hand/Body/Face features, augmentation, hand-activity
  data       : dataset download, items, splits (ไม่รั่ว), DataLoaders (isolated / sentence)
  model      : SignEncoder (cls / ctc / emb head) + ArcFace margin head
  train      : train_stage_a (CE + ArcFace) / train_stage_b (CTC, encoder frozen)
  metrics    : CTC decode, WER/CER, BLEU/chrF, open-set AUROC
  extractor  : .mp4 → landmark DataFrame (MediaPipe Holistic)
  openset    : prototypes + segmentation + known/'_'/learned decision (ทางหลักของ sequence extraction)
  memory     : Qdrant local store (embeddings + metadata) + annotate → learned prototypes
  llm / tts  : natural Thai (คง ___) / speech
  viz        : Thai font global + timeline + overlay video
  pipeline   : SLMPipeline.run(video)
"""
from .config import cfg, DEVICE, seed_all, to_dev
