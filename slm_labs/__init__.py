"""SLM Labs — Thai Sign Language → (open-vocabulary) sign extraction → LLM → Thai → TTS

modules
  config     : Config / DEVICE / seed
  vocab      : 52 classes, CSV schema, filename annotation
  features   : landmarks → Hand/Body/Face features, augmentation, hand-activity, global/local views
  data       : dataset download, items, splits, DataLoaders (isolated / sentence / SSL views)
  model      : SignEncoder (+emb head) , SignDINO (SSL)
  train      : train_ssl / train_stage_a / train_stage_b
  metrics    : CTC decode, WER/CER, BLEU/chrF, open-set AUROC
  extractor  : .mp4 → landmark DataFrame (MediaPipe Holistic)
  openset    : prototypes + segmentation + known/unknown decision
  memory     : Qdrant local store (embeddings + metadata) + annotate → learned prototypes
  llm / tts  : natural Thai (คง ___) / speech
  viz        : Thai font global + timeline + overlay video
  pipeline   : SLMPipeline.run(video)
"""
from .config import cfg, DEVICE, seed_all, to_dev
