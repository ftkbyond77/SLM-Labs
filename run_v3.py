# -*- coding: utf-8 -*-
"""SLM Labs v3 — full experiment driver (GPU, conda env `hugging`).

  python run_v3.py                 # full run
  python run_v3.py --fast          # 2 epochs per stage (smoke test)

ผลลัพธ์ทั้งหมดถูกเขียนลง outputs/ (checkpoints, prototypes, *.csv, summary_v3.json)
notebook `SLM_Labs_main.ipynb` เรียกใช้ฟังก์ชันเดียวกันนี้เป็นราย section
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")

from slm_labs.config import cfg, DEVICE, seed_all
from slm_labs.vocab import N_CLASSES, SIGN_CLASSES, CLASS2WORD, NULL_CLASS, KNOWN_WORDS, UNK
from slm_labs.data import make_splits, split_report, make_loader, make_synthetic_sentences, holdout_classes, trim_items
from slm_labs.features import time_reverse
from slm_labs.model import build_model
from slm_labs.train import train_stage_a, train_stage_b, save_checkpoint, load_checkpoint
from slm_labs.metrics import eval_isolated, eval_sequence, eval_isolated_shifted, openset_auroc, tune_blank_penalty
from slm_labs.openset import (Prototypes, analyze_clip, eval_sequence_segments, eval_segment_count,
                              known_unknown_scores, tune_segmentation, tune_openset_thresholds, SEG_PARAMS)

OUT = cfg.OUT_DIR


def load_items():
    p = OUT / f"items_raw_{cfg.TARGET_FPS:g}fps.pkl"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python scripts_build_items.py` first")
    return pickle.load(open(p, "rb"))


def stage_a(iso, tag, emb_w=None, epochs=None):
    seed_all(cfg.SEED)
    m = build_model(N_CLASSES, DEVICE)
    dl_tr = make_loader(iso["train"], cfg.MAX_FRAMES_ISO, True, False)
    dl_va = make_loader(iso["val"], cfg.MAX_FRAMES_ISO, False, False)
    h = train_stage_a(m, dl_tr, dl_va, epochs or cfg.EPOCHS_A, tag=tag, emb_w=emb_w)
    return m, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    a = ap.parse_args()
    if a.fast:
        cfg.EPOCHS_A = cfg.EPOCHS_B = 2; cfg.SYN_SENTENCES = 60
    print(f"device={DEVICE} torch={torch.__version__} fps={cfg.TARGET_FPS:g} split={cfg.SPLIT_MODE} fast={a.fast}")
    S, t00 = {}, time.time()

    # ---------------- 1. data ----------------
    iso_user, iso_expert, sent_all = load_items()
    iso_user, iso_expert = trim_items(iso_user), trim_items(iso_expert)
    print(f"  trim_isolated={cfg.TRIM_ISOLATED} | frames median user={np.median([len(i['feat']['hand']) for i in iso_user]):.0f} "
          f"expert={np.median([len(i['feat']['hand']) for i in iso_expert]):.0f} sentence={np.median([len(i['feat']['hand']) for i in sent_all]):.0f}")
    iso, sent = make_splits(iso_user, iso_expert, sent_all)
    rep = split_report(iso, sent); print(rep.to_string(index=False))
    assert rep["overlap"].sum() == 0, "LEAKAGE detected"
    print(f"isolated {len(iso['train'])}/{len(iso['val'])}/{len(iso['test'])} | "
          f"sentence {len(sent['train'])}/{len(sent['val'])}/{len(sent['test'])} "
          f"(patterns {len({i['pattern'] for i in sent['train']})}/{len({i['pattern'] for i in sent['val']})}/{len({i['pattern'] for i in sent['test']})})")
    S["splits"] = dict(iso={k: len(v) for k, v in iso.items()}, sent={k: len(v) for k, v in sent.items()},
                       leakage=int(rep["overlap"].sum()), mode=cfg.SPLIT_MODE, fps=cfg.TARGET_FPS)

    dl_iso_te = make_loader(iso["test"], cfg.MAX_FRAMES_ISO, False, False)

    # ---------------- 2. Stage A (+ ablation emb_w=0 = พฤติกรรม v1/v2) ----------------
    print("\n=== Stage A (CE + ArcFace) ===")
    m, hA = stage_a(iso, "v3_stageA")
    resA = eval_isolated(m, dl_iso_te)
    print(f"  isolated TEST acc={resA['acc']:.4f} macroF1={resA['macro_f1']:.4f}")
    hA.to_csv(OUT / "hist_v3_stageA.csv", index=False)

    print("\n=== Stage A ablation: emb_w=0 (ไม่ฝึก emb head = v1/v2) ===")
    m0, hA0 = stage_a(iso, "v3_stageA_noarc", emb_w=0.0)
    resA0 = eval_isolated(m0, dl_iso_te)
    print(f"  isolated TEST acc={resA0['acc']:.4f} macroF1={resA0['macro_f1']:.4f}")

    # ---------------- 3. prototypes + open-set ----------------
    print("\n=== Prototypes / open-set ===")
    protos = Prototypes.build(m, iso["train"], iso["val"])
    protos0 = Prototypes.build(m0, iso["train"], iso["val"])
    for nm, pr in [("v3 (ArcFace)", protos), ("no-arc (v1/v2 style)", protos0)]:
        print(f"  {nm:22s} tau_sim={pr.sim_thr:.3f}  val NN-acc={pr.stats['val_nn_acc']:.3f} "
              f"correct-sim={pr.stats['val_correct_sim_mean']:.3f} wrong-sim={pr.stats['val_wrong_sim_mean']:.3f}")

    os_rows = []
    rev = [dict(feat=time_reverse(it["feat"]), label=it["label"]) for it in iso["test"]]
    for nm, mdl, pr in [("v3 (ArcFace)", m, protos), ("no-arc (v1/v2 style)", m0, protos0)]:
        ks, us = known_unknown_scores(mdl, pr, iso["test"], rev)
        os_rows.append(dict(model=nm, protocol="pseudo-unknown (time-reversed)", **openset_auroc(ks, us),
                            known_mean=round(float(ks.mean()), 4), unknown_mean=round(float(us.mean()), 4)))

    # held-out classes = "คำที่ไม่เคยเห็นจริง ๆ"
    print("\n=== open-set with true held-out classes ===")
    iso_ho, unk_items, held = holdout_classes(iso, cfg.HOLDOUT_CLASSES)
    print("  held-out:", [CLASS2WORD[SIGN_CLASSES[c]] for c in held], "| unknown clips:", len(unk_items))
    m_ho, _ = stage_a(iso_ho, "v3_holdout", epochs=cfg.EPOCHS_A)
    pr_ho = Prototypes.build(m_ho, iso_ho["train"], iso_ho["val"])
    ks, us = known_unknown_scores(m_ho, pr_ho, iso_ho["test"], unk_items)
    r = openset_auroc(ks, us)
    os_rows.append(dict(model="v3 (ArcFace)", protocol=f"held-out {len(held)} classes", **r,
                        known_mean=round(float(ks.mean()), 4), unknown_mean=round(float(us.mean()), 4)))
    m_ho0, _ = stage_a(iso_ho, "v3_holdout_noarc", emb_w=0.0, epochs=cfg.EPOCHS_A)
    pr_ho0 = Prototypes.build(m_ho0, iso_ho["train"], iso_ho["val"])
    ks0, us0 = known_unknown_scores(m_ho0, pr_ho0, iso_ho["test"], unk_items)
    os_rows.append(dict(model="no-arc (v1/v2 style)", protocol=f"held-out {len(held)} classes", **openset_auroc(ks0, us0),
                        known_mean=round(float(ks0.mean()), 4), unknown_mean=round(float(us0.mean()), 4)))
    os_df = pd.DataFrame(os_rows).round(4); print(os_df.to_string(index=False))
    S["openset"] = os_df.to_dict("records"); S["holdout_classes"] = [CLASS2WORD[SIGN_CLASSES[c]] for c in held]

    # ---------------- 4. segmentation tuning (sentence-train เท่านั้น) ----------------
    print("\n=== segmentation tuning (on sentence-train) ===")
    tune = tune_segmentation(sent["train"], set_global=True)
    print("  best:", {k: v for k, v in tune.iloc[0].items()}, "\n  SEG_PARAMS =", SEG_PARAMS)
    S["seg_tune"] = tune.head(5).to_dict("records")

    # ---------------- 5a. Stage B-ctc : CTC head บน encoder ที่ freeze ----------------
    print("\n=== Stage B-ctc (CTC head, encoder FROZEN) ===")
    syn = make_synthetic_sentences(iso["train"], cfg.SYN_SENTENCES, sent["train"])
    dl_seq_tr = make_loader(sent["train"] + syn, cfg.MAX_FRAMES_SEQ, True, True, bs=16)
    dl_seq_va = make_loader(sent["val"], cfg.MAX_FRAMES_SEQ, False, True, bs=16)
    dl_seq_te = make_loader(sent["test"], cfg.MAX_FRAMES_SEQ, False, True, bs=16)
    hB = train_stage_b(m, dl_seq_tr, dl_seq_va, cfg.EPOCHS_B, tag="v3_stageB")
    hB.to_csv(OUT / "hist_v3_stageB.csv", index=False)
    print("  ก่อน tune blank penalty: val WER=%.4f" % eval_sequence(m, dl_seq_va, blank_penalty=0.0)["wer"])
    bp = tune_blank_penalty(m, dl_seq_va); print(bp.head(4).to_string(index=False))
    print("  chosen BLANK_PENALTY =", cfg.BLANK_PENALTY)
    ctcB = eval_sequence(m, dl_seq_te); ctcB_nof = eval_sequence(m, dl_seq_te, use_face=False)
    ctcB_raw = eval_sequence(m, dl_seq_te, blank_penalty=0.0)
    print(f"  sentence TEST  WER={ctcB['wer']:.4f} CER={ctcB['cer']:.4f} sentAcc={ctcB['sent_acc']:.3f} "
          f"(bp=0 → WER={ctcB_raw['wer']:.4f})")
    print("  ex ref:", ctcB["refs"][0], "\n  ex hyp:", ctcB["hyps"][0])

    resA_after = eval_isolated(m, dl_iso_te)
    print(f"  isolated TEST after Stage B = {resA_after['acc']:.4f} (before {resA['acc']:.4f}) "
          f"→ {'PRESERVED' if abs(resA_after['acc'] - resA['acc']) < 1e-9 else 'CHANGED'}")

    # ---------------- 5b. Stage B-seg : segment → classify (ทางหลักของ pipeline) ----------------
    print("\n=== Stage B-seg (segment → classify + open-set) ===")
    thr = tune_openset_thresholds(m, protos, sent["val"])
    print(thr.head(5).to_string(index=False))
    print(f"  chosen SEG_CLS_CONF={cfg.SEG_CLS_CONF:.2f} tau_sim={protos.sim_thr:.2f}")
    protos.save(OUT / "prototypes.npz")
    segB = eval_sequence_segments(m, protos, sent["test"])
    segB_noctc = eval_sequence_segments(m, protos, sent["test"], use_ctc_cuts=False)
    print(f"  sentence TEST  WER={segB['wer']:.4f} CER={segB['cer']:.4f} sentAcc={segB['sent_acc']:.3f} "
          f"| ไม่ใช้ CTC cuts: WER={segB_noctc['wer']:.4f}")
    print("  ex ref:", segB["refs"][0], "\n  ex hyp:", segB["hyps"][0])
    S["stage_b"] = dict(seg=dict(wer=segB["wer"], cer=segB["cer"], sent_acc=segB["sent_acc"],
                                 wer_no_ctc_cuts=segB_noctc["wer"], seg_cls_conf=cfg.SEG_CLS_CONF, tau=protos.sim_thr),
                        ctc=dict(wer=ctcB["wer"], cer=ctcB["cer"], sent_acc=ctcB["sent_acc"],
                                 wer_bp0=ctcB_raw["wer"], blank_penalty=cfg.BLANK_PENALTY, wer_noface=ctcB_nof["wer"]),
                        iso_acc_before=resA["acc"], iso_acc_after=resA_after["acc"])
    save_checkpoint(m, OUT / "sign_encoder_v3.pt",
                    extra=dict(blank_penalty=cfg.BLANK_PENALTY, seg_params=dict(SEG_PARAMS), seg_cls_conf=cfg.SEG_CLS_CONF))

    # ---------------- 6. robustness + segment count ----------------
    print("\n=== domain-shift robustness ===")
    shift = pd.DataFrame([dict(strength=s, **{k: v for k, v in eval_isolated_shifted(m, iso["test"], strength=s, seed=cfg.SEED).items()})
                          for s in [0.0, 1.0, 1.6, 2.2]]).round(4)
    print(shift.to_string(index=False)); S["shift"] = shift.to_dict("records")

    seg_rows = [dict(method="motion-only", **{k: v for k, v in eval_segment_count(m, protos, sent["test"], use_model=False).items() if k != "rows"}),
                dict(method="motion+CTC cuts", **{k: v for k, v in eval_segment_count(m, protos, sent["test"]).items() if k != "rows"})]
    seg_df = pd.DataFrame(seg_rows).round(3); print(seg_df.to_string(index=False)); S["segcount"] = seg_df.to_dict("records")

    S["stage_a"] = dict(acc=resA["acc"], macro_f1=resA["macro_f1"], acc_noarc=resA0["acc"], macro_f1_noarc=resA0["macro_f1"],
                        proto=protos.stats, proto_noarc=protos0.stats, tau=protos.sim_thr, tau_noarc=protos0.sim_thr)

    # ---------------- 6b. apples-to-apples: v3 บน split เดิม (ที่รั่ว) ของ v1/v2 ----------------
    print("\n=== v3 on the OLD (leaky) split, for a like-for-like comparison with v1/v2 ===")
    iso_old, sent_old = make_splits(iso_user, iso_expert, sent_all, mode="random", sent_by_pattern=False)
    ov = len({i["pattern"] for i in sent_old["train"]} & {i["pattern"] for i in sent_old["test"]})
    print(f"  sentence-pattern overlap train/test ของ split เดิม = {ov} (split ใหม่ = 0)")
    m_old, _ = stage_a(iso_old, "v3_oldsplit")
    r_old = eval_isolated(m_old, make_loader(iso_old["test"], cfg.MAX_FRAMES_ISO, False, False))
    print(f"  v3 @ old split: isolated TEST acc={r_old['acc']:.4f} F1={r_old['macro_f1']:.4f}   "
          f"(v1=0.9759 v2=0.3373 | v3 @ honest split={resA['acc']:.4f})")
    S["old_split"] = dict(acc=r_old["acc"], macro_f1=r_old["macro_f1"], pattern_overlap=int(ov))

    # ---------------- 7. real videos ----------------
    if not a.skip_videos:
        print("\n=== real videos (data_test/*.mp4) ===")
        from slm_labs.extractor import HolisticExtractor
        from slm_labs.memory import SignMemory
        from slm_labs.pipeline import SLMPipeline
        from slm_labs.llm import set_face_baseline
        set_face_baseline(iso["train"], OUT / "face_baseline.npz")
        EXPECTED = {"ไปทานข้าวด้วยกันมั้ย": ["ไป", "_", "ข้าว", "ด้วยกัน", "_"],
                    "ฉันปลอบเพื่อนร้องไห้": ["ฉัน", "_", "_", "_"]}
        mem = SignMemory(root=cfg.MEMORY_DIR / "v3"); mem.clear()
        pipe = SLMPipeline(model=m, protos=protos, memory=mem, extractor=HolisticExtractor())
        vids = []
        for v in sorted(cfg.DATA_TEST_DIR.glob("*.mp4")):
            r = pipe.run(v, expected_words=EXPECTED.get(v.stem), speak=True)
            vids.append(dict(clip=v.stem, words=" ".join(r["words"]), expected=" ".join(EXPECTED.get(v.stem, [])),
                             n_segments=len([s for s in r["slots"] if s["status"] != "null"]),
                             n_expected=len(EXPECTED.get(v.stem, [])),
                             n_known=r["n_known"], n_unknown=r["n_unknown"], known_recall=r.get("known_recall"),
                             wer=r.get("wer"), thai=r["thai"], llm=r["llm"], wav=r["wav"], stored=r["memory_stored"]))
        vdf = pd.DataFrame(vids); print(vdf.drop(columns=["wav"]).to_string(index=False))
        vdf.to_csv(OUT / "data_test_results_v3.csv", index=False, encoding="utf-8-sig")
        S["videos"] = vids

    S["cfg"] = cfg.to_dict(); S["minutes"] = round((time.time() - t00) / 60, 1)
    (OUT / "summary_v3.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\ndone in {S['minutes']} min → outputs/summary_v3.json")


if __name__ == "__main__":
    main()
