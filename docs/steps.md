# Task tracker

## Notation
- [x] complete
- [ ] pending
- [~] in progress

---

## Phase 0 — Local preparation

- [x] Write implementation plan (docs/implementation_plan.md)
- [x] Create activate_env.sh and install_env.sh
- [x] Write all pipeline scripts (phase 1.2 through phase 5)
- [x] Write SLURM job scripts
- [ ] Prepare labels.csv with columns: subject_id, session, label

---

## Phase 1 — Data preparation on Wynton

- [ ] Upload Leuko_abcd/ to /wynton/home/sugrue/loubard/workspace/
- [ ] Upload labels.csv to Leuko_abcd/
- [ ] Run manifest builder:
      python phase1_2_build_manifest.py --labels labels.csv --out_csv manifest.csv
- [ ] Verify manifest: check row count and positive rate
- [ ] Verify transform pipeline on one sample (CPU, login node)

---

## Phase 3 — Training

- [ ] Submit training job: sbatch jobs/train_phase3.sh
- [ ] Monitor logs: tail -f logs/train_*.out
- [ ] Confirm Val AUPREC > 0.50 by epoch 30
- [ ] Wait for training to converge (100 epochs, 24-40 hours on A100)

---

## Phase 4 — Grad-CAM heatmaps

- [ ] Edit jobs/gradcam_phase4.sh: set CHECKPOINT to the best_model.pt path
- [ ] Submit: sbatch jobs/gradcam_phase4.sh
- [ ] Inspect aggregate heatmap in FSLeyes or nibabel

---

## Phase 5 — Pseudo-mask self-training (conditional)

- [ ] Only proceed if Phase 3 Val AUPREC exceeded 0.60
- [ ] Edit jobs/train_phase5.sh: set CHECKPOINT and HEATMAP_DIR
- [ ] Submit: sbatch jobs/train_phase5.sh

---

## Results

| Phase | Result | Notes |
|---|---|---|
| Manifest build | | Total rows, positive count |
| Phase 3 training | | Best Val AUPREC, epoch |
| Phase 4 heatmaps | | Aggregate map quality |
| Phase 5 fine-tuning | | Best DiceFocal loss |
