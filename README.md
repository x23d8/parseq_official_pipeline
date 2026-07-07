# Official PARSeq ANPR Pipeline

Folder nay gom code pipeline official PARSeq cho ANPR. Dataset khong duoc dua vao folder nay; cac script mac dinh tim dataset tu repo root, vi du `ocr_dataset_rescued_bbox_new`.

## Structure

- `parseq/`: source official PARSeq vendored tu repo goc.
- `train_no_refinement/`: code fine-tune official PARSeq voi `refine_iters=0` mac dinh.
- `preprocessing_best_config/`: code va notebook thu nghiem de tim best image preprocessing config.
- `refinement_finetune/`: notebook `PARSeq_Official_ANPR_Refinement_Finetune.ipynb` cho fine-tune va ablation iterative refinement.
- `outputs/`: noi cac script/notebook moi ghi ket qua mac dinh.

## Run No-Refinement Training

```powershell
python parseq_official_pipeline\train_no_refinement\parseq_official_anpr_pipeline.py `
  --data-root ocr_dataset_rescued_bbox_new `
  --epochs 5 `
  --batch-size 16
```

Neu muon bat preprocessing:

```powershell
python parseq_official_pipeline\train_no_refinement\parseq_official_anpr_pipeline.py --preprocess --preprocess-config clahe_sharpen
```

## Find Best Preprocessing Config

```powershell
python parseq_official_pipeline\preprocessing_best_config\find_best_preprocessing_config.py `
  --data-root ocr_dataset_rescued_bbox_new `
  --checkpoint parseq_official_pipeline\outputs\train_no_refinement\best_official_parseq_anpr.pt
```

Ket qua mac dinh:

- `parseq_official_pipeline/outputs/preprocessing_best_config/preprocessing_sweep_results.csv`
- `parseq_official_pipeline/outputs/preprocessing_best_config/best_preprocessing_config.json`

## Refinement Notebook

Notebook refinement nam tai:

```text
parseq_official_pipeline/refinement_finetune/PARSeq_Official_ANPR_Refinement_Finetune.ipynb
```

Khi chay lai, notebook tu tim repo root va ghi ket qua vao `parseq_official_pipeline/outputs/refinement_finetune`.
