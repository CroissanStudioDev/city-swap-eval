# City Swap Adversarial Fairness Evaluation

Scripts for Part 1 of the resume screening fairness analysis.

## Quick Start

Open `city_swap_batch_eval.ipynb` from your `notebooks/` workflow and run all cells.

The notebook expects:

- `../data/processed/test.csv`
- `../data/processed/label_to_supercategory_v1.csv`
- model checkpoints under `models/`

## Output

Results saved to `results/city_swap_all/city_swap_all_models.json`

## Models tested

1. Baseline BERT (`bert_9classes_final`)
2. GroupDRO (`bert_gdro_eta01_2ep`)
3. Data Scrubbing (`bert_scrubbing`)
4. Oversampling (`bert_oversample_only`)
5. Focal Loss (`bert_focal_loss`)
6. Adversarial Debiasing (`bert_adversarial`)
7. Label Smoothing (`bert_label_smoothing`)
8. Attribution Regularization (`bert_attr_reg`)
9. Combined Scrub+GroupDRO (`bert_debiased_combo`)
10. Combined Scrub+GroupDRO Best (`combined_scrubbing_groupdro`)

## Notes

- Non-scrubbing models are evaluated on raw resume text before and after city replacement.
- Scrubbing models are evaluated on scrubbed text before and after city replacement.
- Invalid checkpoints with a missing classifier head are skipped from evaluation.
