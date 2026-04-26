# Submission Recommendation

## Recommended File

- Main file in repo root:
  [submission_recommended_splitv2_plus_pairedlr8e4.csv](submission_recommended_splitv2_plus_pairedlr8e4.csv)
- Source run:
  `blend_terminal_splitv2_plus_pairedlr8e4_refined`
- Local validation:
  `oof_auc = 0.9783852921796876`

## Why This Version

- It is higher than the current old-universe fallback
  `blend_terminal_old_universe_refined = 0.9781914962338988`
- It is higher than the earlier splitv2 terminal blend
  `blend_terminal_splitv2_refined = 0.9781785765041795`
- It stays inside one `splitv2` universe, so it is easier to justify than the mixed-pool ceiling blend.

## Upload Rule

- Rename the file to `姓名_学号_submission.csv` before upload.
- The file already follows the template order from [name_sid_submission.csv](name_sid_submission.csv).

## Related Files

- [blend.json](outputs/runs/blend_terminal_splitv2_plus_pairedlr8e4_refined/blend.json)
- [metrics.json](outputs/runs/blend_terminal_splitv2_plus_pairedlr8e4_refined/metrics.json)
- [submission copy in outputs](outputs/submissions/blend_terminal_splitv2_plus_pairedlr8e4_refined_submission.csv)
