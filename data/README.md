# IPL Dataset Directory

Place your standard Kaggle IPL dataset files in this directory:
- `matches.csv` (Match level metadata: teams, venue, toss, winner, etc.)
- `deliveries.csv` (Ball-by-ball delivery data: match_id, inning, over, ball, batsman, bowler, total_runs, player_dismissed)

### Recommended Kaggle Sources:
1. **IPL Complete Dataset (2008-2024 / 2020)**:
   - Kaggle Dataset: [IPL Complete Dataset (2008-2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
   - Alternative: [IPL Ball by Ball 2008 to 2023](https://www.kaggle.com/datasets/littleraj30/ipl-dataset)

### Fallback Behavior:
If `matches.csv` and `deliveries.csv` are not found when running `python train_model.py`, the training script will automatically create realistic sample data covering 2,000+ match chase balls across all active franchises so you can test and demonstrate the pipeline immediately without waiting for Kaggle downloads.
