"""
IPL Win Probability Predictor - Training Pipeline
==================================================
This script loads raw IPL match & delivery records, performs data cleaning,
feature engineering for 2nd innings run chases, and trains a Scikit-Learn
Pipeline using Logistic Regression.

Target Output:
- models/pipe.pkl: Serialized sklearn Pipeline
- models/metadata.json: Training metadata and feature configuration
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report

# ==========================================
# 1. Configuration & Constants
# ==========================================
ACTIVE_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

TEAM_NAME_MAPPINGS = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
}

COMMON_CITIES = [
    'Ahmedabad', 'Bangalore', 'Chandigarh', 'Chennai', 'Cuttack',
    'Delhi', 'Dharamsala', 'Dubai', 'Hyderabad', 'Indore',
    'Jaipur', 'Kolkata', 'Lucknow', 'Mohali', 'Mumbai',
    'Navi Mumbai', 'Pune', 'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MATCHES_PATH = os.path.join(DATA_DIR, 'matches.csv')
DELIVERIES_PATH = os.path.join(DATA_DIR, 'deliveries.csv')
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, 'pipe.pkl')
METADATA_OUTPUT_PATH = os.path.join(MODELS_DIR, 'metadata.json')


# ==========================================
# 2. Synthetic Data Generator (Fallback)
# ==========================================
def generate_sample_dataset():
    """
    Generates a realistic synthetic IPL dataset for initial testing
    if raw Kaggle CSV files are not yet present in data/.
    """
    print("[INFO] Kaggle CSVs not found in data/. Generating realistic sample IPL dataset...")
    np.random.seed(42)
    n_matches = 120
    match_records = []
    delivery_records = []

    for m_id in range(1, n_matches + 1):
        t1, t2 = np.random.choice(ACTIVE_TEAMS, size=2, replace=False)
        city = np.random.choice(COMMON_CITIES)
        toss_winner = np.random.choice([t1, t2])
        toss_decision = np.random.choice(['bat', 'field'])
        
        batting_first = toss_winner if toss_decision == 'bat' else (t2 if toss_winner == t1 else t1)
        chasing_team = t2 if batting_first == t1 else t1
        
        # 1st Innings: realistic score between 130 and 220
        first_innings_total = int(np.random.normal(168, 22))
        first_innings_total = max(110, min(first_innings_total, 240))
        target = first_innings_total + 1
        
        # Determine actual match winner with a realistic probability distribution
        # Stronger chase chances for reasonable targets
        chase_prob = 1.0 / (1.0 + np.exp((first_innings_total - 165) / 25))
        chaser_won = np.random.rand() < chase_prob
        winner = chasing_team if chaser_won else batting_first

        match_records.append({
            'id': m_id,
            'season': 2023,
            'city': city,
            'team1': batting_first,
            'team2': chasing_team,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'winner': winner,
            'dl_applied': 0
        })

        # 2nd Innings Deliveries simulation
        curr_score = 0
        wickets = 0
        ball_count = 0
        
        for over_idx in range(1, 21):
            if wickets >= 10 or curr_score >= target:
                break
            for ball_idx in range(1, 7):
                if wickets >= 10 or curr_score >= target:
                    break
                ball_count += 1
                
                # Dynamic run generation based on chase required rate
                runs_needed = target - curr_score
                balls_remaining = 120 - ball_count + 1
                req_rate = (runs_needed * 6) / max(balls_remaining, 1)
                
                # Higher RRR slightly increases wicket probability
                wicket_prob = min(0.04 + (req_rate * 0.005), 0.20)
                is_wicket = np.random.rand() < wicket_prob
                
                if is_wicket:
                    wickets += 1
                    ball_runs = 0
                    dismissed = f"Player_{wickets}"
                else:
                    ball_runs = int(np.random.choice([0, 1, 2, 4, 6], p=[0.35, 0.40, 0.10, 0.10, 0.05]))
                    curr_score += ball_runs
                    dismissed = np.nan

                delivery_records.append({
                    'match_id': m_id,
                    'inning': 2,
                    'batting_team': chasing_team,
                    'bowling_team': batting_first,
                    'over': over_idx,
                    'ball': ball_idx,
                    'total_runs': ball_runs,
                    'player_dismissed': dismissed
                })

    df_matches = pd.DataFrame(match_records)
    df_deliveries = pd.DataFrame(delivery_records)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df_matches.to_csv(os.path.join(DATA_DIR, 'sample_matches.csv'), index=False)
    df_deliveries.to_csv(os.path.join(DATA_DIR, 'sample_deliveries.csv'), index=False)
    print(f"[INFO] Generated sample matches ({len(df_matches)}) and deliveries ({len(df_deliveries)}) in {DATA_DIR}")
    return df_matches, df_deliveries


# ==========================================
# 3. Data Loading & Preprocessing
# ==========================================
def load_and_preprocess_data():
    """
    Loads match and delivery data, applies name normalization,
    filters defunct teams, imputes missing values, and calculates
    2nd innings run-chase features.
    """
    if os.path.exists(MATCHES_PATH) and os.path.exists(DELIVERIES_PATH):
        print(f"[INFO] Loading Kaggle datasets from {DATA_DIR}...")
        matches = pd.read_csv(MATCHES_PATH)
        deliveries = pd.read_csv(DELIVERIES_PATH)
        is_synthetic = False
    else:
        matches, deliveries = generate_sample_dataset()
        is_synthetic = True

    # 1. Normalize Team Names
    for col in ['team1', 'team2', 'winner']:
        if col in matches.columns:
            matches[col] = matches[col].replace(TEAM_NAME_MAPPINGS)

    for col in ['batting_team', 'bowling_team']:
        if col in deliveries.columns:
            deliveries[col] = deliveries[col].replace(TEAM_NAME_MAPPINGS)

    # 2. Filter for Active Franchises
    matches = matches[matches['team1'].isin(ACTIVE_TEAMS) & matches['team2'].isin(ACTIVE_TEAMS)]
    
    # 3. Remove Rain-Affected (DLS) or Tied / Incomplete matches
    if 'dl_applied' in matches.columns:
        matches = matches[matches['dl_applied'] == 0]
    matches = matches.dropna(subset=['winner'])

    # 4. Impute & Clean Cities
    if 'city' in matches.columns:
        if 'venue' in matches.columns:
            # Impute missing cities from venue names (e.g. Dubai, Sharjah, Abu Dhabi)
            matches['city'] = matches['city'].fillna(matches['venue'].apply(lambda x: str(x).split()[0]))
        matches['city'] = matches['city'].replace({'Bengaluru': 'Bangalore'})
        # Retain only common / valid cities or fill remainder with 'Other'
        matches['city'] = matches['city'].fillna('Mumbai')
    else:
        matches['city'] = 'Mumbai'

    # 5. Calculate 1st Innings Target Score (Total Runs + 1)
    if not is_synthetic:
        # Sum total runs scored in 1st innings per match
        innings1_scores = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
        innings1_scores['total_runs_x'] = innings1_scores['total_runs'] + 1
        innings1_scores = innings1_scores[['match_id', 'total_runs_x']]
        
        # Merge target score into matches dataframe
        match_df = matches.merge(innings1_scores, left_on='id', right_on='match_id')
    else:
        # In synthetic mode, calculate from match targets
        match_df = matches.copy()
        # Find the actual total from deliveries if needed or approximate
        target_series = deliveries.groupby('match_id')['total_runs'].sum() + 1
        match_df['total_runs_x'] = match_df['id'].map(target_series).fillna(165).astype(int)

    # 6. Merge with 2nd Innings Deliveries
    delivery_df = match_df.merge(deliveries, left_on='id', right_on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2]

    # 7. Compute Cumulative Score & Runs Left
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y' if 'total_runs_y' in delivery_df.columns else 'total_runs'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

    # 8. Compute Balls Left
    # Over is 1-indexed (1 to 20), ball is 1-indexed (1 to 6)
    delivery_df['balls_bowled'] = (delivery_df['over'] - 1) * 6 + delivery_df['ball']
    delivery_df['balls_left'] = 120 - delivery_df['balls_bowled']

    # 9. Compute Wickets Left
    delivery_df['is_dismissal'] = delivery_df['player_dismissed'].notnull().astype(int)
    wickets_lost = delivery_df.groupby('match_id')['is_dismissal'].cumsum()
    delivery_df['wickets_left'] = 10 - wickets_lost

    # 10. Compute Run Rates (CRR & RRR)
    # CRR = (Current Runs * 6) / Balls Bowled
    delivery_df['crr'] = np.where(
        delivery_df['balls_bowled'] > 0,
        (delivery_df['current_score'] * 6) / delivery_df['balls_bowled'],
        0.0
    )
    # RRR = (Runs Left * 6) / Balls Left
    delivery_df['rrr'] = np.where(
        delivery_df['balls_left'] > 0,
        (delivery_df['runs_left'] * 6) / delivery_df['balls_left'],
        0.0
    )

    # 11. Target Variable: Result (1 if Batting/Chasing Team Won, 0 if Bowling Team Won)
    delivery_df['result'] = (delivery_df['batting_team'] == delivery_df['winner']).astype(int)

    # 12. Data Cleaning & Sanity Filters
    # Valid chase constraints: balls_left > 0, runs_left >= 0, wickets_left >= 0
    clean_df = delivery_df[
        (delivery_df['balls_left'] > 0) &
        (delivery_df['runs_left'] >= 0) &
        (delivery_df['wickets_left'] >= 0) &
        (delivery_df['batting_team'].isin(ACTIVE_TEAMS)) &
        (delivery_df['bowling_team'].isin(ACTIVE_TEAMS))
    ].copy()

    # Filter out extreme outliers in RRR (e.g. impossible run chases > 40 rpo)
    clean_df['rrr'] = clean_df['rrr'].clip(lower=0, upper=36.0)
    clean_df['crr'] = clean_df['crr'].clip(lower=0, upper=36.0)

    # Select final feature set
    feature_cols = [
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets_left',
        'total_runs_x',
        'crr',
        'rrr'
    ]

    final_df = clean_df[feature_cols + ['result']].dropna()
    print(f"[INFO] Processed dataset size: {final_df.shape[0]} ball-by-ball chase scenarios.")
    return final_df


# ==========================================
# 4. Pipeline Building & Model Training
# ==========================================
def train_and_evaluate_model(final_df):
    """
    Builds the ColumnTransformer and LogisticRegression pipeline,
    splits data with strict featurization ordering, evaluates metrics,
    and serializes the trained model.
    """
    X = final_df.drop('result', axis=1)
    y = final_df['result']

    # Strict ML split before any fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"[INFO] Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

    # Preprocessing: OneHotEncoder for categorical columns
    categorical_cols = ['batting_team', 'bowling_team', 'city']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Classifier: Logistic Regression gives smooth, well-calibrated probabilities
    classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

    # Scikit-Learn Pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Fit pipeline strictly on training data
    print("[INFO] Fitting Scikit-Learn Pipeline...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy Score : {acc * 100:.2f}%")
    print(f"Log Loss       : {loss:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("=" * 50 + "\n")

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save Pipeline to pickle
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"[SUCCESS] Trained pipeline successfully saved to: {MODEL_OUTPUT_PATH}")

    # Save Metadata for app and documentation
    metadata = {
        'model_name': 'IPL Win Probability Predictor Pipeline',
        'algorithm': 'LogisticRegression(solver=liblinear)',
        'accuracy': round(float(acc), 4),
        'log_loss': round(float(loss), 4),
        'roc_auc': round(float(roc_auc), 4),
        'active_teams': ACTIVE_TEAMS,
        'cities': sorted(list(final_df['city'].unique())),
        'features': list(X.columns),
        'target': 'result (1 = Chasing Team Won, 0 = Bowling Team Won)'
    }

    with open(METADATA_OUTPUT_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUCCESS] Metadata successfully written to: {METADATA_OUTPUT_PATH}")

    return pipe, metadata


# ==========================================
# 5. Main Execution Entry Point
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  IPL WIN PROBABILITY PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    dataset = load_and_preprocess_data()
    train_and_evaluate_model(dataset)
    print("[COMPLETED] Training workflow finished successfully!")
