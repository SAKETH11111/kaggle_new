# Model and training configuration

# The model to run, 'xgb' or 'lgb'
MODEL_TO_RUN = 'lgb'

# Random state for reproducibility
RANDOM_STATE = 42

# XGBoost parameters
xgb_params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@3',
    'max_depth': 10,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10.0,
    'learning_rate': 0.05,
    'seed': RANDOM_STATE,
    'tree_method': 'hist'
}

# LightGBM parameters (placeholder, can be tuned later)
lgb_params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [3],
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'seed': RANDOM_STATE,
    'boosting_type': 'gbdt',
}