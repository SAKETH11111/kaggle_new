import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def train_xgb_model(X_tr, y_tr, X_val, y_val, groups_tr, groups_val, model_params):
    """
    Trains an XGBoost model and returns the model and validation predictions.
    """
    logger.info("Preparing data for XGBoost...")
    
    X_tr_xgb = X_tr.copy()
    X_val_xgb = X_val.copy()
    
    # Create group sizes for XGBoost
    group_sizes_tr = groups_tr.value_counts().sort_index().values
    group_sizes_val = groups_val.value_counts().sort_index().values

    logger.info("Creating XGBoost DMatrix objects…")
    dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr, group=group_sizes_tr)
    dval = xgb.DMatrix(X_val_xgb, label=y_val, group=group_sizes_val)

    logger.info("Training XGBoost model...")
    start_time = datetime.now()
    
    model = xgb.train(
        model_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    training_time = datetime.now() - start_time
    logger.info(f"XGBoost training completed in {training_time}")
    
    # Make predictions
    val_preds = model.predict(dval)
    
    return model, val_preds

def train_lgb_model(X_tr, y_tr, X_val, y_val, groups_tr, groups_val, model_params, cat_features):
    """
    Trains a LightGBM model and returns the model and validation predictions.
    """
    logger.info("Preparing data for LightGBM...")

    # Create group sizes
    group_sizes_tr = groups_tr.value_counts().sort_index()
    group_sizes_val = groups_val.value_counts().sort_index()

    logger.info("Creating LightGBM Dataset objects...")
    dtrain = lgb.Dataset(
        X_tr, 
        label=y_tr, 
        group=group_sizes_tr,
        categorical_feature=cat_features,
        free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val, 
        label=y_val, 
        group=group_sizes_val,
        categorical_feature=cat_features,
        reference=dtrain,
        free_raw_data=False
    )

    logger.info("Training LightGBM model...")
    start_time = datetime.now()

    model = lgb.train(
        model_params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    training_time = datetime.now() - start_time
    logger.info(f"LightGBM training completed in {training_time}")

    # Make predictions
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    return model, val_preds