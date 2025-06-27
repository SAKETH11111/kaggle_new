#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Set display options for better readability
# pd.set_option('display.max_columns', 50)


# In[ ]:





# ## 1. Configuration

# In[3]:


# Global parameters
TRAIN_SAMPLE_FRAC = 0.5  # Sample 50% of data for faster iteration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Initialize Kaggle API
# api = KaggleApi()
# api.authenticate()


# ## 2. Load Data

# In[4]:


# Load parquet files
train = pd.read_parquet('kaggle/input/train.parquet')
test = pd.read_parquet('kaggle/input/test.parquet')


# In[5]:


print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Unique ranker_ids in train: {train['ranker_id'].nunique():,}")
print(f"Selected rate: {train['selected'].mean():.3f}")


# ## 3. Data Sampling & Preprocessing

# In[6]:


# Sample by ranker_id to keep groups intact
if TRAIN_SAMPLE_FRAC < 1.0:
    unique_rankers = train['ranker_id'].unique()
    n_sample = int(len(unique_rankers) * TRAIN_SAMPLE_FRAC)
    sampled_rankers = np.random.RandomState(RANDOM_STATE).choice(
        unique_rankers, size=n_sample, replace=False
    )
    train = train[train['ranker_id'].isin(sampled_rankers)]
    print(f"Sampled train to {len(train):,} rows ({train['ranker_id'].nunique():,} groups)")


# In[7]:


# Convert ranker_id to string for CatBoost
train['ranker_id'] = train['ranker_id'].astype(str)
test['ranker_id'] = test['ranker_id'].astype(str)


# ## 4. Feature Engineering

# In[8]:


cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber'
]


# In[8]:


# TODO: add time profiling
def create_features(df):
    """
    Return a copy of df enriched with engineered features.
    Fixed issues with zero-importance features.
    """
    df = df.copy()

    def hms_to_minutes(s: pd.Series) -> np.ndarray:
        """Vectorised 'HH:MM:SS' → minutes (seconds ignored)."""
        mask = s.notna()
        out = np.zeros(len(s), dtype=float)
        if mask.any():
            parts = s[mask].astype(str).str.split(':', expand=True)
            out[mask] = (
                pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
                + pd.to_numeric(parts[1], errors="coerce").fillna(0)
            )
        return out

    # Duration columns
    dur_cols = (
        ["legs0_duration", "legs1_duration"]
        + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    )
    for col in dur_cols:
        if col in df.columns:
            df[col] = hms_to_minutes(df[col])

    # Feature container
    feat = {}

    # Price features
    feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
    feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
    feat["log_price"] = np.log1p(df["totalPrice"])

    # Duration features
    df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
    feat["duration_ratio"] = np.where(
        df["legs1_duration"].fillna(0) > 0,
        df["legs0_duration"] / (df["legs1_duration"] + 1),
        1.0,
    )

    # Fix segment count features
    # Count actual segments based on non-null duration values
    for leg in (0, 1):
        seg_count = 0
        for seg in range(4):  # Check up to 4 segments
            col = f"legs{leg}_segments{seg}_duration"
            if col in df.columns:
                seg_count += df[col].notna().astype(int)
            else:
                break
        feat[f"n_segments_leg{leg}"] = seg_count

    feat["total_segments"] = feat["n_segments_leg0"] + feat["n_segments_leg1"]

    # Fix trip type detection
    # is_one_way should be 1 when there's no return leg
    feat["is_one_way"] = (
        df["legs1_duration"].isna() | 
        (df["legs1_duration"] == 0) |
        df["legs1_segments0_departureFrom_airport_iata"].isna()
    ).astype(int)

    feat["has_return"] = (1 - feat["is_one_way"]).astype(int)

    # Rank features
    grp = df.groupby("ranker_id")
    feat["price_rank"] = grp["totalPrice"].rank()
    feat["price_pct_rank"] = grp["totalPrice"].rank(pct=True)
    feat["duration_rank"] = grp["total_duration"].rank()
    feat["is_cheapest"] = (grp["totalPrice"].transform("min") == df["totalPrice"]).astype(int)
    feat["is_most_expensive"] = (grp["totalPrice"].transform("max") == df["totalPrice"]).astype(int)
    feat["price_from_median"] = grp["totalPrice"].transform(
        lambda x: (x - x.median()) / (x.std() + 1)
    )

    # Frequent-flyer features - only for airlines actually present in data
    ff = df["frequentFlyer"].fillna("").astype(str)
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")

    # Check which airlines are actually in the data
    carrier_cols = ["legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"]
    present_airlines = set()
    for col in carrier_cols:
        if col in df.columns:
            present_airlines.update(df[col].dropna().unique())

    # Only create ff features for airlines present in data
    for al in ["SU", "S7", "U6", "TK"]:  # Keep only major Russian/Turkish airlines
        if al in present_airlines:
            feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)

    # Check if FF matches carrier
    feat["ff_matches_carrier"] = 0
    for al in ["SU", "S7", "U6", "TK"]:
        if f"ff_{al}" in feat and "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["ff_matches_carrier"] |= (
                (feat.get(f"ff_{al}", 0) == 1) & 
                (df["legs0_segments0_marketingCarrier_code"] == al)
            ).astype(int)

    # Binary flags
    feat["is_vip_freq"] = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int)
    feat["has_corporate_tariff"] = (~df["corporateTariffCode"].isna()).astype(int)

    # Baggage and fees
    feat["baggage_total"] = (
        df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
        + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
    )
    feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
    feat["total_fees"] = (
        df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
    )
    feat["has_fees"] = (feat["total_fees"] > 0).astype(int)
    feat["fee_rate"] = feat["total_fees"] / (df["totalPrice"] + 1)

    # Time-of-day features
    for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            feat[f"{col}_hour"] = dt.dt.hour.fillna(12)
            feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
            h = dt.dt.hour.fillna(12)
            feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)

    # Fix direct flight detection
    feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
    feat["is_direct_leg1"] = np.where(
        feat["is_one_way"] == 1,
        0,  # One-way flights don't have leg1
        (feat["n_segments_leg1"] == 1).astype(int)
    )
    feat["both_direct"] = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype(int)

    # Cheapest direct flight
    df["_is_direct"] = feat["is_direct_leg0"] == 1
    direct_groups = df[df["_is_direct"]].groupby("ranker_id")["totalPrice"]
    if len(direct_groups) > 0:
        direct_min_price = direct_groups.min()
        feat["is_direct_cheapest"] = (
            df["_is_direct"] & 
            (df["totalPrice"] == df["ranker_id"].map(direct_min_price))
        ).astype(int)
    else:
        feat["is_direct_cheapest"] = 0
    df.drop(columns="_is_direct", inplace=True)

    # Other features
    feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)
    feat["group_size"] = df.groupby("ranker_id")["Id"].transform("count")
    feat["group_size_log"] = np.log1p(feat["group_size"])

    # Check if major carrier
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
    else:
        feat["is_major_carrier"] = 0

    # Popular routes
    popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"}
    feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)

    # Cabin class features
    feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
    feat["cabin_class_diff"] = (
        df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
    )

    # Merge new features
    df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)

    # Final NaN handling
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("missing")

    return df


# In[9]:


# OPTIMIZED PARALLEL FEATURE ENGINEERING
import time
import multiprocessing as mp
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def hms_to_minutes_fast(s):
    """Optimized 'HH:MM:SS' → minutes conversion."""
    mask = s.notna()
    out = np.zeros(len(s), dtype=float)
    if mask.any():
        parts = s[mask].astype(str).str.split(':', expand=True)
        out[mask] = (
            pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
            + pd.to_numeric(parts[1], errors="coerce").fillna(0)
        )
    return out

def create_basic_features(df):
    """Create all non-groupby features efficiently."""
    feat = {}

    # Convert duration columns
    dur_cols = (
        ["legs0_duration", "legs1_duration"]
        + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    )
    for col in dur_cols:
        if col in df.columns:
            df[col] = hms_to_minutes_fast(df[col])

    # Price features (vectorized)
    feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
    feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
    feat["log_price"] = np.log1p(df["totalPrice"])

    # Duration features
    df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
    feat["duration_ratio"] = np.where(
        df["legs1_duration"].fillna(0) > 0,
        df["legs0_duration"] / (df["legs1_duration"] + 1),
        1.0,
    )

    # Segment counting (optimized)
    for leg in (0, 1):
        seg_count = np.zeros(len(df), dtype=int)
        for seg in range(4):
            col = f"legs{leg}_segments{seg}_duration"
            if col in df.columns:
                seg_count += df[col].notna().astype(int)
            else:
                break
        feat[f"n_segments_leg{leg}"] = seg_count

    feat["total_segments"] = feat["n_segments_leg0"] + feat["n_segments_leg1"]

    # Trip type detection
    feat["is_one_way"] = (
        df["legs1_duration"].isna() | 
        (df["legs1_duration"] == 0) |
        df["legs1_segments0_departureFrom_airport_iata"].isna()
    ).astype(int)
    feat["has_return"] = (1 - feat["is_one_way"]).astype(int)

    # Frequent flyer features
    ff = df["frequentFlyer"].fillna("").astype(str)
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")

    # Airline features
    carrier_cols = ["legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"]
    present_airlines = set()
    for col in carrier_cols:
        if col in df.columns:
            present_airlines.update(df[col].dropna().unique())

    for al in ["SU", "S7", "U6", "TK"]:
        if al in present_airlines:
            feat[f"ff_{al}"] = ff.str.contains(rf"\\b{al}\\b").astype(int)

    # FF matches carrier
    feat["ff_matches_carrier"] = 0
    for al in ["SU", "S7", "U6", "TK"]:
        if f"ff_{al}" in feat and "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["ff_matches_carrier"] |= (
                (feat.get(f"ff_{al}", 0) == 1) & 
                (df["legs0_segments0_marketingCarrier_code"] == al)
            ).astype(int)

    # Binary flags
    feat["is_vip_freq"] = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int)
    feat["has_corporate_tariff"] = (~df["corporateTariffCode"].isna()).astype(int)

    # Baggage and fees
    feat["baggage_total"] = (
        df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
        + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
    )
    feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
    feat["total_fees"] = (
        df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
    )
    feat["has_fees"] = (feat["total_fees"] > 0).astype(int)
    feat["fee_rate"] = feat["total_fees"] / (df["totalPrice"] + 1)

    # Optimized time features
    datetime_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    for col in datetime_cols:
        if col in df.columns:
            # Fast string-based hour extraction when possible
            try:
                feat[f"{col}_hour"] = df[col].str[11:13].astype(float, errors='coerce').fillna(12)
            except:
                dt = pd.to_datetime(df[col], errors="coerce")
                feat[f"{col}_hour"] = dt.dt.hour.fillna(12)

            # Only parse datetime for weekday (more expensive)
            dt = pd.to_datetime(df[col], errors="coerce")
            feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)

            h = feat[f"{col}_hour"]
            feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)

    # Direct flight features
    feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
    feat["is_direct_leg1"] = np.where(
        feat["is_one_way"] == 1,
        0,
        (feat["n_segments_leg1"] == 1).astype(int)
    )
    feat["both_direct"] = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype(int)

    # Other simple features
    feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)

    if "legs0_segments0_marketingCarrier_code" in df.columns:
        feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
    else:
        feat["is_major_carrier"] = 0

    popular_routes = {"MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"}
    feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)

    feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
    feat["cabin_class_diff"] = (
        df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
    )

    return feat

def process_group_chunk(chunk_data):
    """Process a chunk of groups in parallel."""
    ranker_ids, df_chunk = chunk_data

    # Group features for this chunk
    group_features = {}

    for ranker_id in ranker_ids:
        group_df = df_chunk[df_chunk['ranker_id'] == ranker_id]
        if len(group_df) == 0:
            continue

        group_size = len(group_df)

        # Price rankings
        price_ranks = group_df['totalPrice'].rank()
        price_pct_ranks = group_df['totalPrice'].rank(pct=True)
        duration_ranks = group_df['total_duration'].rank()

        # Price statistics
        min_price = group_df['totalPrice'].min()
        max_price = group_df['totalPrice'].max()
        median_price = group_df['totalPrice'].median()
        std_price = group_df['totalPrice'].std()

        for idx in group_df.index:
            row_price = group_df.loc[idx, 'totalPrice']

            group_features[idx] = {
                'price_rank': price_ranks.loc[idx],
                'price_pct_rank': price_pct_ranks.loc[idx],
                'duration_rank': duration_ranks.loc[idx],
                'is_cheapest': int(row_price == min_price),
                'is_most_expensive': int(row_price == max_price),
                'price_from_median': (row_price - median_price) / (std_price + 1),
                'group_size': group_size
            }

    return group_features

def create_features_optimized(df, n_jobs=None, chunk_size=1000):
    """
    Optimized parallel version of create_features using joblib.
    """
    start_time = time.time()
    print(f\"Starting optimized feature engineering on {len(df):,} rows...\")\n    
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, 8)  # Leave one core free

    df = df.copy()

    # Step 1: Create basic features (fast, vectorized operations)
    print(\"Creating basic features...\")\n    basic_feat = create_basic_features(df)

    # Add basic features to DataFrame
    for col, values in basic_feat.items():
        df[col] = values

    # Step 2: Create group features in parallel
    print(f\"Creating group features using {n_jobs} cores...\")\n    
    unique_rankers = df['ranker_id'].unique()

    # Create chunks of ranker_ids
    chunks = []
    for i in range(0, len(unique_rankers), chunk_size):
        chunk_rankers = unique_rankers[i:i + chunk_size]
        chunks.append((chunk_rankers, df))

    # Process chunks in parallel
    print(f\"Processing {len(chunks)} chunks...\")\n    all_group_features = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(\n        delayed(process_group_chunk)(chunk) for chunk in chunks\n    )\n    
    # Merge all group features
    group_features_dict = {}\n    for chunk_features in all_group_features:\n        group_features_dict.update(chunk_features)\n    
    # Add group features to DataFrame
    for feature_name in ['price_rank', 'price_pct_rank', 'duration_rank', \n                        'is_cheapest', 'is_most_expensive', 'price_from_median', 'group_size']:\n        df[feature_name] = df.index.map(lambda idx: group_features_dict.get(idx, {}).get(feature_name, 0))\n    
    df['group_size_log'] = np.log1p(df['group_size'])\n    \n    # Step 3: Special features requiring global view\n    print(\"Creating special features...\")\n    \n    # Cheapest direct flight\n    df[\"_is_direct\"] = df[\"is_direct_leg0\"] == 1\n    if df[\"_is_direct\"].any():\n        direct_groups = df[df[\"_is_direct\"]].groupby(\"ranker_id\")[\"totalPrice\"].min()\n        df[\"is_direct_cheapest\"] = (\n            df[\"_is_direct\"] & \n            (df[\"totalPrice\"] == df[\"ranker_id\"].map(direct_groups).fillna(df[\"totalPrice\"]))\n        ).astype(int)\n    else:\n        df[\"is_direct_cheapest\"] = 0\n    df.drop(columns=\"_is_direct\", inplace=True)\n    \n    # Final cleanup\n    print(\"Final cleanup...\")\n    for col in df.select_dtypes(include=\"number\").columns:\n        df[col] = df[col].fillna(0)\n    for col in df.select_dtypes(include=\"object\").columns:\n        df[col] = df[col].fillna(\"missing\")\n    \n    elapsed = time.time() - start_time\n    speedup = 651.6 / elapsed  # Original time was 651.6 seconds\n    print(f\"\\n✅ Feature engineering completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)\")\n    print(f\"🚀 Speedup: {speedup:.1f}x faster than original!\")\n    print(f\"📊 Created {len([c for c in df.columns if c not in ['Id', 'ranker_id', 'selected']])} features\")\n    \n    return df

# Replace the original function\ncreate_features_parallel = create_features_optimized"


# In[9]:


# Apply feature engineering
train = create_features(train)
test = create_features(test)


# ## 5. Feature Selection

# In[10]:


# Categorical features
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber'
]

# Columns to exclude (uninformative or problematic)
exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
    'frequentFlyer',  # Already processed
    # Exclude constant or near-constant columns
    'bySelf', 'pricingInfo_passengerCount',
    # Exclude baggageAllowance_weightMeasurementType columns (likely constant)
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments1_baggageAllowance_weightMeasurementType',
    'legs1_segments0_baggageAllowance_weightMeasurementType',
    'legs1_segments1_baggageAllowance_weightMeasurementType',
    # Exclude ff features for airlines not in data
    'ff_DP', 'ff_UT', 'ff_EK', 'ff_N4', 'ff_5N', 'ff_LH'
]


# Exclude segment 2-3 columns (>98% missing)
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

feature_cols = [col for col in train.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")


# ## 6. Train/Validation Split

# In[11]:


# Prepare data
X_train = train[feature_cols]
y_train = train['selected']
groups_train = train['ranker_id']

X_test = test[feature_cols]
groups_test = test['ranker_id']

# Group-based split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
groups_tr, groups_val = groups_train.iloc[train_idx], groups_train.iloc[val_idx]

print(f"Train: {len(X_tr):,} rows, Val: {len(X_val):,} rows, Test: {len(X_test):,} rows")


# ## 7. Model Training

# In[12]:


get_ipython().run_cell_magic('capture', '', 'pip install -U xgboost\n')


# In[13]:


import xgboost as xgb


# In[ ]:


# Prepare data for XGBoost
# Convert categorical features to numeric codes for XGBoost
X_tr_xgb = X_tr.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()

# Label encode categorical features
for col in cat_features_final:
    if col in X_tr_xgb.columns:
        # Create a mapping from train d/ata
        unique_vals = pd.concat([X_tr_xgb[col], X_val_xgb[col], X_test_xgb[col]]).unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}

        X_tr_xgb[col] = X_tr_xgb[col].map(mapping).fillna(-1).astype(int)
        X_val_xgb[col] = X_val_xgb[col].map(mapping).fillna(-1).astype(int)
        X_test_xgb[col] = X_test_xgb[col].map(mapping).fillna(-1).astype(int)


# In[15]:


# Create group sizes for XGBoost
group_sizes_tr = pd.DataFrame(groups_tr).groupby('ranker_id').size().values
group_sizes_val = pd.DataFrame(groups_val).groupby('ranker_id').size().values

# Create XGBoost DMatrix
dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr, group=group_sizes_tr)
dval = xgb.DMatrix(X_val_xgb, label=y_val, group=group_sizes_val)

# XGBoost parameters
xgb_params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@3',
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10.0,
    'learning_rate': 0.05,
    'seed': RANDOM_STATE,
    'n_jobs': -1
}

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50
)


# ## 8. Model Evaluation

# In[16]:


# Convert scores to probabilities using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x / 10))

# HitRate@3 calculation
def calculate_hitrate_at_k(df, k=3):
    """Calculate HitRate@k for groups with >10 options"""
    hits = []
    for ranker_id, group in df.groupby('ranker_id'):
        if len(group) > 10:
            top_k = group.nlargest(k, 'pred')
            hit = (top_k['selected'] == 1).any()
            hits.append(hit)
    return np.mean(hits) if hits else 0.0

def evaluate_model(y_true, y_pred, groups, model_name="Model"):
    """Evaluate model performance"""
    df = pd.DataFrame({
        'ranker_id': groups,
        'pred': y_pred,
        'selected': y_true
    })

    # Get top prediction per group
    top_preds = df.loc[df.groupby('ranker_id')['pred'].idxmax()]
    top_preds['prob'] = sigmoid(top_preds['pred'])

    # Calculate metrics
    logloss = log_loss(top_preds['selected'], top_preds['prob'])
    hitrate_at_3 = calculate_hitrate_at_k(df, k=3)
    accuracy = (top_preds['selected'] == 1).mean()

    print(f"{model_name} Validation Metrics:")
    print(f"HitRate@3 (groups >10): {hitrate_at_3:.4f}")
    print(f"LogLoss:                {logloss:.4f}")
    print(f"Top-1 Accuracy:         {accuracy:.4f}")

    return df, hitrate_at_3


# In[17]:


# Evaluate XGBoost
xgb_val_preds = xgb_model.predict(dval)
xgb_val_df, xgb_hr3 = evaluate_model(y_val, xgb_val_preds, groups_val, "XGBoost")


# In[18]:


# Get XGBoost feature importance
xgb_importance = xgb_model.get_score(importance_type='gain')

# XGBoost уже возвращает словарь с именами фич, просто конвертируем в DataFrame
xgb_importance_df = pd.DataFrame([
    {'feature': k, 'xgb_importance': v} 
    for k, v in xgb_importance.items()
]).sort_values('xgb_importance', ascending=False)

print(xgb_importance_df.iloc[:30].to_string())


# ## 9. Generate Predictions

# In[19]:


# Generate predictions for test set with XGBoost
group_sizes_test = test.groupby('ranker_id').size().values
dtest = xgb.DMatrix(X_test_xgb, group=group_sizes_test)
xgb_test_preds = xgb_model.predict(dtest)

submission_xgb = test[['Id', 'ranker_id']].copy()
submission_xgb['pred_score'] = xgb_test_preds
submission_xgb['selected'] = submission_xgb.groupby('ranker_id')['pred_score'].rank(
    ascending=False, method='first'
).astype(int)

# Save submissions
submission_xgb[['Id', 'ranker_id', 'selected']].to_csv('submission.csv', index=False)

print(f"XGBoost submission saved. Shape: {submission_xgb.shape}")


# In[ ]:





# ## Submit to Competition with API

# In[20]:


# # Submit to competition
# api.competition_submit(
#     file_name="submission.parquet", 
#     competition="aeroclub-recsys-2025", 
#     message="CatBoost Ranking Baseline"
# )

