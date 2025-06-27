import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import xgboost as xgb
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import psutil
import os
import pickle
import hashlib
from pathlib import Path

# Set up logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('training_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure tqdm for better progress bars
tqdm.pandas(desc="Processing")

# Define categorical features
CAT_COLS = [
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

logger.info("=" * 80)
logger.info("Starting XGBoost Ranker Baseline for FlightRank 2025")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
#  Early configuration (defined BEFORE first use)
# -----------------------------------------------------------------------------

# Cache configuration
CACHE_DIR = Path("data_cache")
CACHE_VERSION = "v2.2"  
ENABLE_CACHE = False

# Core run-time parameters
RANDOM_STATE = 42
# Force use of all available CPU cores
cpu_count = mp.cpu_count()
N_JOBS = cpu_count  # Use all 112 cores
target_ram_usage = 0.85
TRAIN_SAMPLE_FRAC = 0.01 # Use a small sample for verification

def get_cache_key(*args):
    """Generate a cache key from arguments."""
    content = str(args) + CACHE_VERSION
    return hashlib.md5(content.encode()).hexdigest()

def save_to_cache(data, cache_key, cache_type="data"):
    """Save data to cache with compression."""
    if not ENABLE_CACHE:
        return
    
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size for logging
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {cache_type} to cache: {cache_file.name} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def load_from_cache(cache_key, cache_type="data"):
    """Load data from cache."""
    if not ENABLE_CACHE:
        return None
    
    cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        # Get file size for logging
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        logger.info(f"Loaded {cache_type} from cache: {cache_file.name} ({size_mb:.1f} MB) ⚡")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None

def clear_cache():
    """Clear all cached data."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")

# Global parameters with memory management
TRAIN_SAMPLE_FRAC = 1.0  # Sample 100% of data for robust training
RANDOM_STATE = 42

# Dynamic high-performance configuration
total_ram_gb = psutil.virtual_memory().total / (1024**3)
cpu_count = mp.cpu_count()

# Use 90% of available RAM and all CPU cores
target_ram_usage = 0.90  
available_ram_gb = total_ram_gb * target_ram_usage
N_JOBS = cpu_count  # Use all available cores dynamically
np.random.seed(RANDOM_STATE)

logger.info(f"System Configuration:")
logger.info(f"  - CPU cores: {cpu_count}")
logger.info(f"  - Using jobs: {N_JOBS} (all cores)")
logger.info(f"  - Total RAM: {total_ram_gb:.1f} GB")
logger.info(f"  - Target RAM usage: {available_ram_gb:.1f} GB ({target_ram_usage*100:.0f}%)")
logger.info(f"  - Training sample fraction: {TRAIN_SAMPLE_FRAC}")
logger.info(f"  - Random state: {RANDOM_STATE}")

def timer(func):
    """Decorator to time function execution and monitor memory"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        memory_used = end_memory - start_memory
        
        logger.info(f"Function '{func.__name__}' completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory usage: {end_memory:.1f}% (Δ{memory_used:+.1f}%)")
        
        # Warning if memory usage is too high
        memory_threshold = target_ram_usage * 100
        if end_memory > memory_threshold:
            logger.warning(f"High memory usage detected: {end_memory:.1f}%")
            
        return result
    return wrapper

def monitor_memory():
    """Monitor current memory usage"""
    memory = psutil.virtual_memory()
    logger.info(f"Current memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
    return memory.percent

@timer
def load_data():
    """Load parquet files with caching"""
    cache_key = get_cache_key("raw_data")
    cached_data = load_from_cache(cache_key, "raw_data")
    
    if cached_data is not None:
        train, test = cached_data
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        logger.info(f"Unique ranker_ids in train: {train['ranker_id'].nunique():,}")
        logger.info(f"Selected rate: {train['selected'].mean():.3f}")
        return train, test
    
    logger.info("Loading data files...")
    with tqdm(total=2, desc="Loading files") as pbar:
        train = pd.read_parquet('kaggle/input/train.parquet')
        pbar.update(1)
        test = pd.read_parquet('kaggle/input/test.parquet')
        pbar.update(1)
    
    # Cache the raw data
    save_to_cache((train, test), cache_key, "raw_data")
    
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logger.info(f"Unique ranker_ids in train: {train['ranker_id'].nunique():,}")
    logger.info(f"Selected rate: {train['selected'].mean():.3f}")
    
    return train, test

# Load data
train, test = load_data()

@timer
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to their most memory-efficient types.
    """
    logger.info(f"Optimizing memory for dataframe with shape {df.shape}...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    return df

# Optimize memory usage after loading
train = optimize_memory(train)
test = optimize_memory(test)

@timer
def sample_data(train, sample_frac):
    """Sample training data by ranker_id"""
    if sample_frac >= 1.0:
        return train
        
    logger.info(f"Sampling {sample_frac*100}% of training data by ranker_id...")
    unique_rankers = train['ranker_id'].unique()
    n_sample = int(len(unique_rankers) * sample_frac)
    
    with tqdm(desc="Sampling data") as pbar:
        sampled_rankers = np.random.RandomState(RANDOM_STATE).choice(
            unique_rankers, size=n_sample, replace=False
        )
        pbar.update(50)
        
        train_sampled = train[train['ranker_id'].isin(sampled_rankers)]
        pbar.update(50)
    
    logger.info(f"Sampled train to {len(train_sampled):,} rows ({train_sampled['ranker_id'].nunique():,} groups)")
    return train_sampled

# Sample by ranker_id to keep groups intact
train = sample_data(train, TRAIN_SAMPLE_FRAC)

@timer
def convert_ranker_ids(train, test):
    """Convert ranker_id to string for processing"""
    logger.info("Converting ranker_id to string...")
    
    def convert_df(df, desc):
        with tqdm(desc=desc) as pbar:
            df['ranker_id'] = df['ranker_id'].astype(str)
            pbar.update(100)
        return df
    
    # Use threading for I/O bound operations
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_train = executor.submit(convert_df, train, "Converting train ranker_ids")
        future_test = executor.submit(convert_df, test, "Converting test ranker_ids")
        
        train = future_train.result()
        test = future_test.result()
    
    return train, test

train, test = convert_ranker_ids(train, test)

def process_duration_chunk(chunk_data):
    """Process duration columns for a chunk of data"""
    chunk, dur_cols = chunk_data
    
    def hms_to_minutes(s: pd.Series) -> np.ndarray:
        """Vectorised 'HH:MM:SS' → minutes (seconds ignored)."""
        mask = s.notna()
        out = np.zeros(len(s), dtype=float)
        if mask.any():
            # Use string accessor on the Series directly
            parts = s[mask].astype(str).str.split(':', expand=True)
            
            # Convert parts to numeric, ensuring proper handling
            hours = pd.to_numeric(parts.iloc[:, 0], errors="coerce").fillna(0)
            minutes = pd.to_numeric(parts.iloc[:, 1], errors="coerce").fillna(0)
            
            out[mask] = (hours.astype(float) * 60 + minutes.astype(float))
        return out
    
    for col in dur_cols:
        if col in chunk.columns:
            chunk[col] = hms_to_minutes(chunk[col])
    
    return chunk

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df enriched with engineered features.
    Uses parallel processing for intensive operations.
    """
    logger.info(f"Starting feature engineering for dataset with shape {df.shape}")
    df = df.copy()

    # Process duration columns in parallel chunks
    logger.info("Processing duration columns in parallel...")
    dur_cols = (
        ["legs0_duration", "legs1_duration"]
        + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    )
    
    # Force maximum parallelism - use all available CPU cores
    memory_usage = psutil.virtual_memory().percent
    memory_limit = target_ram_usage * 100 - 5  # 5% buffer before reducing parallelism
    if memory_usage > memory_limit:
        # Only reduce parallelism if memory is critically high
        effective_jobs = max(cpu_count // 2, N_JOBS // 2)
        logger.warning(f"Critical memory usage ({memory_usage:.1f}%), reducing parallelism to {effective_jobs} jobs")
    else:
        effective_jobs = N_JOBS  # Use all 112 cores
    
    logger.info(f"Using {effective_jobs} cores for parallel processing")
    
    # Split dataframe into more chunks for better parallelization
    chunk_size = max(500, len(df) // (effective_jobs * 2))  # Smaller chunks for better distribution
    chunks = [df.iloc[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]
    chunk_data = [(chunk, dur_cols) for chunk in chunks]
    
    with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
        with tqdm(total=len(chunks), desc=f"Processing duration chunks on {effective_jobs} cores") as pbar:
            processed_chunks = []
            for result in executor.map(process_duration_chunk, chunk_data):
                processed_chunks.append(result)
                pbar.update(1)
    
    # Combine processed chunks
    df = pd.concat(processed_chunks, ignore_index=True)

    logger.info("Creating price and duration features...")
    # Feature container
    feat = {}

    with tqdm(desc="Price features", total=3) as pbar:
        # Price features
        feat["price_per_tax"] = df["totalPrice"] / (df["taxes"] + 1)
        pbar.update(1)
        feat["tax_rate"] = df["taxes"] / (df["totalPrice"] + 1)
        pbar.update(1)
        feat["log_price"] = np.log1p(df["totalPrice"])
        pbar.update(1)

    with tqdm(desc="Duration features", total=2) as pbar:
        # Duration features
        df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
        pbar.update(1)
        feat["duration_ratio"] = np.where(
            df["legs1_duration"].fillna(0) > 0,
            df["legs0_duration"] / (df["legs1_duration"] + 1),
            1.0,
        )
        pbar.update(1)

    logger.info("Computing segment counts...")
    # Fix segment count features
    with tqdm(desc="Segment counts", total=2) as pbar:
        for leg in (0, 1):
            seg_count = 0
            for seg in range(4):  # Check up to 4 segments
                col = f"legs{leg}_segments{seg}_duration"
                if col in df.columns:
                    seg_count += df[col].notna().astype(int)
                else:
                    break
            feat[f"n_segments_leg{leg}"] = seg_count
            pbar.update(1)

    feat["total_segments"] = feat["n_segments_leg0"] + feat["n_segments_leg1"]

    logger.info("Detecting trip types...")
    with tqdm(desc="Trip types", total=2) as pbar:
        # Fix trip type detection
        feat["is_one_way"] = (
            df["legs1_duration"].isna() | 
            (df["legs1_duration"] == 0) |
            df["legs1_segments0_departureFrom_airport_iata"].isna()
        ).astype(int)
        pbar.update(1)
        
        feat["has_return"] = (1 - feat["is_one_way"]).astype(int)
        pbar.update(1)

    logger.info("Creating 'days-out' feature...")
    with tqdm(desc="Days-out feature", total=1) as pbar:
        request_dt = pd.to_datetime(df['requestDate'], errors='coerce')
        departure_dt = pd.to_datetime(df['legs0_departureAt'], errors='coerce')
        
        # Calculate the difference in days
        days_out = (departure_dt - request_dt).dt.days
        feat['days_out'] = days_out
        pbar.update(1)

    logger.info("Creating ranking features...")
    # Optimize ranking features with parallel computation
    grp = df.groupby("ranker_id")
    
    def compute_ranking_features():
        # Vectorized calculations for speed
        with tqdm(desc="Ranking features (vectorized)", total=6) as pbar:
            price_ranks = grp["totalPrice"].rank()
            pbar.update(1)
            price_pct_ranks = grp["totalPrice"].rank(pct=True)
            pbar.update(1)
            duration_ranks = grp["total_duration"].rank()
            pbar.update(1)
            
            price_min = grp["totalPrice"].transform("min")
            is_cheapest = (price_min == df["totalPrice"]).astype(int)
            pbar.update(1)
            
            price_max = grp["totalPrice"].transform("max")
            is_most_expensive = (price_max == df["totalPrice"]).astype(int)
            pbar.update(1)

            # Optimized median/std calculation, replacing slow lambda function
            medians = grp["totalPrice"].transform("median")
            stds = grp["totalPrice"].transform("std")
            price_from_median = (df["totalPrice"] - medians) / (stds + 1)
            pbar.update(1)

        return {
            "price_rank": price_ranks,
            "price_pct_rank": price_pct_ranks,
            "duration_rank": duration_ranks,
            "is_cheapest": is_cheapest,
            "is_most_expensive": is_most_expensive,
            "price_from_median": price_from_median
        }
    
    ranking_features = compute_ranking_features()
    feat.update(ranking_features)

    logger.info("Processing frequent flyer data...")
    # Frequent-flyer features - only for airlines actually present in data
    ff = df["frequentFlyer"].fillna("").astype(str)
    feat["n_ff_programs"] = ff.str.count("/") + (ff != "")

    # Check which airlines are actually in the data
    carrier_cols = ["legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code"]
    present_airlines = set()
    for col in carrier_cols:
        if col in df.columns:
            present_airlines.update(df[col].dropna().unique())

    logger.info(f"Airlines present in data: {sorted(present_airlines)}")

    # Only create ff features for airlines present in data
    with tqdm(desc="FF features", total=4) as pbar:
        for al in ["SU", "S7", "U6", "TK"]:  # Keep only major Russian/Turkish airlines
            if al in present_airlines:
                feat[f"ff_{al}"] = ff.str.contains(rf"\b{al}\b").astype(int)
            pbar.update(1)

    # Check if FF matches carrier
    feat["ff_matches_carrier"] = 0
    for al in ["SU", "S7", "U6", "TK"]:
        if f"ff_{al}" in feat and "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["ff_matches_carrier"] |= (
                (feat.get(f"ff_{al}", 0) == 1) & 
                (df["legs0_segments0_marketingCarrier_code"] == al)
            ).astype(int)

    logger.info("Creating binary flag features...")
    with tqdm(desc="Binary flags", total=8) as pbar:
        # Binary flags
        feat["is_vip_freq"] = ((df["isVip"] == 1) | (feat["n_ff_programs"] > 0)).astype(int)
        pbar.update(1)
        feat["has_corporate_tariff"] = (~df["corporateTariffCode"].isna()).astype(int)
        pbar.update(1)

        # Baggage and fees
        feat["baggage_total"] = (
            df["legs0_segments0_baggageAllowance_quantity"].fillna(0)
            + df["legs1_segments0_baggageAllowance_quantity"].fillna(0)
        )
        pbar.update(1)
        feat["has_baggage"] = (feat["baggage_total"] > 0).astype(int)
        pbar.update(1)
        feat["total_fees"] = (
            df["miniRules0_monetaryAmount"].fillna(0) + df["miniRules1_monetaryAmount"].fillna(0)
        )
        pbar.update(1)
        feat["has_fees"] = (feat["total_fees"] > 0).astype(int)
        pbar.update(1)
        feat["fee_rate"] = feat["total_fees"] / (df["totalPrice"] + 1)
        pbar.update(1)

    logger.info("Creating time-of-day features...")
    # Time-of-day features
    time_cols = ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt")
    with tqdm(desc="Time features", total=len(time_cols)) as pbar:
        for col in time_cols:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                feat[f"{col}_hour"] = dt.dt.hour.fillna(12)
                feat[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
                h = dt.dt.hour.fillna(12)
                feat[f"{col}_business_time"] = (((6 <= h) & (h <= 9)) | ((17 <= h) & (h <= 20))).astype(int)
            pbar.update(1)

    logger.info("Processing direct flight features...")
    with tqdm(desc="Direct flight features", total=4) as pbar:
        # Fix direct flight detection
        feat["is_direct_leg0"] = (feat["n_segments_leg0"] == 1).astype(int)
        pbar.update(1)
        feat["is_direct_leg1"] = np.where(
            feat["is_one_way"] == 1,
            0,  # One-way flights don't have leg1
            (feat["n_segments_leg1"] == 1).astype(int)
        )
        pbar.update(1)
        feat["both_direct"] = (feat["is_direct_leg0"] & feat["is_direct_leg1"]).astype(int)
        pbar.update(1)

        # Cheapest direct flight
        df["_is_direct"] = feat["is_direct_leg0"] == 1
        direct_groups = df[df["_is_direct"]].groupby("ranker_id")["totalPrice"]
        if len(direct_groups) > 0:
            direct_min_price = direct_groups.min()
            # Use .to_dict() to resolve potential map ambiguity for the linter
            min_price_map = direct_min_price.to_dict()
            # Use a lambda to ensure mapping is correctly interpreted
            feat["is_direct_cheapest"] = (
                df["_is_direct"] & 
                (df["totalPrice"] == df["ranker_id"].map(lambda r: min_price_map.get(r)))
            ).astype(int)
        else:
            feat["is_direct_cheapest"] = 0
        df.drop(columns="_is_direct", inplace=True)
        pbar.update(1)

    logger.info("Creating miscellaneous features...")
    with tqdm(desc="Misc features", total=7) as pbar:
        # Other features
        feat["has_access_tp"] = (df["pricingInfo_isAccessTP"] == 1).astype(int)
        pbar.update(1)
        feat["group_size"] = df.groupby("ranker_id")["Id"].transform("count")
        pbar.update(1)
        feat["group_size_log"] = np.log1p(feat["group_size"])
        pbar.update(1)

        # Check if major carrier
        if "legs0_segments0_marketingCarrier_code" in df.columns:
            feat["is_major_carrier"] = df["legs0_segments0_marketingCarrier_code"].isin(["SU", "S7", "U6"]).astype(int)
        else:
            feat["is_major_carrier"] = 0
        pbar.update(1)

        # Popular routes
        popular_routes = ["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"]
        feat["is_popular_route"] = df["searchRoute"].isin(popular_routes).astype(int)
        pbar.update(1)

        # Cabin class features
        feat["avg_cabin_class"] = df[["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]].mean(axis=1)
        pbar.update(1)
        feat["cabin_class_diff"] = (
            df["legs0_segments0_cabinClass"].fillna(0) - df["legs1_segments0_cabinClass"].fillna(0)
        )
        pbar.update(1)

    logger.info("Creating interaction features...")
    with tqdm(desc="Interaction features", total=2) as pbar:
        feat["price_per_dur"] = df["totalPrice"] / (df["total_duration"] + 1)
        pbar.update(1)
        feat["policy_violation"] = (~feat["has_access_tp"].astype(bool)).astype(int) * feat["has_return"]
        pbar.update(1)

    logger.info("Merging engineered features...")
    # Merge new features
    with tqdm(desc="Merging features") as pbar:
        df = pd.concat([df, pd.DataFrame(feat, index=df.index)], axis=1)
        pbar.update(100)

    logger.info("Handling NaN values...")
    # Final NaN handling
    numeric_cols = df.select_dtypes(include="number").columns
    object_cols = df.select_dtypes(include="object").columns
    
    with tqdm(desc="Filling NaN values", total=2) as pbar:
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        pbar.update(1)
        
        for col in object_cols:
            df[col] = df[col].fillna("missing")
        pbar.update(1)

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df

# Apply feature engineering with caching
logger.info("Applying feature engineering...")
cache_key = get_cache_key("features", TRAIN_SAMPLE_FRAC)
cached_features = load_from_cache(cache_key, "features")

if cached_features is not None:
    train, test = cached_features
else:
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_train = executor.submit(create_features, train)
        future_test = executor.submit(create_features, test)
        
        train = future_train.result()
        test = future_test.result()
    
    # Cache the processed features
    save_to_cache((train, test), cache_key, "features")


logger.info("Defining columns to exclude...")
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
cat_features_final = [col for col in CAT_COLS if col in feature_cols]

logger.info(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

@timer
def prepare_data():
    """Prepare training data with caching"""
    cache_key = get_cache_key("prepared_data", len(feature_cols), TRAIN_SAMPLE_FRAC)
    cached_data = load_from_cache(cache_key, "prepared")
    
    if cached_data is not None:
        logger.info("Loaded prepared data from cache")
        return cached_data
    
    logger.info("Preparing training data...")
    
    # Parallel data extraction
    def extract_train_data():
        return train[feature_cols], train['selected'], train['ranker_id']
    
    def extract_test_data():
        return test[feature_cols], test['ranker_id']
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        with tqdm(desc="Data preparation", total=2) as pbar:
            future_train = executor.submit(extract_train_data)
            future_test = executor.submit(extract_test_data)
            
            X_train, y_train, groups_train = future_train.result()
            pbar.update(1)
            
            X_test, groups_test = future_test.result()
            pbar.update(1)
    
    result = (X_train, y_train, groups_train, X_test, groups_test)
    
    # Cache the prepared data
    save_to_cache(result, cache_key, "prepared")
    
    return result

X_train, y_train, groups_train, X_test, groups_test = prepare_data()

@timer
def perform_train_val_split(X_train, y_train, groups_train, train_df):
    """Perform stratified group-based k-fold cross-validation with caching."""
    cache_key = get_cache_key("cv_folds", len(X_train), RANDOM_STATE)
    cached_folds = load_from_cache(cache_key, "cv_folds")
    
    if cached_folds is not None:
        logger.info("Loaded CV folds from cache")
        for fold, (train_idx, val_idx) in enumerate(cached_folds):
            logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Val size={len(val_idx)}")
        return cached_folds
    
    logger.info("Performing 5-fold stratified group cross-validation...")
    
    # Ensure 'has_return' is available for stratification
    if 'has_return' not in train_df.columns:
        raise ValueError("Stratification column 'has_return' not found in the training data.")
        
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # The stratification is done on 'has_return'
    stratify_on = train_df.loc[X_train.index]['has_return']
    
    # Store indices for each fold
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, stratify_on, groups=groups_train)):
        folds.append((train_idx, val_idx))
        logger.info(f"Fold {fold+1}: Train size={len(train_idx)}, Val size={len(val_idx)}")
    
    # Cache the folds
    save_to_cache(folds, cache_key, "cv_folds")
        
    return folds

from sklearn.model_selection import StratifiedGroupKFold

def encode_categorical_column(args):
    col, X_tr_col, X_val_col, X_test_col = args
    
    # Create a mapping from all data
    unique_vals = pd.concat([X_tr_col, X_val_col, X_test_col]).unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    
    encoded_tr = X_tr_col.map(mapping).fillna(-1).astype(int)
    encoded_val = X_val_col.map(mapping).fillna(-1).astype(int)
    encoded_test = X_test_col.map(mapping).fillna(-1).astype(int)
    
    return col, encoded_tr, encoded_val, encoded_test

# (Inside the main block, after loading and preparing data)

# Loop through each fold for training and evaluation
all_val_preds = []
all_val_true = []
all_val_groups = []
fold_hit_rates = []

# Define helper functions for the CV loop

@timer
def prepare_xgboost_data(X_tr, X_val, y_tr, y_val, groups_tr, groups_val, X_test, cat_features_final):
    """Prepare data for XGBoost with parallel label encoding"""
    logger.info("Preparing data for XGBoost...")
    
    X_tr_xgb = X_tr.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()

    # Label encode categorical features in parallel using ALL cores
    logger.info("Label encoding categorical features...")
    
    # Prepare arguments for parallel processing
    encoding_args = []
    for col in cat_features_final:
        if col in X_tr_xgb.columns:
            encoding_args.append((col, X_tr_xgb[col], X_val_xgb[col], X_test_xgb[col]))
    
    # Process in parallel with maximum parallelization - use all 112 cores
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        with tqdm(total=len(encoding_args), desc=f"Encoding categorical features on {N_JOBS} cores") as pbar:
            # Submit all jobs at once for better parallelization
            futures = {executor.submit(encode_categorical_column, args): args[0] for args in encoding_args}
            for future in as_completed(futures):
                col, encoded_tr, encoded_val, encoded_test = future.result()
                X_tr_xgb[col] = encoded_tr
                X_val_xgb[col] = encoded_val
                X_test_xgb[col] = encoded_test
                pbar.update(1)
    
    # Create group sizes for XGBoost
    group_sizes_tr = groups_tr.value_counts().sort_index().values
    group_sizes_val = groups_val.value_counts().sort_index().values

    # -------------------------------------------------------------
    # STEP 2: Build DMatrix objects
    # -------------------------------------------------------------
    logger.info("Creating XGBoost DMatrix objects…")

    def create_train_dmatrix():
        return xgb.DMatrix(X_tr_xgb, label=y_tr, group=group_sizes_tr)

    def create_val_dmatrix():
        return xgb.DMatrix(X_val_xgb, label=y_val, group=group_sizes_val)

    return create_train_dmatrix(), create_val_dmatrix(), y_val, groups_val

# Perform 5-fold cross-validation
folds = perform_train_val_split(X_train, y_train, groups_train, train)

# XGBoost parameters with all cores configured
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
    'n_jobs': N_JOBS, 
    'tree_method': 'hist'  # Keep hist for performance, it's a safe optimization
}

logger.info(f"XGBoost parameters: {xgb_params}")
logger.info(f"XGBoost will use {N_JOBS} cores for training")

# Convert scores to probabilities using sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x / 10))

# HitRate@3 calculation
def calculate_hitrate_at_k(df, k=3):
    """Calculate HitRate@k for groups with >10 options"""
    hits = []
    
    with tqdm(desc=f"Calculating HitRate@{k}", total=df['ranker_id'].nunique()) as pbar:
        for _, group in df.groupby('ranker_id'):
            if len(group) > 10:
                top_k = group.nlargest(k, 'pred')
                hit = (top_k['selected'] == 1).any()
                hits.append(hit)
            pbar.update(1)
    
    return np.mean(hits) if hits else 0.0

def evaluate_model_fold(y_true, y_pred, groups, fold_num):
    """Evaluate model performance for a specific fold"""
    logger.info(f"Evaluating Fold {fold_num+1} performance...")
    
    df = pd.DataFrame({
        'ranker_id': groups,
        'pred': y_pred,
        'selected': y_true
    })

    # Calculate HitRate@3 for groups with >10 options
    hitrate_at_3 = calculate_hitrate_at_k(df, k=3)
    
    # Get top prediction per group for other metrics
    top_preds = df.loc[df.groupby('ranker_id')['pred'].idxmax()]
    top_preds['prob'] = sigmoid(top_preds['pred'])
    
    # Calculate metrics
    logloss = log_loss(top_preds['selected'], top_preds['prob'])
    accuracy = (top_preds['selected'] == 1).mean()

    logger.info(f"Fold {fold_num+1} Metrics:")
    logger.info(f"  HitRate@3 (groups >10): {hitrate_at_3:.4f}")
    logger.info(f"  LogLoss:                {logloss:.4f}")
    logger.info(f"  Top-1 Accuracy:         {accuracy:.4f}")

    return hitrate_at_3, logloss, accuracy

fold_models = []

# Iterate through each fold
logger.info("Starting 5-fold cross-validation...")
monitor_memory()  # Initial memory check

for fold_num, (train_idx, val_idx) in enumerate(folds):
    logger.info(f"\n=== Processing Fold {fold_num+1}/5 ===")
    
    # Memory check before each fold
    current_memory = monitor_memory()
    memory_limit = target_ram_usage * 100 - 5  # 5% buffer
    if current_memory > memory_limit:
        logger.warning(f"High memory usage before fold {fold_num+1}: {current_memory:.1f}%")
        import gc
        gc.collect()  # Force garbage collection
        logger.info("Garbage collection completed")
        monitor_memory()
    
    # Split data for this fold
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    groups_tr = groups_train.iloc[train_idx]
    groups_val = groups_train.iloc[val_idx]
    
    logger.info(f"Fold {fold_num+1}: Train size={len(X_tr)}, Val size={len(X_val)}")
    
    # Prepare XGBoost data for this fold
    dtrain, dval, y_val_filtered, groups_val_filtered = prepare_xgboost_data(X_tr, X_val, y_tr, y_val, groups_tr, groups_val, X_test, cat_features_final)
    
    # Train model for this fold
    logger.info(f"Training XGBoost model for Fold {fold_num+1}...")
    start_time = datetime.now()
    fold_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False  # Reduce verbosity during CV
    )
    training_time = datetime.now() - start_time
    logger.info(f"Fold {fold_num+1} training completed in {training_time}")
    
    # Make predictions for this fold
    fold_val_preds = fold_model.predict(dval)
    
    # Evaluate this fold
    fold_hr3, fold_logloss, fold_acc = evaluate_model_fold(
        y_val_filtered, fold_val_preds, groups_val_filtered, fold_num
    )
    
    # Store results
    all_val_preds.extend(fold_val_preds)
    all_val_true.extend(y_val_filtered)
    all_val_groups.extend(groups_val_filtered)
    fold_hit_rates.append(fold_hr3)
    fold_models.append(fold_model)
    
    # Memory cleanup after each fold
    del dtrain, dval, fold_val_preds
    import gc
    gc.collect()
    logger.info(f"Fold {fold_num+1} cleanup completed")
    monitor_memory()

# Calculate overall CV performance
logger.info("\n=== Cross-Validation Results ===")
logger.info(f"Individual fold HitRate@3: {[f'{hr:.4f}' for hr in fold_hit_rates]}")
logger.info(f"Mean CV HitRate@3: {np.mean(fold_hit_rates):.4f} (+/- {np.std(fold_hit_rates)*2:.4f})")
logger.info(f"Best fold HitRate@3: {np.max(fold_hit_rates):.4f}")
logger.info(f"Worst fold HitRate@3: {np.min(fold_hit_rates):.4f}")

print("\n=== Cross-Validation Results ===")
print(f"Individual fold HitRate@3: {[f'{hr:.4f}' for hr in fold_hit_rates]}")
print(f"Mean CV HitRate@3: {np.mean(fold_hit_rates):.4f} (+/- {np.std(fold_hit_rates)*2:.4f})")
print(f"Best fold HitRate@3: {np.max(fold_hit_rates):.4f}")
print(f"Worst fold HitRate@3: {np.min(fold_hit_rates):.4f}")

# Train final model on full data for test predictions
logger.info("\nTraining final model on full training data...")

# 1. Encode categorical features for the full training and test sets
logger.info("Label encoding categorical features for final model...")
X_train_xgb_final = X_train.copy()
X_test_xgb_final = X_test.copy()

encoding_args = []
for col in cat_features_final:
    if col in X_train_xgb_final.columns:
        # For final model, val set is empty.
        encoding_args.append((col, X_train_xgb_final[col], pd.Series(dtype='object'), X_test_xgb_final[col]))

with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
    with tqdm(total=len(encoding_args), desc="Encoding final categorical features") as pbar:
        futures = {executor.submit(encode_categorical_column, args): args[0] for args in encoding_args}
        for future in as_completed(futures):
            col, encoded_tr, _, encoded_test = future.result()
            X_train_xgb_final[col] = encoded_tr
            X_test_xgb_final[col] = encoded_test
            pbar.update(1)

# 2. Create the final DMatrix for training
logger.info("Creating final XGBoost DMatrix for training...")
group_sizes_train_final = groups_train.value_counts().sort_index().values
dtrain_final = xgb.DMatrix(X_train_xgb_final, label=y_train, group=group_sizes_train_final)

# 3. Train the final model
# Use the mean of best performing parameters from CV
best_iteration = int(np.mean([model.best_iteration for model in fold_models]))
logger.info(f"Training final model for {best_iteration} rounds...")
final_model = xgb.train(
    xgb_params,
    dtrain_final,
    num_boost_round=best_iteration,
    verbose_eval=50
)

# Generate test predictions with final model
@timer
def generate_test_predictions(final_model, X_test_xgb_final, test_df):
    """Generate predictions for test set"""
    logger.info("Generating test predictions...")
    
    with tqdm(desc="Test predictions", total=3) as pbar:
        group_sizes_test = test_df.groupby('ranker_id').size().values
        pbar.update(1)
        dtest = xgb.DMatrix(X_test_xgb_final, group=group_sizes_test)
        pbar.update(1)
        test_preds = final_model.predict(dtest)
        pbar.update(1)
    
    return test_preds

test_preds = generate_test_predictions(final_model, X_test_xgb_final, test)

@timer
def create_submission(test_df, test_preds):
    """Create submission file"""
    logger.info("Creating submission file...")
    
    with tqdm(desc="Creating submission", total=3) as pbar:
        submission = test_df[['Id', 'ranker_id']].copy()
        pbar.update(1)
        submission['pred_score'] = test_preds
        pbar.update(1)
        submission['selected'] = submission.groupby('ranker_id')['pred_score'].rank(
            ascending=False, method='first'
        ).astype(int)
        pbar.update(1)
    
    # Save submission
    submission[['Id', 'ranker_id', 'selected']].to_csv('submission.csv', index=False)
    
    logger.info(f"Submission saved. Shape: {submission.shape}")
    return submission

submission = create_submission(test, test_preds)

# Display feature importance from final model
xgb_importance = final_model.get_score(importance_type='gain')
xgb_importance_df = pd.DataFrame([
    {'feature': k, 'xgb_importance': v} 
    for k, v in xgb_importance.items()
]).sort_values('xgb_importance', ascending=False)

logger.info("Top 30 most important features:")
print(xgb_importance_df.iloc[:30].to_string())

logger.info("=" * 80)
logger.info("5-fold Cross-Validation pipeline completed successfully!")
logger.info("=" * 80)
