#!/usr/bin/env python3
"""
Cache Management Helper for XGBoost FlightRank Pipeline

This script helps manage the data cache to speed up development iterations.
"""

import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Manage XGBoost FlightRank data cache")
    parser.add_argument("--clear", action="store_true", help="Clear all cached data")
    parser.add_argument("--list", action="store_true", help="List cached files")
    parser.add_argument("--size", action="store_true", help="Show cache size")
    
    args = parser.parse_args()
    
    cache_dir = Path("data_cache")
    
    if args.clear:
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            cache_dir.rmdir()
            print(f"Cleared {len(cache_files)} cache files")
        else:
            print("No cache directory found")
    
    elif args.list:
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            if cache_files:
                print(f"Found {len(cache_files)} cached files:")
                for f in sorted(cache_files):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {f.name}: {size_mb:.1f} MB")
            else:
                print("Cache directory exists but is empty")
        else:
            print("No cache directory found")
    
    elif args.size:
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            print(f"Cache size: {total_size / (1024 * 1024 * 1024):.2f} GB ({len(cache_files)} files)")
        else:
            print("No cache directory found")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 