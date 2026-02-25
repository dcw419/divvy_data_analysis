import pandas as pd
import zipfile
import os
import glob
import time

import pandas as pd
import zipfile
import os
import glob
import time

# ==========================================
# 🚨 核心修改 1：将空间数据（经纬度）加入白名单
# ==========================================
STRATEGY_COLS = [
    'ride_id', 'rideable_type', 'started_at', 'ended_at', 
    'start_station_name', 'end_station_name', 'member_casual',
    'start_lat', 'start_lng'  # <--- 新增：这决定了你能不能画出牛逼的 GIS 地图
]

def load_raw_data(data_dir):
    print(f"   [Loader] Scanning raw files in: {data_dir}")
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    zip_files.sort()
    
    if not zip_files:
        print("   ❌ No .zip files found!")
        return None

    df_list = []
    for f in zip_files:
        try:
            with zipfile.ZipFile(f, 'r') as z:
                csv_name = [n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__')][0]
                with z.open(csv_name) as file:
                    temp_df = pd.read_csv(file, usecols=STRATEGY_COLS, parse_dates=['started_at', 'ended_at'])
                    df_list.append(temp_df)
                    print(f"   -> Loaded: {os.path.basename(f)} | Rows: {len(temp_df):,}")
        except Exception as e:
            print(f"   -> ⚠️ Skipped {f}: {e}")

    if not df_list:
        return None
    return pd.concat(df_list, ignore_index=True)

def clean_data(df):
    print(f"   [Cleaner] Cleaning {len(df):,} rows...")
    df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    
    # ==========================================
    # 🚨 核心修改 2：过滤 GPS 缺失的脏数据
    # ==========================================
    mask = (df['duration_min'] >= 1) & (df['duration_min'] <= 1440) & \
           (df['start_station_name'].notna()) & (df['end_station_name'].notna()) & \
           (df['start_lat'].notna()) & (df['start_lng'].notna()) # <--- 必须有经纬度才能画图
           
    df_clean = df.loc[mask].copy()
    
    # 注意：start_lat 和 start_lng 必须保持 Float 浮点数，不能转 category
    for col in ['rideable_type', 'member_casual', 'start_station_name', 'end_station_name']:
        df_clean[col] = df_clean[col].astype('category')
        
    return df_clean

def get_processed_data(data_dir, cache_dir, force_reload=False):
    """
    智能数据加载器：将缓存存入 cache_dir
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_path = os.path.join(cache_dir, "cached_data.parquet")
    
    if os.path.exists(cache_path) and not force_reload:
        print(f"\n[⚡ Cache Hit] Found cached data: {cache_path}")
        try:
            start_time = time.time()
            df = pd.read_parquet(cache_path)
            print(f"   ✅ Data Loaded in {time.time()-start_time:.2f}s! Rows: {len(df):,}")
            return df
        except Exception as e:
            print(f"   ⚠️ Cache corrupted: {e}. Reloading raw data...")
    
    print(f"\n[🐢 Cache Miss] Loading from raw sources (This might take a while)...")
    raw_df = load_raw_data(data_dir)
    
    if raw_df is not None:
        clean_df = clean_data(raw_df)
        print(f"   💾 Saving cache to: {cache_path}")
        clean_df.to_parquet(cache_path, index=False)
        print("   ✅ Cache created successfully.")
        return clean_df
    return None
