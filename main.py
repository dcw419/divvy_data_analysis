import os
import sys
import data_processing
import analysis_ops
import algorithm
# --- 1. 路径配置 (Path Configuration) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 定义子文件夹路径
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures") # 放图片
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")   # 放Excel
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")     # 放Parquet缓存

# 🚨 开关：强制重新读取数据 (如果新增了ZIP文件，改为 True)
FORCE_RELOAD = False
# 2. 配置环境与成本常量
business_params = {
        'C_e': 6.0,    'C_c': 0.5,    # 换电/调度边际成本
        'F_e': 2.0,    'F_c': 0.5,    # 折旧成本
        'M_e': 5000,   'M_c': 5000,   # 资产规模上限
        'Q_min': 2000                 # SLA 最低底线
    }
def main():
    print("="*50)
    print("🚴 Shared Bike Strategy Analytics Pipeline 🚴")
    print("="*50)
    
    # 2. 自动创建所有必要的文件夹
    for folder in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, CACHE_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📂 Created directory: {folder}")

    # 3. ETL 阶段 (Extract, Transform, Load)
    try:
        # 注意：我们将 CACHE_DIR 传给数据处理模块，让它把缓存存在专门的地方
        df_final = data_processing.get_processed_data(DATA_DIR, CACHE_DIR, force_reload=FORCE_RELOAD)
        
        if df_final is None:
            print("❌ ETL failed. No data returned.")
            return
            
    except Exception as e:
        print(f"❌ Critical Error during Data Processing: {e}")
        return

    # 4. 分析阶段 (Analytics & Visualization)
    try:
        # 注意：我们将 OUTPUT_DIR 传进去，具体的子文件夹 (tables/figures) 在分析模块内部拼接
        # analysis_ops.analyze_user_segmentation(df_final, OUTPUT_DIR)
        # analysis_ops.analyze_tidal_flow(df_final, OUTPUT_DIR)
        # analysis_ops.analyze_asset_efficiency(df_final, OUTPUT_DIR)
        # analysis_ops.analyze_forecast_2026(df_final, OUTPUT_DIR)
        # analysis_ops.analyze_hourly_bimodal(df_final,OUTPUT_DIR,target_year=2026,target_month=1)
        # analysis_ops.analyze_station_intelligence_strategy(
        #     df_final, 
        #     OUTPUT_DIR, 
        #     target_year=2026, 
        #     target_month=1
        # )
        # analysis_ops.analyze_winter_strategy(df_final,OUTPUT_DIR,target_year=2026,target_month=1)
        # analysis_ops.analyze_asset_efficiency_detail(df_final,OUTPUT_DIR,target_year=2026,target_month=1)
        # analysis_ops.analyze_unit_economics_and_margin(df_final,OUTPUT_DIR,target_year=2026,target_month=1)
        # analysis_ops.analyze_station_kmeans_clustering(df_final,OUTPUT_DIR,target_year=2026,target_month=1)
        algorithm.run_pricing_optimization(
        raw_df=df_final, 
        current_weather=-10, 
        current_hour=8, 
        params=business_params)
    except Exception as e:
        import traceback
        traceback.print_exc() # 打印详细报错信息
        print(f"❌ Critical Error during Analysis: {e}")
        return

    print("\n" + "="*50)
    print(f"🎉 All Done!")
    print(f"📊 Excel Reports -> {TABLES_DIR}")
    print(f"📈 Chart Images  -> {FIGURES_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()

#############################
CLI后端版本
############################
# import argparse
# import os
# import time
# import analysis_ops  # 导入我们封装好的核心业务函数库

# def main():
#     # 1. 初始化命令行参数解析器
#     parser = argparse.ArgumentParser(description="🚲 Divvy 共享单车策略运营分析引擎 (CLI)")
    
#     # 2. 路径配置 (对接我们之前写的高级缓存管道)
#     parser.add_argument('--data_dir', type=str, default='./data', help='原始 zip 数据存放目录')
#     parser.add_argument('--cache_dir', type=str, default='./cache', help='Parquet 缓存文件目录')
#     parser.add_argument('--outdir', type=str, default='./output', help='图表、地图和数据看板输出目录')
    
#     # 3. 业务参数配置
#     parser.add_argument('--year', type=int, default=2023, help='要分析的目标年份 (默认: 2023)')
#     parser.add_argument('--month', type=int, default=1, help='要分析的目标月份 (默认: 1)')
    
#     # 布尔值开关：如果命令行输入了 --force_reload，则为 True，否则为 False
#     parser.add_argument('--force_reload', action='store_true', help='是否跳过缓存，强制重新清洗原始数据')
    
#     # 4. 任务选择 (加入了你要求的 regression 和刚写的 kmeans)
#     parser.add_argument('--task', type=str, default='all', 
#                         choices=['all', 'bimodal', 'efficiency', 'imbalance', 'ue', 'kmeans', 'regression'], 
#                         help='选择要执行的单一分析模块 (默认: all)')

#     # 5. 解析用户在终端输入的命令
#     args = parser.parse_args()

#     # --- 终端 UI 打印 ---
#     print("\n" + "="*50)
#     print(f"🚀 启动 Divvy 策略分析后端引擎...")
#     print(f"📅 目标时间: {args.year}年 {args.month}月")
#     print(f"🎯 执行任务: {args.task.upper()}")
#     print("="*50)

#     # ==========================================
#     # 核心修复：调用你的“智能缓存数据管道”，而不是死板地读单个 CSV
#     # ==========================================
#     df = analysis_ops.get_processed_data(
#         data_dir=args.data_dir, 
#         cache_dir=args.cache_dir, 
#         force_reload=args.force_reload
#     )
    
#     if df is None or len(df) == 0:
#         print("❌ 致命错误：数据加载失败或数据为空，引擎终止。")
#         return

#     # ==========================================
#     # 任务路由分发 (Task Router)
#     # ==========================================
#     start_time = time.time()
    
#     if args.task in ['all', 'bimodal']:
#         # 由于你原本的 bimodal 没有加 year/month 参数，这里为了兼容可以先不传
#         analysis_ops.analyze_hourly_bimodal(df, args.outdir)
        
#     if args.task in ['all', 'efficiency']:
#         analysis_ops.analyze_asset_efficiency_detail(df, args.outdir, target_year=args.year, target_month=args.month)
        
#     if args.task in ['all', 'imbalance']:
#         analysis_ops.analyze_station_intelligence_strategy(df, args.outdir, target_year=args.year, target_month=args.month)
        
#     if args.task in ['all', 'ue']:
#         analysis_ops.analyze_unit_economics_and_margin(df, args.outdir, target_year=args.year, target_month=args.month)
        
#     if args.task in ['all', 'kmeans']:
#         # 调用刚写好的 K-Means 站点聚类和 Folium 地图
#         analysis_ops.analyze_station_kmeans_clustering(df, args.outdir, target_year=args.year, target_month=args.month)
        
#     if args.task in ['regression']:
#         print("\n[Analysis] Running OLS Regression...")
#         print("💡 提示：这里需要你在 analysis_ops.py 中补充统计学回归代码，目前为占位符。")

#     print("\n" + "="*50)
#     print(f"🎉 全部任务执行完毕！总耗时: {time.time() - start_time:.2f} 秒")
#     print(f"📂 产出物已保存至: {os.path.abspath(args.outdir)}")
#     print("="*50 + "\n")

# if __name__ == "__main__":
#     main()
