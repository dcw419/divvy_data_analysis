import argparse
import pandas as pd
import os
# 假设你之前写的那些画图函数都保存在 analysis_ops.py 中
import analysis_ops 

def main():
    # 1. 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description="🚲 Divvy 共享单车商业分析后端引擎")
    
    # 2. 定义你可以从外部传入的参数
    parser.add_argument('--file', type=str, default='202601-divvy-tripdata.csv', help='CSV 数据文件的路径')
    parser.add_argument('--year', type=int, default=2026, help='要分析的年份 (默认: 2026)')
    parser.add_argument('--month', type=int, default=1, help='要分析的月份 (默认: 1)')
    parser.add_argument('--task', type=str, default='all', 
                        choices=['all', 'bimodal', 'efficiency', 'imbalance', 'ue'], 
                        help='选择要执行的分析任务 (默认: all)')
    parser.add_argument('--outdir', type=str, default='./figures', help='图表输出目录')

    # 3. 解析用户在终端输入的命令
    args = parser.parse_args()

    print("="*50)
    print(f"🚀 启动后端分析引擎...")
    print(f"📊 目标数据: {args.year}年 {args.month}月")
    print(f"执行任务: {args.task.upper()}")
    print("="*50)

    # 4. 加载数据
    if not os.path.exists(args.file):
        print(f"❌ 错误: 找不到数据文件 {args.file}")
        return
        
    print(f"正在加载数据 {args.file}，请稍候...")
    df = pd.read_csv(args.file)
    print(f"✅ 数据加载成功，共 {len(df):,} 条记录。")

    # 5. 根据传入的 --task 参数，选择性调用不同的函数
    if args.task in ['all', 'bimodal']:
        analysis_ops.analyze_hourly_bimodal(df, args.outdir)
        
    if args.task in ['all', 'efficiency']:
        analysis_ops.analyze_asset_efficiency(df, args.outdir, target_year=args.year, target_month=args.month)
        
    if args.task in ['all', 'imbalance']:
        analysis_ops.analyze_station_intelligence_strategy(df, args.outdir, target_year=args.year, target_month=args.month)
        
    if args.task in ['all', 'ue']:
        # 这是我们之前写的单体经济模型/毛利分析
        analysis_ops.analyze_unit_economics_and_margin(df, args.outdir, target_year=args.year, target_month=args.month)

    print("\n🎉 全部后端任务执行完毕！")

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
