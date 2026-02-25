import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
from prophet import Prophet 
import numpy as np
from scipy.interpolate import make_interp_spline
# 筛选特定的时间分析
# ==========================================
# 🛠️ 通用工具函数 (Helper Function)
# ==========================================
def filter_data_by_period(df, year=None, month=None):
    """
    通用工具: 根据年份和月份筛选数据
    参数:
        year: int (可选), e.g., 2026
        month: int (可选), e.g., 1
    返回:
        筛选后的 DataFrame
    """
    # 1. 确保是时间格式 (这是最基础的防御性编程)
    if not pd.api.types.is_datetime64_any_dtype(df['started_at']):
        df = df.copy()
        df['started_at'] = pd.to_datetime(df['started_at'])
    
    # 2. 构建筛选条件
    mask = pd.Series([True] * len(df), index=df.index) # 默认全选
    
    label = "All Data"
    
    if year:
        mask &= (df['started_at'].dt.year == year)
        label = f"{year}年"
        
    if month:
        mask &= (df['started_at'].dt.month == month)
        label += f"{month}月"
        
    # 3. 执行筛选
    df_filtered = df.loc[mask].copy()
    
    # 4. 打印日志 (让你知道发生了什么)
    print(f"\n🔍 [Data Filter] Target: {label}")
    if len(df_filtered) == 0:
        print(f"   ⚠️ 警告: 该时间段无数据！(Rows: 0)")
        print("   ⚠️ 自动回退: 使用原始全量数据进行演示。")
        return df # 兜底策略：如果没有数据，返回原表，防止后面报错
    else:
        print(f"   ✅ 成功锁定: {len(df_filtered):,} 条订单")
        return df_filtered
    
# ==========================================
# 0. 中文显示配置 & 风格
# ==========================================
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')
sns.set_palette("husl")

# ==========================================
# 辅助函数：自动分类保存 (Helper Function)
# ==========================================
def _save(fig, data, name, output_dir):
    """
    自动将图片存入 figures 文件夹，数据存入 tables 文件夹
    """
    # 1. 路径拼接
    fig_dir = os.path.join(output_dir, "figures")
    tbl_dir = os.path.join(output_dir, "tables")
    
    # 2. 保存图片
    if fig:
        save_path = os.path.join(fig_dir, f"{name}.png")
        # 使用 bbox_inches='tight' 防止标题被切
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"   📈 Image saved: figures/{name}.png")
    
    # 3. 保存Excel
    if data is not None:
        excel_path = os.path.join(tbl_dir, f"Data_{name}.xlsx")
        data.to_excel(excel_path, index=False)
        print(f"   📊 Table saved: tables/Data_{name}.xlsx")

# ==========================================
# 业务分析函数
# ==========================================

def analyze_user_segmentation(df, output_dir):
    print("\n[Analysis 1] User Segmentation...")
    df['day_name'] = df['started_at'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    stats = df.groupby(['day_name', 'member_casual'], observed=True).size().reset_index(name='ride_count')
    stats['day_name'] = pd.Categorical(stats['day_name'], categories=days_order, ordered=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=stats, x='day_name', y='ride_count', hue='member_casual', ax=ax)
    ax.set_title('Weekly User Behavior: Member vs Casual (用户分层行为)')
    
    _save(fig, stats, "01_User_Segmentation", output_dir)

def analyze_tidal_flow(df, output_dir):
    print("\n[Analysis 2] Tidal Flow (7AM-9AM)...")
    morning_df = df[df['started_at'].dt.hour.between(7, 9)]
    
    outflow = morning_df['start_station_name'].value_counts().reset_index()
    outflow.columns = ['station', 'out_count']
    inflow = morning_df['end_station_name'].value_counts().reset_index()
    inflow.columns = ['station', 'in_count']
    
    flow = pd.merge(outflow, inflow, on='station', how='outer').fillna(0)
    flow['net_flow'] = flow['in_count'] - flow['out_count']
    
    top_deficit = flow.sort_values('net_flow').head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_deficit, x='net_flow', y='station', palette='Reds_r', ax=ax)
    ax.set_title('Top 10 Deficit Stations (Morning Rush) - 缺车站点分析')
    ax.axvline(0, color='black')
    
    _save(fig, top_deficit, "02_Tidal_Flow_Deficit", output_dir)

def analyze_asset_efficiency(df, output_dir):
    print("\n[Analysis 3] Asset Efficiency...")
    stats = df.groupby('rideable_type', observed=True).agg(
        Total_Rides=('ride_id', 'count'),
        Avg_Duration=('duration_min', 'mean')
    ).reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=stats, x='rideable_type', y='Total_Rides', ax=ax1, color='skyblue', alpha=0.6)
    ax1.set_ylabel('Ride Volume')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=stats, x='rideable_type', y='Avg_Duration', ax=ax2, color='red', marker='o', linewidth=3)
    ax2.set_ylabel('Avg Duration (min)')
    
    ax1.set_title('Asset Efficiency: Volume vs Duration (资产效率)')
    
    _save(fig, stats, "03_Asset_Efficiency", output_dir)

def analyze_forecast_2026(df, output_dir):
    print("\n[Analysis 4] Forecasting 2026 (Prophet)...")
    
    # 1. Prophet 数据准备
    daily_rides = df.groupby(df['started_at'].dt.date).size().reset_index(name='y')
    daily_rides['ds'] = pd.to_datetime(daily_rides['started_at'])
    daily_rides = daily_rides[['ds', 'y']]
    
    # 2. 建模
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name='US') 
    m.fit(daily_rides)
    
    # 3. 预测
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    
    # 4. 准备 Excel 数据
    export_df = pd.DataFrame()
    export_df['Date'] = forecast['ds']
    export_df['Day_Name'] = forecast['ds'].dt.day_name()
    export_df['Predicted_Rides'] = forecast['yhat'].round(0)
    export_df['Trend'] = forecast['trend'].round(0)
    
    mask_2026 = export_df['Date'] >= '2026-01-01'
    final_excel = export_df.loc[mask_2026]
    
    # 保存数据 (调用辅助函数，但这里不需要传fig，因为Prophet图比较特殊)
    _save(None, final_excel, "04_Forecast_2026", output_dir)

    # 5. 特殊绘图处理 (Prophet 的图需要单独处理保存)
    
    # 图 1: 趋势图
    fig1 = m.plot(forecast)
    ax1 = fig1.gca()
    ax1.set_title('2026 Demand Forecast (Total Volume) - 总量预测', fontsize=16, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rides')
    
    # 手动保存到 figures 文件夹
    fig1.savefig(os.path.join(output_dir, "figures", "04_Forecast_Trend.png"), 
                 dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("   📈 Image saved: figures/04_Forecast_Trend.png")

    # 图 2: 成分图
    fig2 = m.plot_components(forecast)
    fig2.suptitle('Forecast Components (成分分解): Trend, Weekly & Yearly', 
                  fontsize=18, fontweight='bold', y=0.98)
    
    # 给子图加中文标题
    axes = fig2.get_axes()
    for ax in axes:
        y_label = ax.get_ylabel()
        if 'trend' in y_label:
            ax.set_title('1. Long-Term Growth (长期趋势)', loc='left', fontsize=12, fontweight='bold')
        elif 'holidays' in y_label:
            ax.set_title('2. Holiday Effects (节假日效应)', loc='left', fontsize=12, fontweight='bold')
        elif 'weekly' in y_label:
            ax.set_title('3. Weekly Pattern (周度周期)', loc='left', fontsize=12, fontweight='bold')
        elif 'yearly' in y_label:
            ax.set_title('4. Yearly Seasonality (年度季节性)', loc='left', fontsize=12, fontweight='bold')

    # 布局调整
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 手动保存到 figures 文件夹
    fig2.savefig(os.path.join(output_dir, "figures", "04_Forecast_Components.png"), 
                 dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("   📈 Image saved: figures/04_Forecast_Components.png")


def analyze_hourly_bimodal(df, output_dir):
    """
    分析 5: 日均小时级双峰效应 (Average Hourly Bimodal Pattern)
    逻辑升级：从“历史总和”改为“日均单量”，更符合业务直觉
    """
    print("\n[Analysis 5] Generating Daily Average Hourly Pattern...")
    
    # 1. 特征工程
    df['hour'] = df['started_at'].dt.hour
    df['date'] = df['started_at'].dt.date # 拿到具体日期 (e.g., 2023-01-01)
    df['day_type'] = df['started_at'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # ==========================================
    # 核心修改：计算逻辑变了！
    # ==========================================
    
    # 第一步：算出“每一天、每个小时”的具体单量
    # 结果类似：
    # 2023-01-01 (Weekend) | 8点 | member | 15单
    # 2023-01-02 (Weekday) | 8点 | member | 200单
    daily_hourly_counts = df.groupby(['date', 'day_type', 'member_casual', 'hour'], observed=True).size().reset_index(name='count')
    
    # 第二步：算出“平均值”
    # 也就是把所有 Weekday 的 8点数据拿来求 Mean
    avg_stats = daily_hourly_counts.groupby(['day_type', 'member_casual', 'hour'], observed=True)['count'].mean().reset_index()
    
    # ==========================================
    # 绘图逻辑 (保持平滑曲线)
    # ==========================================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # 定义画图函数 (复用之前的逻辑，只是输入变成了平均值)
    def plot_smooth_line(data, ax, title):
        for user_type in ['member', 'casual']:
            subset = data[data['member_casual'] == user_type]
            
            # 排序确保 0-23 顺序正确
            subset = subset.sort_values('hour')
            
            x = subset['hour'].values
            y = subset['count'].values # 这里已经是平均值了
            
            if len(x) > 3: # 数据点够多才做平滑
                x_new = np.linspace(x.min(), x.max(), 300)
                try:
                    spl = make_interp_spline(x, y, k=3)
                    y_smooth = spl(x_new)
                    y_smooth = y_smooth.clip(min=0)
                    ax.plot(x_new, y_smooth, label=user_type, linewidth=3)
                    ax.fill_between(x_new, y_smooth, alpha=0.2)
                except:
                    ax.plot(x, y, label=user_type, marker='o')
            else:
                ax.plot(x, y, label=user_type, marker='o')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day (0-23)')
        ax.set_ylabel('Average Rides (Daily)') # Y轴标签变了
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_smooth_line(avg_stats[avg_stats['day_type'] == 'Weekday'], axes[0], 'Weekday: Avg Commute (工作日日均)')
    plot_smooth_line(avg_stats[avg_stats['day_type'] == 'Weekend'], axes[1], 'Weekend: Avg Leisure (周末日均)')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(output_dir, "figures", "05_Hourly_Average_Pattern.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"   📈 Image saved: figures/05_Hourly_Average_Pattern.png")
    
    # 保存 Excel 供查阅
    avg_stats.to_excel(os.path.join(output_dir, "tables", "Data_05_Hourly_Average.xlsx"), index=False)
# # def analyze_hourly_bimodal(df, output_dir):
#     """
#     分析 5: 小时级双峰效应 (Hourly Bimodal Pattern)
#     目标: 揭示一天 24 小时内的流量脉冲，验证“通勤双峰”
#     """
#     print("\n[Analysis 5] Hourly Bimodal Pattern...")
    
#     # 1. 数据特征工程
#     # 提取小时 (0-23)
#     df['hour'] = df['started_at'].dt.hour
    
#     # 区分 工作日 vs 周末
#     # dt.dayofweek: 0=Mon, 4=Fri, 5=Sat, 6=Sun
#     df['day_type'] = df['started_at'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
#     # 2. 聚合数据
#     # 我们需要算“平均每小时单量”，为了平滑数据，我们先按小时算总和
#     hourly_stats = df.groupby(['day_type', 'member_casual', 'hour'], observed=True).size().reset_index(name='ride_count')
    
#     # 3. 绘图 (创建两个子图：左边是工作日，右边是周末)
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
#     # --- 左图：工作日 (Weekday) ---
#     sns.lineplot(
#         data=hourly_stats[hourly_stats['day_type'] == 'Weekday'],
#         x='hour', y='ride_count', hue='member_casual',
#         ax=axes[0], linewidth=3, marker='o'
#     )
#     axes[0].set_title('Weekday: Commute Peaks (工作日：通勤双峰)', fontsize=14, fontweight='bold')
#     axes[0].set_xlabel('Hour of Day (0-23)')
#     axes[0].set_ylabel('Ride Volume')
#     axes[0].set_xticks(range(0, 24, 2)) # 每2小时显示一个刻度
#     axes[0].grid(True, alpha=0.3)
    
#     # --- 右图：周末 (Weekend) ---
#     sns.lineplot(
#         data=hourly_stats[hourly_stats['day_type'] == 'Weekend'],
#         x='hour', y='ride_count', hue='member_casual',
#         ax=axes[1], linewidth=3, marker='o'
#     )
#     axes[1].set_title('Weekend: Leisure Curve (周末：休闲单峰)', fontsize=14, fontweight='bold')
#     axes[1].set_xlabel('Hour of Day (0-23)')
#     axes[1].set_xticks(range(0, 24, 2))
#     axes[1].grid(True, alpha=0.3)
    
#     # 4. 保存
#     # 使用 tight_layout 防止重叠
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, "figures", "05_Hourly_Bimodal_Pattern.png")
#     # 如果 figures 文件夹不存在，存到 output_dir
#     if not os.path.exists(os.path.dirname(save_path)):
#         save_path = os.path.join(output_dir, "05_Hourly_Bimodal_Pattern.png")
        
#     fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
#     print(f"   📈 Image saved: {save_path}")
    
#     # 导出 Excel 方便看具体数值
#     excel_path = os.path.join(output_dir, "tables", "Data_05_Hourly_Stats.xlsx")
#     if not os.path.exists(os.path.dirname(excel_path)):
#          excel_path = os.path.join(output_dir, "Data_05_Hourly_Stats.xlsx")
         
#     hourly_stats.to_excel(excel_path, index=False)
#     print(f"   📊 Table saved: {excel_path}")



# ==========================================
# 🧠 核心策略分析函数
# ==========================================
def analyze_station_intelligence_strategy(df, output_dir, target_year=2026, target_month=1):
    """
    高级策略分析: 站点画像与智能调度算法 (Station Intelligence)
    【修复版】解决了 Categorical 类型无法 fillna(0) 的报错
    """
    # 1. 调用工具函数筛选数据
    df_target = filter_data_by_period(df, year=target_year, month=target_month)
    
    # ==========================================
    # 2. 特征工程 (Feature Engineering)
    # ==========================================
    df_target['hour'] = df_target['started_at'].dt.hour
    df_target['is_weekend'] = df_target['started_at'].dt.dayofweek >= 5
    
    # 过滤掉没有站点名称的“幽灵订单”
    valid_trips = df_target[df_target['start_station_name'].notna() & df_target['end_station_name'].notna()].copy()
    
    # A. 计算出发特征 (Outflow Profile)
    station_stats = valid_trips.groupby('start_station_name', observed=True).agg(
        Total_Outflow=('ride_id', 'count'),
        AM_Peak_Outflow=('hour', lambda x: ((x >= 7) & (x <= 9)).sum()),
        Weekend_Outflow=('is_weekend', 'sum'),
        Avg_Duration=('duration_min', 'mean')
    ).reset_index()
    
    # B. 计算到达特征 (Inflow Profile)
    inflow_stats = valid_trips.groupby('end_station_name', observed=True).size().reset_index(name='Total_Inflow')
    
    # ==========================================
    # 🚨 核心修复：类型转换 (Fixing TypeError)
    # ==========================================
    # 在合并前，把 category 类型转为 object，防止 fillna(0) 报错
    station_stats['start_station_name'] = station_stats['start_station_name'].astype('object')
    inflow_stats['end_station_name'] = inflow_stats['end_station_name'].astype('object')
    
    # C. 合并画像表 (Merge)
    # outer join 会产生很多 NaN
    station_profile = pd.merge(station_stats, inflow_stats, left_on='start_station_name', right_on='end_station_name', how='outer')
    
    # 智能合并名字：如果 start 是 NaN (只有到达)，就取 end 的名字
    station_profile['Station_Name'] = station_profile['start_station_name'].fillna(station_profile['end_station_name'])
    
    # 现在已经是 object 类型了，可以安全地填充 0
    station_profile = station_profile.fillna(0)
    
    # 计算比例指标 (加1防止除零)
    station_profile['AM_Ratio'] = station_profile['AM_Peak_Outflow'] / (station_profile['Total_Outflow'] + 1)
    station_profile['Weekend_Ratio'] = station_profile['Weekend_Outflow'] / (station_profile['Total_Outflow'] + 1)

    # ==========================================
    # 3. 核心策略：标签体系生成 (Tagging System)
    # ==========================================
    print("   🏷️  Generating Station Tags (CDP)...")
    
    def get_tags(row):
        tags = []
        # --- 流量分级 ---
        # 如果所有站点都是0 (极端情况)，分位数计算可能会有问题，加个保护
        threshold_hot = station_profile['Total_Outflow'].quantile(0.9)
        if threshold_hot > 0 and row['Total_Outflow'] > threshold_hot: 
            tags.append('🔥核心热点')
        elif row['Total_Outflow'] < 5: 
            tags.append('❄️僵尸点')
            
        # --- 业务属性 ---
        if row['AM_Ratio'] > 0.25: tags.append('🏠通勤-住宅')
        if row['Weekend_Ratio'] > 0.40: tags.append('🌳休闲-景区')
        if row['Avg_Duration'] < 10: tags.append('⚡️短途刚需')
        
        # --- 异常状态 ---
        if row['Total_Outflow'] == 0 and row['Total_Inflow'] > 10:
            tags.append('⚠️只进不出(淤积)')
            
        return ",".join(tags) if tags else "普通站点"

    station_profile['Station_Tags'] = station_profile.apply(get_tags, axis=1)

    # ==========================================
    # 4. 核心算法：智能调度指令 (Smart Rebalancing)
    # ==========================================
    station_profile['Net_Flow'] = station_profile['Total_Inflow'] - station_profile['Total_Outflow']
    
    # 🔴 红包车策略
    red_packet_list = station_profile[
        (station_profile['Net_Flow'] > 10) & 
        (station_profile['Station_Tags'].str.contains('热点|住宅'))
    ].sort_values('Net_Flow', ascending=False)
    
    # 🔵 调度车策略
    truck_dispatch_list = station_profile[
        station_profile['Net_Flow'] < -10
    ].sort_values('Net_Flow', ascending=True)

    # ==========================================
    # 5. 保存结果与可视化
    # ==========================================
    file_suffix = f"{target_year}{target_month:02d}"
    
    # --- 保存 Excel ---
    excel_name = f"07_Strategy_Ops_{file_suffix}.xlsx"
    save_path = os.path.join(output_dir, "tables", excel_name)
    if not os.path.exists(os.path.dirname(save_path)): save_path = os.path.join(output_dir, excel_name)
    
    # 只保留关键列，让表格更干净
    cols_to_save = ['Station_Name', 'Station_Tags', 'Net_Flow', 'Total_Outflow', 'Total_Inflow', 'AM_Ratio', 'Weekend_Ratio', 'Avg_Duration']
    # 确保列存在 (防止改名失败等意外)
    cols_final = [c for c in cols_to_save if c in station_profile.columns]
    
    station_profile[cols_final].to_excel(save_path, index=False)
    print(f"   ✅ Strategy Table saved: {save_path}")

    # --- 可视化 ---
# === 在 analyze_station_intelligence_strategy 的画图部分替换为以下代码 ===
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=station_profile, x='Total_Outflow', y='Total_Inflow', color='grey', alpha=0.3, s=30)
    
    if not red_packet_list.empty:
        sns.scatterplot(data=red_packet_list.head(20), x='Total_Outflow', y='Total_Inflow', color='red', s=100)
        # 🚨 新增：标注淤积 Top 3 的名字 + 具体淤积数量
        for idx, row in red_packet_list.head(3).iterrows():
            plt.annotate(f"{row['Station_Name']}\n(淤积 +{int(row['Net_Flow'])})",
                         xy=(row['Total_Outflow'], row['Total_Inflow']),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='darkred',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7))

    if not truck_dispatch_list.empty:
        sns.scatterplot(data=truck_dispatch_list.head(20), x='Total_Outflow', y='Total_Inflow', color='blue', s=100)
        # 🚨 新增：标注缺车 Top 3 的名字 + 具体缺口数量
        for idx, row in truck_dispatch_list.head(3).iterrows():
            plt.annotate(f"{row['Station_Name']}\n(缺口 {int(row['Net_Flow'])})",
                         xy=(row['Total_Outflow'], row['Total_Inflow']),
                         xytext=(10, -20), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='darkblue',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", alpha=0.7))

    limit = max(station_profile['Total_Outflow'].max(), station_profile['Total_Inflow'].max()) + 10
    plt.plot([0, limit], [0, limit], '--', color='black', alpha=0.5, label='Perfect Balance (In = Out)')
    # ... 其他标题和保存逻辑 ...
    
    plt.title(f'Station Intelligence ({target_year}-{target_month}): Supply vs Demand', fontsize=14)
    plt.xlabel('Demand (Outflow)')
    plt.ylabel('Supply (Inflow)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    img_name = f"07_Station_Imbalance_{file_suffix}.png"
    img_save_path = os.path.join(output_dir, "figures", img_name)
    if not os.path.exists(os.path.dirname(img_save_path)): img_save_path = os.path.join(output_dir, img_name)
        
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    print(f"   📈 Chart saved: {img_save_path}")
    
    # --- 终端输出 ---
    print("\n" + "="*50)
    print(f"🤖 [AI Strategy] {target_year}-{target_month} Operational Directives")
    print("="*50)
    print(f"🔴 [红包车] 建议开启数量: {len(red_packet_list)} 个站点")
    if not red_packet_list.empty:
        top_s = red_packet_list.iloc[0]
        print(f"   Top 1: {top_s['Station_Name']} (积压 +{int(top_s['Net_Flow'])})")
    
    print(f"\n🔵 [调度车] 建议补货数量: {len(truck_dispatch_list)} 个站点")
    if not truck_dispatch_list.empty:
        top_d = truck_dispatch_list.iloc[0]
        print(f"   Top 1: {top_d['Station_Name']} (缺口 {int(top_d['Net_Flow'])})")
    print("="*50)

def analyze_winter_strategy(df, output_dir, target_year=2026, target_month=1):
    """
    策略分析: 针对特定月份（如冬季/淡季）的精细化运营分析
    通用化改造：支持传入任意年份和月份
    """
    # 动态生成报告标题
    report_title = f"{target_year}年{target_month}月"
    print(f"\n[Strategy Ops] Generating Strategy Report for {report_title}...")

    # 1. 调用通用函数筛选数据 (解耦核心)
    # 这会复用我们之前写好的筛选逻辑，如果数据不存在会自动处理
    df_target = filter_data_by_period(df, year=target_year, month=target_month)

    # ==========================================
    # 场景 A: 找出适合推 "短途一口价" 的散客
    # ==========================================
    short_trips = df_target[
        (df_target['member_casual'] == 'casual') & 
        (df_target['duration_min'] <= 10)
    ]
    
    opportunity_size = len(short_trips)
    total_casual = len(df_target[df_target['member_casual'] == 'casual'])
    
    print(f"\n1️⃣ [定价策略] 短途散客分析 ({report_title})")
    if total_casual > 0:
        ratio = opportunity_size / total_casual
        print(f"   - 散客总单量: {total_casual:,}")
        print(f"   - 10分钟内短途单: {opportunity_size:,} (占比 {ratio:.1%})")
        print(f"   💡 策略建议: 针对这 {ratio:.1%} 的刚需人群，推出 '10分钟 $1' 一口价。")
    else:
        print("   ⚠️ 该月份没有散客数据，无法计算转化率。")

    # ==========================================
    # 场景 B: 找出适合推 "电单车升舱" 的用户
    # ==========================================
    # 逻辑: 还能坚持骑 经典车 > 15分钟 的用户
    hardcore_users = df_target[
        (df_target['rideable_type'] == 'classic_bike') & 
        (df_target['duration_min'] > 15)
    ]
    
    upsell_pool = len(hardcore_users)
    
    print(f"\n2️⃣ [推荐策略] 电单车升舱潜客分析")
    print(f"   - 长途经典车用户: {upsell_pool:,}")
    print(f"   💡 策略建议: 向这 {upsell_pool:,} 名用户推送 '电单车体验券'，主打省力痛点。")

    # ==========================================
    # 可视化: 时长分布图
    # ==========================================
   # === 在 analyze_winter_strategy 的画图部分替换为以下代码 ===
    plt.figure(figsize=(10, 6))
    
    # 画直方图
    ax = sns.histplot(data=df_target, x='duration_min', hue='member_casual', 
                      element="step", bins=range(0, 60, 2))
    
    plt.title(f'Trip Duration Distribution ({report_title})', fontsize=14, fontweight='bold')
    plt.xlabel('Trip Duration (minutes)')
    plt.ylabel('Ride Volume')
    plt.axvline(10, color='red', linestyle='--', linewidth=2, label='10min Threshold')
    
    # 🚨 新增：计算比例并添加带箭头的醒目文本框
    if total_casual > 0:
        ratio = opportunity_size / total_casual
        # 获取 Y 轴最大值，用来定位文本框的高度
        max_y = ax.get_ylim()[1] 
        
        plt.annotate(
            f'⭐ 核心刚需洞察:\n散客群体中有 {ratio:.1%} 的订单\n骑行时间不到 10 分钟',
            xy=(5, max_y * 0.5),         # 箭头指到的位置 (X=5分钟, Y=一半高度)
            xytext=(15, max_y * 0.7),    # 文本框所在位置 (X=15分钟处)
            arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
            fontsize=12, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffeaea", ec="red", lw=1.5, alpha=0.9)
        )
    
    plt.legend()
    # ... 保存图表 ...
    
    # 动态生成文件名 (防止覆盖)
    file_suffix = f"{target_year}{target_month:02d}"
    img_name = f"06_Strategy_Duration_{file_suffix}.png"
    
    save_path = os.path.join(output_dir, "figures", img_name)
    # 容错：如果 figures 文件夹不存在，存到上一级
    if not os.path.exists(os.path.dirname(save_path)):
        save_path = os.path.join(output_dir, img_name)
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   📈 Chart saved: {save_path}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_asset_efficiency_detail(df, output_dir, target_year=2026, target_month=1):
    print(f"\n[Analysis 3] Asset Efficiency ({target_year}-{target_month:02d})...")
    
    # 1. 核心改动：精准锁定 2026 年 1 月的数据
    if not pd.api.types.is_datetime64_any_dtype(df['started_at']):
        df['started_at'] = pd.to_datetime(df['started_at'])
        
    mask = (df['started_at'].dt.year == target_year) & (df['started_at'].dt.month == target_month)
    df_target = df.loc[mask].copy()
    
    if len(df_target) == 0:
        print(f"   ⚠️ 警告：没有找到 {target_year}年{target_month}月 的数据！自动回退使用全量数据。")
        df_target = df.copy()
    else:
        print(f"   ✅ 成功提取 {len(df_target):,} 条订单进行资产分析。")

    # 2. 聚合计算 (使用筛选后的 df_target)
    stats = df_target.groupby('rideable_type', observed=True).agg(
        Total_Rides=('ride_id', 'count'),
        Avg_Duration=('duration_min', 'mean')
    ).reset_index()
    
    # 3. 绘制双轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左轴：单量 (柱状图)
    sns.barplot(data=stats, x='rideable_type', y='Total_Rides', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_ylabel('Ride Volume (Total Rides)', fontweight='bold')
    
    # 右轴：时长 (折线图)
    ax2 = ax1.twinx()
    sns.lineplot(data=stats, x='rideable_type', y='Avg_Duration', ax=ax2, color='red', marker='o', markersize=10, linewidth=3)
    ax2.set_ylabel('Avg Duration (min)', color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 动态标题
    ax1.set_title(f'Asset Efficiency: Volume vs Duration ({target_year}-{target_month:02d} 资产效率)', fontsize=14, fontweight='bold')
    
    # 4. 动态文件名保存 (加上年月后缀，防止覆盖历史全量图表)
    file_suffix = f"{target_year}{target_month:02d}"
    filename = f"03_Asset_Efficiency_{file_suffix}"
    
    
    _save(fig, stats, filename, output_dir)


def analyze_unit_economics_and_margin(df, output_dir, target_year=2026, target_month=1):
    """
    商业高级分析：单体经济模型 (UE) 与毛利测算
    目标：计算真实定价下的单均营收、单均成本和最终毛利率
    """
    print(f"\n[Analysis - Unit Economics] Running Financial Model ({target_year}-{target_month:02d})...")
    
    # 1. 筛选时间范围
    if not pd.api.types.is_datetime64_any_dtype(df['started_at']):
        df['started_at'] = pd.to_datetime(df['started_at'])
        
    mask = (df['started_at'].dt.year == target_year) & (df['started_at'].dt.month == target_month)
    df_target = df.loc[mask].copy()
    
    if len(df_target) == 0:
        print("   ⚠️ 未找到指定月份数据，使用全量数据进行财务模拟。")
        df_target = df.copy()

    # 清洗：计算时长并过滤异常值
    df_target['ended_at'] = pd.to_datetime(df_target['ended_at'])
    df_target['duration_min'] = (df_target['ended_at'] - df_target['started_at']).dt.total_seconds() / 60
    df_clean = df_target[(df_target['duration_min'] >= 1) & (df_target['duration_min'] <= 1440)].copy()

    # ==========================================
    # 2. 构建财务模型 (The Financial Engine)
    # ==========================================
    # --- 定价规则 (Revenue) ---
    base_price = 1.50      # 前 15 分钟一口价
    free_minutes = 15      # 免费/起步时长
    overtime_rate = 0.50   # 超过 15 分钟后，每分钟的计费
    
    # 核心公式：计算每单营收
    df_clean['Revenue'] = np.where(
        df_clean['duration_min'] <= free_minutes,
        base_price,
        base_price + (df_clean['duration_min'] - free_minutes) * overtime_rate
    )
    
    # --- 成本规则 (Cost) 预估 ---
    # 经典车：折旧与基础维护 (较低)
    classic_cost_per_ride = 0.30 
    # 电单车：折旧高 + 充电成本 + 人工换电调度 (极高)
    electric_cost_per_ride = 1.20 
    
    df_clean['Cost'] = np.where(df_clean['rideable_type'] == 'classic_bike', classic_cost_per_ride, electric_cost_per_ride)
    
    # --- 毛利计算 (Gross Profit) ---
    df_clean['Gross_Profit'] = df_clean['Revenue'] - df_clean['Cost']

    # ==========================================
    # 3. 聚合财务报表
    # ==========================================
    financial_report = df_clean.groupby('rideable_type').agg(
        Total_Rides=('ride_id', 'count'),
        Avg_Revenue=('Revenue', 'mean'), # 单均营收 (ARPU)
        Avg_Cost=('Cost', 'mean'),       # 单均成本
        Avg_Profit=('Gross_Profit', 'mean') # 单均毛利
    ).reset_index()
    
    # 计算整体毛利率 (Margin %)
    financial_report['Gross_Margin'] = financial_report['Avg_Profit'] / financial_report['Avg_Revenue']
    
    # 格式化车型名称展示
    financial_report['Vehicle_Type'] = financial_report['rideable_type'].str.replace('_', ' ').str.title()

    # 打印控制台财务摘要
    print("\n💰 [UE 测算结果] 资产单体经济模型:")
    print("-" * 50)
    for idx, row in financial_report.iterrows():
        print(f"[{row['Vehicle_Type']}] 总单量: {row['Total_Rides']:,}")
        print(f"   - 单均营收 (ARPU): ${row['Avg_Revenue']:.2f}")
        print(f"   - 单均成本 (Cost): ${row['Avg_Cost']:.2f}")
        print(f"   - 单均毛利 (Profit): ${row['Avg_Profit']:.2f} (毛利率 {row['Gross_Margin']:.1%})")
    print("-" * 50)

    # ==========================================
    # 4. 数据可视化：单体经济模型对比图
    # ==========================================
    # 使用堆叠柱状图的变体，展示 Revenue = Cost + Profit
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置柱子位置和宽度
    x = np.arange(len(financial_report))
    width = 0.4
    
    # 画底部的红色柱子：代表成本 (Cost)
    bars_cost = ax.bar(x, financial_report['Avg_Cost'], width, label='Avg Cost per Ride (成本)', color='#e74c3c', edgecolor='black')
    
    # 画上方的绿色柱子：代表毛利 (Gross Profit)，堆叠在成本之上
    # bottom 参数使其堆叠
    bars_profit = ax.bar(x, financial_report['Avg_Profit'], width, bottom=financial_report['Avg_Cost'], label='Avg Gross Profit (毛利)', color='#2ecc71', edgecolor='black')

    # 添加数值标签 (毛利)
    for i, bar in enumerate(bars_profit):
        profit_val = financial_report['Avg_Profit'].iloc[i]
        margin_pct = financial_report['Gross_Margin'].iloc[i]
        # 在绿色柱子中间写上利润金额和利润率
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                f'+${profit_val:.2f}\n({margin_pct:.0%})',
                ha='center', va='center', color='black', fontweight='bold', fontsize=11)

    # 添加数值标签 (成本)
    for i, bar in enumerate(bars_cost):
        cost_val = financial_report['Avg_Cost'].iloc[i]
        # 在红色柱子中间写上成本金额
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'-${cost_val:.2f}',
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        
    # 在柱子顶部标出总营收 (ARPU)
    for i in range(len(financial_report)):
        total_rev = financial_report['Avg_Revenue'].iloc[i]
        ax.text(x[i], total_rev + 0.1, f'Total ARPU: ${total_rev:.2f}', 
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)

    # 图表装饰
    ax.set_ylabel('Amount (USD)', fontweight='bold', fontsize=12)
    ax.set_title(f'Unit Economics: Profitability by Vehicle Type ({target_year}-{target_month:02d})', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(financial_report['Vehicle_Type'], fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    
    # 突出电单车成本高昂的商业洞察文本框
    max_y = financial_report['Avg_Revenue'].max()
    plt.annotate(
        "\n电单车虽营收能力强，但高昂的换电/折旧成本\n严重挤压了利润空间。经典车单均毛利更高。",
        xy=(1, financial_report['Avg_Revenue'].iloc[1]), # 假设电单车在索引1
        xytext=(0.5, max_y * 0.8),
        fontsize=11, fontweight='bold', color='darkred',
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffeaea", ec="red", lw=1.5, alpha=0.9)
    )

    # 5. 保存图表
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_suffix = f"{target_year}{target_month:02d}"
    save_path = os.path.join(output_dir, f"08_Unit_Economics_Margin_{file_suffix}.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   📈 财务测算图表已保存至: {save_path}")
    
    return financial_report

def analyze_station_kmeans_clustering(df, output_dir, target_year=2026, target_month=1):
    print(f"\n[Analysis - Machine Learning & GIS] Running Station Clustering & Map Generation ({target_year}-{target_month:02d})...")
    
    # 1. 获取数据与清洗
    df_target = filter_data_by_period(df, year=target_year, month=target_month)
    df_clean = df_target[df_target['start_station_name'].notna() & df_target['start_lat'].notna()].copy()
    
    if len(df_clean) == 0:
        print("   ⚠️ 警告：当前时间段没有有效站点数据，聚类终止。")
        return

    # ==========================================
    # 2. 特征工程 (同时提取经纬度用于地图)
    # ==========================================
    df_clean['hour'] = df_clean['started_at'].dt.hour
    df_clean['is_weekend'] = df_clean['started_at'].dt.dayofweek >= 5
    
    # 🚨 新增：在聚合时算出该站点的平均经纬度
    station_features = df_clean.groupby('start_station_name', observed=True).agg(
        Total_Rides=('ride_id', 'count'),
        Avg_Duration=('duration_min', 'mean'),
        Weekend_Rides=('is_weekend', 'sum'),
        AM_Peak_Rides=('hour', lambda x: ((x >= 7) & (x <= 9)).sum()),
        Lat=('start_lat', 'mean'),  # 提取纬度
        Lng=('start_lng', 'mean')   # 提取经度
    ).reset_index()
    
    station_features = station_features[station_features['Total_Rides'] >= 15].copy()
    station_features['Weekend_Ratio'] = station_features['Weekend_Rides'] / station_features['Total_Rides']
    station_features['AM_Peak_Ratio'] = station_features['AM_Peak_Rides'] / station_features['Total_Rides']
    
    features = ['Avg_Duration', 'Weekend_Ratio', 'AM_Peak_Ratio']
    X = station_features[features]

    # ==========================================
    # 3. K-Means 聚类与业务打标
    # ==========================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    station_features['Cluster'] = kmeans.fit_predict(X_scaled)

    centroids = station_features.groupby('Cluster')[features].mean()
    
    def assign_business_label(cluster_id):
        row = centroids.loc[cluster_id]
        if row['AM_Peak_Ratio'] == centroids['AM_Peak_Ratio'].max():
            return '🏢 核心通勤点 (早高峰潮汐极强)'
        elif row['Weekend_Ratio'] == centroids['Weekend_Ratio'].max() or row['Avg_Duration'] == centroids['Avg_Duration'].max():
            return '🌳 周末休闲点 (周末高频/骑行久)'
        else:
            return '🏠 均衡生活区 (日常散客/全天候)'

    station_features['Station_Persona'] = station_features['Cluster'].apply(assign_business_label)

    # ==========================================
    # 4. 生成 GIS 交互式地图 (Folium)
    # ==========================================
    print("   🗺️ Generating Interactive Folium Map...")
    
    # 创建地图，中心点设为所有站点的平均经纬度 (芝加哥)
    m = folium.Map(location=[station_features['Lat'].mean(), station_features['Lng'].mean()], 
                   zoom_start=11, 
                   tiles='CartoDB positron') # 使用清爽的底图风格

    # 为不同的业务画像分配颜色
    color_map = {
        '🏢 核心通勤点 (早高峰潮汐极强)': '#e74c3c', # 红色，表示急需早高峰调度的热点
        '🌳 周末休闲点 (周末高频/骑行久)': '#2ecc71', # 绿色，表示公园/休闲区
        '🏠 均衡生活区 (日常散客/全天候)': '#3498db'  # 蓝色，普通生活区
    }

    # 遍历每个站点，在地图上打点
    for idx, row in station_features.iterrows():
        # 根据单量大小决定圆圈半径，加上 max 和 min 防止圆圈过大或过小
        radius_size = min(max(row['Total_Rides'] / 50, 3), 15) 
        
        # 弹窗内容 (HTML格式)，业务人员点击圆点就能看到具体数据
        popup_html = f"""
        <div style="width: 200px;">
            <b>{row['start_station_name']}</b><br>
            <hr style="margin: 5px 0;">
            <b>属性:</b> {row['Station_Persona']}<br>
            <b>总单量:</b> {row['Total_Rides']} 单<br>
            <b>早高峰占比:</b> {row['AM_Peak_Ratio']:.1%}<br>
            <b>周末占比:</b> {row['Weekend_Ratio']:.1%}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lng']],
            radius=radius_size,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row['start_station_name'], # 鼠标悬停显示站名
            color=color_map.get(row['Station_Persona'], 'gray'),
            fill=True,
            fill_color=color_map.get(row['Station_Persona'], 'gray'),
            fill_opacity=0.7
        ).add_to(m)

    # 保存地图为 HTML 文件
    map_dir = os.path.join(output_dir, "maps")
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
        
    file_suffix = f"{target_year}{target_month:02d}"
    map_path = os.path.join(map_dir, f"09_Station_Map_{file_suffix}.html")
    m.save(map_path)
    print(f"   🗺️ Interactive Map saved: {map_path}")

    # (保留原本的散点图和Excel保存逻辑)
    # ... 省略绘图代码，和你之前的一样 ...
    _save(None, station_features, f"09_KMeans_Station_Clustering_{file_suffix}", output_dir)
